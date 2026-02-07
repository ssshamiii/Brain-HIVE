import os
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Union

from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from PIL import PngImagePlugin
from torch.utils.data import DataLoader

import open_clip
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, broadcast_object_list
from accelerate.logging import get_logger
from diffusers.training_utils import free_memory, set_seed
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    Dinov2Model,
    BitImageProcessor,
)
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from main.data import load_image_dataset, ensure_image_id
from main.cache import (
    list_embedding_parquets,
    canonical_is_complete,
    delete_files,
    canonicalize_rank_shards,
    RollingParquetEmbeddingWriter,
)



logger = get_logger(__name__)

# Prevent PIL text-chunk bomb
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


# =========================================================
# Args
# =========================================================
def parse_args() -> SimpleNamespace:
    p = argparse.ArgumentParser("Build embedding cache (parquet shards)")
    p.add_argument("--config_file", type=str, required=True)
    cli = p.parse_args()

    cfg = OmegaConf.load(cli.config_file)
    return SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))


# =========================================================
# Encoder bundle (moved back into this script)
# =========================================================
class VAEWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)

    def forward(self, pixel_values: torch.Tensor):
        bsz = pixel_values.shape[0]
        
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return (latents.reshape(bsz, -1),)


class OpenCLIPWrapper(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.visual: torch.nn.Module = clip_model.visual
        self.visual.eval()

    def forward(self, pixel_values: torch.Tensor):
        emb = self.visual(pixel_values)
        return (emb,)


class SynclrWrapper(torch.nn.Module):
    def __init__(self, vit: torch.nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, pixel_values: torch.Tensor):
        # timm ViT: forward_features returns [B, N, D], take CLS token
        feats = self.vit.forward_features(pixel_values)[:, 0]
        return (feats,)


def vit_base_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )


def vit_large_patch14(**kwargs):
    return VisionTransformer(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )


def _strip_synclr_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module.visual."):
            out[k.replace("module.visual.", "")] = v
        else:
            out[k] = v
    return out


def load_synclr_vit(pretrained_model_name_or_path: str) -> torch.nn.Module:
    model_name = Path(pretrained_model_name_or_path).stem
    ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(
            f"SynCLR checkpoint missing key 'model': {pretrained_model_name_or_path}"
        )
    state = _strip_synclr_prefix(ckpt["model"])

    if model_name == "synclr_vit_b_16":
        model = vit_base_patch16(num_classes=0)
    elif model_name == "synclr_vit_l_14":
        model = vit_large_patch14(num_classes=0)
    else:
        raise ValueError(f"Unsupported SynCLR model name: {model_name}")

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@dataclass
class EncoderBundle:
    encoder: torch.nn.Module
    preprocess: Union[Callable, CLIPImageProcessor, BitImageProcessor]
    output_key: int  # kept for backward-compat; but we also try .image_embeds/.pooler_output first


def build_encoder_bundle(
    *,
    pretrained_model_name_or_path: str,
    resolution: int = 224,
    cache_dir: str = ".cache",
) -> EncoderBundle:
    """
    Minimal + practical:
      - dinov2 -> BitImageProcessor + use pooler_output (fallback index)
      - vae    -> torchvision transform -> latents
      - synclr -> torchvision transform -> cls features
      - open_clip local weights -> open_clip.create_model_and_transforms
      - default -> HF CLIPVisionModelWithProjection
    """
    output_key = 0

    if "dinov2" in pretrained_model_name_or_path:
        encoder = Dinov2Model.from_pretrained(pretrained_model_name_or_path)
        preprocess = BitImageProcessor.from_pretrained(pretrained_model_name_or_path)
        output_key = 1
        return EncoderBundle(
            encoder=encoder, preprocess=preprocess, output_key=output_key
        )

    if "vae" in pretrained_model_name_or_path:
        encoder = VAEWrapper(pretrained_model_name_or_path)
        preprocess = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return EncoderBundle(
            encoder=encoder, preprocess=preprocess, output_key=output_key
        )

    if "synclr" in pretrained_model_name_or_path:
        encoder = SynclrWrapper(load_synclr_vit(pretrained_model_name_or_path))
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        return EncoderBundle(
            encoder=encoder, preprocess=preprocess, output_key=output_key
        )

    # open_clip local checkpoint path (RN50/RN101 style) – keep your old behavior
    if os.path.isfile(pretrained_model_name_or_path) and any(
        k in pretrained_model_name_or_path for k in ("RN50", "RN101")
    ):
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            Path(pretrained_model_name_or_path).stem,
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            weights_only=False,
        )
        encoder = OpenCLIPWrapper(clip_model)
        return EncoderBundle(
            encoder=encoder, preprocess=preprocess, output_key=output_key
        )

    # default: HF CLIP vision
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path
    )
    preprocess = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)
    return EncoderBundle(encoder=encoder, preprocess=preprocess, output_key=output_key)


def make_collate_fn(
    preprocess: Union[Callable, CLIPImageProcessor, BitImageProcessor],
) -> Callable[[List[dict]], Dict[str, object]]:
    """
    Input examples from HF dataset:
      - image: PIL.Image
      - image_id: str
    Output:
      - model_inputs: torch.Tensor [B,C,H,W]
      - image_ids: List[str]
    """

    def collate(examples: List[dict]) -> Dict[str, object]:
        image_ids = [ex["image_id"] for ex in examples]
        images = [ex["image"].convert("RGB") for ex in examples]

        if isinstance(preprocess, (CLIPImageProcessor, BitImageProcessor)):
            out = preprocess(images=images, return_tensors="pt")
            model_inputs = out["pixel_values"]
        else:
            model_inputs = torch.stack([preprocess(img) for img in images], dim=0)

        return {"model_inputs": model_inputs.contiguous(), "image_ids": image_ids}

    return collate


def extract_embeddings(
    encoder: torch.nn.Module,
    x: torch.Tensor,
    *,
    output_key: int,
) -> torch.Tensor:
    """
    Robustly read embedding from different model output types.
    """
    out = encoder(x)

    # HF models often return ModelOutput with attributes
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output

    # tuple-like fallback
    if isinstance(out, (tuple, list)):
        return out[output_key]

    # last fallback: allow tensor
    if torch.is_tensor(out):
        return out

    raise TypeError(f"Unsupported encoder output type: {type(out)}")


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(project_config=accelerator_project_config)

    # logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # base name for parquet files
    model_key = Path(args.pretrained_model_name_or_path).stem
    base_name = f"{args.dataset_name}_{args.split}_{model_key}"

    # -----------------------------
    # Early skip / cleanup (main proc)
    # -----------------------------
    action = {"mode": "run"}  # run | skip
    if accelerator.is_main_process:
        canon, rankish = list_embedding_parquets(out_dir, base_name)

        if canonical_is_complete(canon):
            logger.info(
                f"[cache] Found complete canonical shards for {base_name}. Skip."
            )
            action["mode"] = "skip"
            if rankish:
                logger.warning(
                    f"[cache] Removing {len(rankish)} leftover rank shards (already complete)."
                )
                delete_files(rankish)
        else:
            # remove interrupted leftovers
            if rankish:
                logger.warning(
                    f"[cache] Found {len(rankish)} rank shards but no complete canonical set. "
                    f"Assume interrupted run; deleting and restarting."
                )
                delete_files(rankish)
            if canon:
                logger.warning(
                    f"[cache] Found {len(canon)} canonical shards but incomplete/corrupted. "
                    f"Deleting and restarting."
                )
                delete_files(canon)

    # broadcast decision
    obj = [action]
    broadcast_object_list(obj, from_process=0)
    action = obj[0]

    if action["mode"] == "skip":
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    # -----------------------------
    # 1) Load HF dataset (image + image_id)
    # -----------------------------
    ds = load_image_dataset(
        dataset_name=args.dataset_name,
        image_directory=args.image_directory,
        split=args.split,
        cache_dir=getattr(args, "cache_dir", ".cache"),
    )
    ds = ensure_image_id(ds)

    # shard per rank
    world = accelerator.num_processes
    rank = accelerator.process_index
    ds = ds.shard(num_shards=world, index=rank, contiguous=True)

    # -----------------------------
    # 2) Build encoder + preprocess (now local in this script)
    # -----------------------------
    bundle = build_encoder_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        resolution=int(args.resolution),
        cache_dir=args.cache_dir,
    )
    encoder = bundle.encoder
    preprocess = bundle.preprocess
    output_key = bundle.output_key

    # dtype selection
    weight_dtype = torch.float32
    mp = accelerator.mixed_precision
    if mp == "fp16" and ("vae" not in args.pretrained_model_name_or_path):
        weight_dtype = torch.float16
    elif mp == "bf16" and ("vae" not in args.pretrained_model_name_or_path):
        weight_dtype = torch.bfloat16

    encoder.to(accelerator.device, dtype=weight_dtype)
    encoder.eval()

    # -----------------------------
    # 3) DataLoader + collate
    # -----------------------------
    collate_fn = make_collate_fn(preprocess)
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # -----------------------------
    # 4) Infer embedding dim (first batch)
    # -----------------------------
    first = next(iter(dl))
    with torch.inference_mode():
        x = first["model_inputs"].to(accelerator.device, dtype=weight_dtype)
        y = extract_embeddings(encoder, x, output_key=output_key)
        y = y.float().detach().cpu()
    emb_dim = int(y.shape[-1])
    del first, x, y
    free_memory()

    # -----------------------------
    # 5) Streaming parquet writer (rank-style)
    # -----------------------------
    writer = RollingParquetEmbeddingWriter(
        out_dir=str(out_dir),
        base_name=base_name,
        rank=rank,
        world=world,
        dim_map={"emb": emb_dim},
        dtype=getattr(args, "dtype", "float16"),
        compression="zstd",
        max_rows_per_file=getattr(args, "max_rows_per_file", 200_000),
    )

    # -----------------------------
    # 6) Loop + flush
    # -----------------------------
    rows_per_flush = int(getattr(args, "rows_per_flush", 4096))
    buf_ids: List[str] = []
    buf_emb: List[torch.Tensor] = []

    pbar = tqdm(
        dl, desc="building embeddings", disable=not accelerator.is_local_main_process
    )
    for batch in pbar:
        model_inputs = batch["model_inputs"].to(accelerator.device)
        if model_inputs.dtype != weight_dtype:
            model_inputs = model_inputs.to(weight_dtype)

        image_ids = batch["image_ids"]

        with torch.inference_mode():
            image_embeds = extract_embeddings(
                encoder, model_inputs, output_key=output_key
            )
            image_embeds = image_embeds.float().detach().cpu()

        buf_ids.extend(image_ids)
        buf_emb.append(image_embeds)

        if len(buf_ids) >= rows_per_flush:
            embs = torch.cat(buf_emb, dim=0)
            writer.write({"image_id": buf_ids, "emb": embs})
            logger.info(f"Rank {rank}: flushed {len(buf_ids)} rows")
            buf_ids.clear()
            buf_emb.clear()
            free_memory()

    if buf_ids:
        embs = torch.cat(buf_emb, dim=0)
        writer.write({"image_id": buf_ids, "emb": embs})
        logger.info(f"Rank {rank}: final flush {len(buf_ids)} rows")

    writer.close()
    accelerator.wait_for_everyone()

    # -----------------------------
    # 7) Canonicalize shards on main proc
    # -----------------------------
    if accelerator.is_main_process:
        finals = canonicalize_rank_shards(
            out_dir, base_name, delete_existing_canon=False
        )
        logger.info(
            f"[cache] canonical shards = {len(finals)} for base_name={base_name}"
        )

    encoder.cpu()
    del encoder
    free_memory()
    accelerator.end_training()


if __name__ == "__main__":
    main()
