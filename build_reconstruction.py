import os
import json
import math
import logging
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import transformers
import diffusers
from omegaconf import OmegaConf
from torchvision import transforms
from diffusers.utils import check_min_version
from diffusers.training_utils import free_memory, set_seed
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionXLPipeline

from main.models_clip import BrainCLIPModel
from main.models_adapter import IPProjectionModel, IPAttnAdapterModel
from main.data import (
    load_things_brain_dataset,
    load_image_dataset,
    build_brain_with_image_dataset,
)
from main.utils_eval import eval_images

check_min_version("0.36.0")
logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser(description="Image Embedding Generation")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to the training config file.",
    )

    args = OmegaConf.load(parser.parse_args().config)

    return args


def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(project_config=accelerator_project_config)

    if args.seed is not None:
        set_seed(args.seed)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
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

    image_saving_directory = os.path.join(args.output_dir, "images")
    if accelerator.is_main_process:
        os.makedirs(image_saving_directory, exist_ok=True)

    accelerator.wait_for_everyone()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    clip_model = BrainCLIPModel.from_pretrained(args.brain_model_name_or_path)

    pipe = StableDiffusionXLPipeline.from_pretrained(args.diffusion_model_name_or_path)

    attn_adapter = IPAttnAdapterModel.from_pretrained(
        args.prior_model_name_or_path, subfolder="attn_adapter"
    )
    attn_adapter.bind_unet(pipe.unet)

    proj = IPProjectionModel.from_pretrained(
        args.prior_model_name_or_path, subfolder="proj"
    )

    proj.to(accelerator.device, weight_dtype)
    clip_model.to(accelerator.device, weight_dtype)
    pipe.to(accelerator.device, weight_dtype)

    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)

    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    logger.info("All models loaded successfully.")

    eval_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )
    brain_ds = load_things_brain_dataset(
        data_directory=args.data_directory,
        split="test",
        subject_ids=args.eval_subject_ids,
        brain_key=args.brain_key,
        selected_channels=args.selected_channels,
    )
    image_ds = load_image_dataset(
        dataset_name="things",
        image_directory=args.image_directory,
        split="test",
        cache_dir=args.cache_dir,
    )
    eval_dataset = build_brain_with_image_dataset(
        brain_ds=brain_ds, image_ds=image_ds, preprocess=eval_transforms
    )

    def collate_fn(batch):
        brain = torch.tensor(
            [b[args.brain_key] for b in batch], dtype=torch.float32
        )  # [B,C,T]
        if args.time_slice is not None:
            t0, t1 = args.time_slice
            brain = brain[:, :, t0:t1].contiguous()

        pixel_values = torch.stack(
            [b["pixel_values"] for b in batch], dim=0
        ).contiguous()
        subject_id = torch.tensor(
            [b["subject_id"] for b in batch], dtype=torch.long
        ).contiguous()
        image_id = [b["image_id"] for b in batch]

        out = {
            "brain_signals": brain,
            "pixel_values": pixel_values,
            "subject_ids": subject_id,
            "image_ids": image_id,
        }
        return out

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=accelerator.num_processes,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
    )

    logger.info("Dataloader loaded successfully.")

    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        desc="steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    text_encoding_pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.diffusion_model_name_or_path, unet=None, vae=None
    )
    text_encoding_pipeline.to(accelerator.device, weight_dtype)

    with torch.inference_mode():
        (
            prompt_embeds,
            _,
            pooled_prompt_embeds,
            _,
        ) = text_encoding_pipeline.encode_prompt(
            "",  # we use empty prompt for inference
            device=accelerator.device,
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
        )

    text_encoding_pipeline.to("cpu")
    del text_encoding_pipeline
    free_memory()

    logger.info("Prompt embeds loaded successfully")

    refer_images = []
    recon_images = []

    for batch in dataloader:
        with accelerator.split_between_processes(batch) as distributed_batch:
            brain_signals = distributed_batch["brain_signals"].to(
                accelerator.device, weight_dtype
            )
            subject_ids = distributed_batch["subject_ids"].to(accelerator.device)
            image_id = distributed_batch["image_ids"][0]

            with torch.inference_mode():
                brain_embeds = clip_model.get_brain_features(
                    brain_signals, subject_ids=subject_ids
                )
                ip_hidden_states = proj(brain_embeds)

            refer_image = distributed_batch["pixel_values"][0]

            images = pipe(
                height=args.resolution,
                width=args.resolution,
                generator=generator,
                prompt_embeds=prompt_embeds.expand(args.num_repeats, -1, -1),
                pooled_prompt_embeds=pooled_prompt_embeds.expand(args.num_repeats, -1),
                cross_attention_kwargs={
                    "ip_hidden_states": ip_hidden_states.expand(
                        args.num_repeats, -1, -1
                    )
                },
                num_inference_steps=4,
                guidance_scale=0.0,
                output_type="pil" if args.save_output else "pt",
            ).images

            if args.save_output:
                for i, image in enumerate(images):
                    save_path = os.path.join(
                        image_saving_directory, f"{image_id}-{i}.png"
                    )
                    image.save(save_path)

                images = [
                    eval_transforms(image).float().cpu() for image in images
                ]  # convert back to pt format

            refer_images.append(refer_image)
            recon_images.extend(images)

        progress_bar.update(1)

    del pipe, clip_model, proj

    accelerator.wait_for_everyone()

    local_refer_images = torch.stack(
        [img.detach().cpu() for img in refer_images], dim=0
    )
    local_recon_images = torch.stack(
        [img.detach().cpu() for img in recon_images], dim=0
    )

    local_refer_images = local_refer_images.to(accelerator.device)
    local_recon_images = local_recon_images.to(accelerator.device)

    all_ref_images = accelerator.gather(local_refer_images)
    all_recon_images = accelerator.gather(local_recon_images)

    if accelerator.is_main_process:
        metrics = []
        logger.info("Runing evaluation...")
        for i in range(args.num_repeats):
            eval_results = eval_images(
                all_ref_images,
                all_recon_images[i :: args.num_repeats],
                device=accelerator.device,
            )
            metrics.append(eval_results)

        sums = defaultdict(float)
        counts = defaultdict(int)

        for data in metrics:
            for k, v in data.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    sums[k] += float(v)
                    counts[k] += 1

        save_path = os.path.join(args.output_dir, "eval_results.json")

        # avg_metric is the metrics computed in this run (already averaged over num_repeats)
        avg_metric = {k: (sums[k] / counts[k]) for k in sums.keys() if counts[k] > 0}

        # merge: keep old keys, but let current run override on conflicts
        if os.path.exists(save_path):
            try:
                with open(save_path, "r") as fp:
                    origin_metrics = json.load(fp)
                if isinstance(origin_metrics, dict):
                    origin_metrics.update(avg_metric)  # NEW overwrites OLD
                    avg_metric = origin_metrics
            except Exception:
                # if the old file is corrupted / unreadable, just write current metrics
                pass

        with open(save_path, "w") as fp:
            json.dump(avg_metric, fp)
        logger.info(avg_metric)

    free_memory()
    accelerator.end_training()


if __name__ == "__main__":
    main()
