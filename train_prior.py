import os
import logging
import argparse
import shutil
import math
from tqdm import tqdm
from argparse import Namespace
from packaging import version

import torch
import diffusers
import transformers
import accelerate
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import PngImagePlugin
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, is_swanlab_available
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import free_memory
from diffusers.schedulers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers import StableDiffusionXLPipeline

from main.models_adapter import IPAttnAdapterModel, IPProjectionModel, FusionEncoderModel
from main.data import (
    load_embedding_datasets,
    load_image_dataset,
    build_emb_with_image_dataset,
)

MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

if is_wandb_available():
    import wandb

if is_swanlab_available():
    import swanlab

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0")
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for eeg2image task with very low resolution."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to the training config file.",
    )

    args = OmegaConf.load(parser.parse_args().config)

    return args


class FusionModel(torch.nn.Module):
    def __init__(
        self,
        fusion_encoder: FusionEncoderModel,
        proj: IPProjectionModel,
        attn_adapter: IPAttnAdapterModel,
        unet: UNet2DConditionModel,
    ):
        super().__init__()
        self.fusion_encoder = fusion_encoder

        self.proj = proj
        self.unet = unet
        self.attn_adapter = attn_adapter

        self.attn_adapter.bind_unet(self.unet)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs,
        image_embeds: dict[str, torch.Tensor],
    ):
        image_embeds = self.fusion_encoder(image_embeds)
        ip_hidden_states = self.proj(image_embeds)

        # Predict the noise residual
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs={"ip_hidden_states": ip_hidden_states},
            return_dict=False,
        )[0]

        return noise_pred


def log_validation(
    eval_dataloader: torch.utils.data.DataLoader,
    fusion_model: FusionEncoderModel,
    proj: IPProjectionModel,
    attn_adapter: IPAttnAdapterModel,
    vae: AutoencoderKL,
    args: Namespace,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    step: int,
):
    logger.info("Running validation...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.eval_diffusion_model_name_or_path, vae=vae
    )

    pipe.to(accelerator.device, weight_dtype)
    attn_adapter.bind_unet(pipe.unet)

    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device)

    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    image_logs = []
    for batch in eval_dataloader:
        image_embeds = batch["embs"]
        for key, tensor in image_embeds.items():
            image_embeds[key] = tensor.to(accelerator.device, weight_dtype)

        with torch.inference_mode():
            image_embeds = fusion_model(image_embeds)
            ip_hidden_states = proj(image_embeds.to(accelerator.device, weight_dtype))

        images = pipe(
            prompt=[""] * image_embeds.shape[0],
            num_images_per_prompt=1,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
            cross_attention_kwargs={"ip_hidden_states": ip_hidden_states},
            num_inference_steps=4,
            guidance_scale=0.0,
        ).images

        for image, pixel_value, image_id in zip(
            images, batch["pixel_values"], batch["image_ids"]
        ):
            image_logs.append(
                {
                    "image": image,
                    "reference": to_pil_image(pixel_value),
                    "name": image_id,
                }
            )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                imagegrid = [np.asarray(log["reference"]), np.asarray(log["image"])]
                imagegrid = np.stack(imagegrid)
                tracker.writer.add_images(
                    log["name"], imagegrid, step, dataformats="NHWC"
                )

        elif tracker.name == "wandb":
            for log in image_logs:
                log_imgs = [
                    wandb.Image(log["reference"], caption="reference"),
                    wandb.Image(log["image"], caption="image"),
                ]
                tracker.log({log["name"]: log_imgs})

        elif tracker.name == "swanlab":
            for log in image_logs:
                log_imgs = [
                    swanlab.Image(log["reference"], caption="reference"),
                    swanlab.Image(log["image"], caption="image"),
                ]
                tracker.log({log["name"]: log_imgs})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    # offload attn-adapter from unet
    attn_adapter.unbind_unet(pipe.unet)
    del pipe
    free_memory()

    vae.to(torch.float32)

    return image_logs


def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
        if accelerator.mixed_precision == "bf16":
            # Due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

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

    # If passed along, set the training seed now.
    if args.get("seed", None) is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.diffusion_model_name_or_path, subfolder="scheduler"
    )
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        args.diffusion_model_name_or_path, subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.diffusion_model_name_or_path, subfolder="unet"
    )

    attn_adapter = IPAttnAdapterModel.from_ip_adapter(
        args.ip_adapter_name_or_path,
        unet.config.block_out_channels,
        tuple(unet.attn_processors.keys()),
    )

    proj = IPProjectionModel.from_ip_adapter(args.ip_adapter_name_or_path)

    fusion_encoder = FusionEncoderModel(
        proj.config.embed_dim,
        proj_meta=OmegaConf.to_container(args.proj_meta),
    )
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, torch.float32)
    fusion_model = FusionModel(fusion_encoder, proj, attn_adapter, unet)
    fusion_model.to(accelerator.device, weight_dtype)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    if isinstance(
                        unwrap_model(model), type(unwrap_model(fusion_model))
                    ):
                        model = unwrap_model(model)
                        model.fusion_encoder.save_pretrained(
                            os.path.join(output_dir, "fusion_encoder")
                        )
                        model.proj.save_pretrained(os.path.join(output_dir, "proj"))
                        model.attn_adapter.save_pretrained(
                            os.path.join(output_dir, "attn_adapter")
                        )
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = FusionEncoderModel.from_pretrained(
                    input_dir, subfolder="fusion_encoder"
                )
                model.fusion_encoder.register_to_config(**load_model.config)
                model.fusion_encoder.load_state_dict(load_model.state_dict())

                load_model = IPProjectionModel.from_pretrained(
                    input_dir, subfolder="proj"
                )
                model.proj.register_to_config(**load_model.config)
                model.proj.load_state_dict(load_model.state_dict())

                load_model = IPAttnAdapterModel.from_pretrained(
                    input_dir, subfolder="attn_adapter"
                )
                model.attn_adapter.register_to_config(**load_model.config)
                model.attn_adapter.load_state_dict(load_model.state_dict())

                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    model_info = {
        "fusion_encoder": fusion_encoder.num_parameters(only_trainable=True),
        "proj": proj.num_parameters(only_trainable=True),
        "attn_adapter": attn_adapter.num_parameters(only_trainable=True),
    }
    logger.info(
        f"All models loaded successfully. Num trainable parameters = {model_info}"
    )
    params = filter(lambda p: p.requires_grad, fusion_model.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    # Prepare dataset and dataloader.
    # DataLoaders creation
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                args.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    image_ds = load_image_dataset(
        dataset_name=args.dataset_name,
        image_directory=args.image_directory,
        split="train",
        cache_dir=args.cache_dir,
    )
    emb_dss = load_embedding_datasets(
        embedding_directory=args.embedding_directory,
        dataset_key=args.dataset_name,
        split="train",
        model_keys=list(OmegaConf.to_container(args.proj_meta).keys()),
        cache_dir=args.cache_dir,
    )
    
    train_dataset = build_emb_with_image_dataset(
        image_ds=image_ds, emb_dss=emb_dss, preprocess=train_transform
    )

    def collate_fn(examples):
        # pixel values
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])  # [B,C,H,W]

        # SDXL add_time_ids (micro-conditioning)
        heights = [int(ex["height"]) for ex in examples]
        widths  = [int(ex["width"]) for ex in examples]

        add_time_ids = []
        for h, w in zip(heights, widths):
            crop_top  = max(0, int(round((h - args.resolution) / 2.0)))
            crop_left = max(0, int(round((w - args.resolution) / 2.0)))
            add_time_ids.append(
                torch.tensor([h, w, crop_top, crop_left, args.resolution, args.resolution], dtype=torch.long)
            )
        add_time_ids = torch.stack(add_time_ids, dim=0).contiguous()  # [B,6]

        # multi-emb
        emb_cols = [k for k in examples[0].keys() if k.startswith("emb_")]
        if not emb_cols:
            raise KeyError(f"No embedding columns found with prefix '{"emb_"}' in dataset examples.")

        embs = {}
        for col in emb_cols:
            name = col[len("emb_"):]
            embs[name] = torch.stack([ex[col] for ex in examples], dim=0).contiguous()  # [B,D]

        image_ids = [ex["image_id"] for ex in examples]

        return {
            "pixel_values": pixel_values,
            "add_time_ids": add_time_ids,
            "embs": embs,
            "image_ids": image_ids,
        }


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if accelerator.is_main_process and args.do_eval:
        eval_transform = transforms.Compose(
            [
                transforms.Resize(
                    args.resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
            ]
        )
        image_ds = load_image_dataset(
            dataset_name=args.eval_dataset_name,
            image_directory=args.eval_image_directory,
            split=args.eval_split,
            cache_dir=args.cache_dir,
        )
        emb_dss = load_embedding_datasets(
            embedding_directory=args.eval_embedding_directory,
            dataset_key=args.eval_dataset_name,
            split=args.eval_split,
            model_keys=list(OmegaConf.to_container(args.proj_meta).keys()),
            cache_dir=args.cache_dir,
        )
        eval_dataset = build_emb_with_image_dataset(
            image_ds=image_ds, emb_dss=emb_dss, preprocess=eval_transform
        )
        if args.max_eval_samples is not None:
            num_samples = min(args.max_eval_samples, len(eval_dataset))
            eval_dataset = eval_dataset.shuffle(args.seed).take(num_samples)

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    logger.info("Dataloader loaded successfully.")

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.get("max_steps", None) is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding
            / accelerator.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * num_update_steps_per_epoch
            * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    fusion_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        fusion_model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    if args.get("max_steps", None) is None:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
        if (
            num_training_steps_for_scheduler
            != args.max_steps * accelerator.num_processes
        ):
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_args = {}
        for k, v in OmegaConf.to_container(args).items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                tracker_args[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    tracker_args[f"{k}.{kk}"] = str(vv)
            else:
                tracker_args[k] = str(v)

        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config=tracker_args)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {args.max_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # get the fixed prompt embeds
    text_encoding_pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.diffusion_model_name_or_path,
        unet=None,
        vae=None,
    )
    text_encoding_pipeline.to(accelerator.device, weight_dtype)
    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = (
            text_encoding_pipeline.encode_prompt(
                "",
                device=accelerator.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        )

    text_encoding_pipeline.to("cpu")
    del text_encoding_pipeline
    free_memory()

    progress_bar = tqdm(
        range(0, args.max_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for _ in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        fusion_model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(fusion_model):
                # Convert images to latent space
                with torch.no_grad():
                    latents = (
                        vae.encode(
                            batch["pixel_values"].to(accelerator.device, torch.float32)
                        )
                        .latent_dist.sample()
                        .to(weight_dtype)
                    )
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                add_time_ids = batch["add_time_ids"].to(
                    accelerator.device, weight_dtype
                )
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)
                # Prepare condition embeds
                image_embeds = batch["embs"]
                for key, tensor in image_embeds.items():
                    image_embeds[key] = tensor.to(accelerator.device, weight_dtype)

                # Predict the noise residual and compute loss

                model_pred = fusion_model(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds.expand(bsz, -1, -1).to(
                        accelerator.device, weight_dtype
                    ),
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds.expand(bsz, -1).to(
                            accelerator.device, weight_dtype
                        ),
                        "time_ids": add_time_ids,
                    },
                    image_embeds=image_embeds,
                )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_device_train_batch_size)
                ).mean()

                train_loss += avg_loss.item() / accelerator.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": train_loss}

                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # Checkpointing
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `save_total_limit`
                        if args.save_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `save_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.save_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.save_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # Validation
                    if args.do_eval and global_step % args.validation_steps == 0:
                        log_validation(
                            eval_dataloader,
                            unwrap_model(fusion_model).fusion_encoder,
                            unwrap_model(fusion_model).proj,
                            unwrap_model(fusion_model).attn_adapter,
                            unwrap_model(vae),
                            args,
                            accelerator,
                            weight_dtype,
                            step=global_step,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = unwrap_model(fusion_model)

        if args.upcast_before_saving:
            unwrapped_model.fusion_encoder.to(torch.float32)
            unwrapped_model.proj.to(torch.float32)
            unwrapped_model.attn_adapter.to(torch.float32)

        unwrapped_model.fusion_encoder.save_pretrained(
            os.path.join(args.output_dir, "fusion_encoder")
        )
        unwrapped_model.proj.save_pretrained(os.path.join(args.output_dir, "proj"))
        unwrapped_model.attn_adapter.save_pretrained(
            os.path.join(args.output_dir, "attn_adapter")
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
