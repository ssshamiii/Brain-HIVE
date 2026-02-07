import os
import logging
import sys
from typing import Dict, List
from dataclasses import dataclass, field, asdict

import torch
import numpy as np
import transformers
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from transformers.trainer_utils import EvalPrediction

from main.data import (
    load_things_brain_dataset,
    load_embedding_datasets,
    build_brain_with_emb_dataset,
)
from main.models_clip import CLIPConfig, BrainCLIPModel
from main.models_adapter import FusionEncoderModel

logger = logging.getLogger(__name__)
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.3")
require_version("datasets>=4.4.1", "To fix: pip install -U datasets")


@dataclass
class ModelArguments:
    """CLIP model configs for training (aligned with CLIPConfig)."""

    # --- common ---
    hidden_size: int = field(
        default=1024,
        metadata={"help": "Shared embedding width for brain/vision projections (d)."},
    )

    intermediate_size: int = field(
        default=1024, metadata={"help": "intermediate hidden size of the mlps"}
    )

    logit_scale_init_value: float = field(
        default=2.6592,
        metadata={
            "help": "Initial value for logit_scale (stored in log-space). "
            "tau_init = 1/exp(logit_scale_init_value). "
            "For tau_init=0.07, use ln(1/0.07)=2.6592."
        },
    )

    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout used in projection / MLP blocks (if applicable)."},
    )

    # --- brain ---
    brain_backbone: str = field(
        default="brain_mlp",
        metadata={
            "help": "Brain encoder backbone name. "
            "E.g. 'brain_projection', 'eeg_net', 'ts_conv', 'nice', 'shallow_net', 'deep_net', 'brain_mlp'."
        },
    )

    brain_channels: int = field(
        default=17,
        metadata={
            "help": "Number of EEG/MEG channels fed to brain encoder (e.g., THINGS-EEG O+P=17)."
        },
    )

    brain_sequence_length: int = field(
        default=250,
        metadata={
            "help": "Temporal length (T) of the brain sequence after preprocessing/downsampling."
        },
    )

    brain_embedding_dim: int = field(
        default=1440,
        metadata={
            "help": "Backbone-specific intermediate embedding dim (used by some EEG backbones, e.g., NICE/TSConv). "
            "For 'brain_projection' it may be unused, but kept for config completeness."
        },
    )

    # --- vision ---
    # proj_meta: map from each vision encoder key -> its raw embedding dim BEFORE projecting to hidden_size.
    proj_meta: Dict[str, int] = field(
        default_factory=lambda: {
            "RN50": 1024,  # CLIP RN50 pooled proj dim is typically 1024
            "CLIP-ViT-B-32-laion2B-s34B-b79K": 512,  # CLIP ViT-B/32 pooled proj dim is typically 512
            "vae": 1024,  # SDXL VAE latent flattened dim for 128x128: 16*16*4=1024
        },
        metadata={
            "help": "Dict[str,int]. Vision embedding dims for each encoder branch. "
            "Example (retrieval): {'RN50':1024,'CLIP-ViT-B-32-laion2B-s34B-b79K':512,'vae':1024}. "
        },
    )

    pretrained_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Load pretrained Fusion Prior as the vision part"},
    )


@dataclass
class DataArguments:
    """Dataset / IO configs."""

    embedding_directory: str = field(
        default=None,
        metadata={"help": "Train embeddings dir."},
    )

    data_directory: str = field(
        default=None,
        metadata={
            "help": "Root directory of the paired brain-vision dataset (signals, ids, optional images)."
        },
    )

    subject_ids: List[int] = field(
        default_factory=lambda: [
            1,
        ],
        metadata={
            "help": "Subject IDs used for training split (e.g., THINGS-EEG subjects)."
        },
    )

    eval_subject_ids: List[int] = field(
        default_factory=lambda: [
            1,
        ],
        metadata={"help": "Subject IDs used for evaluation split. Keep length == 1."},
    )

    brain_key: str = field(
        default="eeg",
        metadata={
            "help": "Key name of brain signal in dataset items (e.g., 'eeg', 'meg')."
        },
    )

    time_slice: List[int] = field(
        default_factory=lambda: [0, 250],
        metadata={
            "help": "Brain temporal window [start, end) used for slicing (in samples) before feeding the backbone."
        },
    )

    avg_trials: bool = field(
        default=True,
        metadata={
            "help": "Whether to average repeated trials per image/condition before training."
        },
    )

    selected_channels: List[str] = field(
        default=None,
        metadata={
            "help": "Optional channel selection spec (e.g., comma-separated indices or a named preset)."
        },
    )

    cache_dir: str = field(
        default=".cache",
        metadata={"help": "Cache dir."},
    )

    max_train_samples: int = field(
        default=None,
        metadata={"help": "Max train samples for debugging."},
    )

    max_eval_samples: int = field(
        default=None,
        metadata={"help": "Max eval samples for debugging."},
    )


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and (
        sys.argv[1].endswith(".json") or sys.argv[1].endswith((".yaml", ".yml"))
    ):
        # If we pass only one argument to the script and it's the path to a json/yaml file,
        # let's parse it to get our arguments.
        if sys.argv[1].endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            model_args, data_args, training_args = parser.parse_yaml_file(
                yaml_file=os.path.abspath(sys.argv[1])
            )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 4. Load model
    model_cfg = asdict(model_args)
    pretrained_model_name_or_path = model_cfg.pop("pretrained_model_name_or_path", None)
    # overwrite part of the config using pretrained model
    if pretrained_model_name_or_path is not None:
        pretrained_cfg = FusionEncoderModel.load_config(
            pretrained_model_name_or_path, subfolder="fusion_encoder"
        )
        model_cfg.update(
            **{key: value for key, value in pretrained_cfg.items() if key in model_cfg}
        )

    model_cfg = CLIPConfig.from_dict(model_cfg)
    model = BrainCLIPModel(model_cfg)

    if pretrained_model_name_or_path is not None:
        model.vision_model.load_state_dict(
            FusionEncoderModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="fusion_encoder"
            ).vision_model.state_dict()
        )
        model.vision_model.requires_grad_(False)

    logger.info(
        f"Number of parameters trainable: {model.num_parameters(only_trainable=True)}"
    )

    # set seed for torch dataloaders
    set_seed(int(training_args.seed))

    # 5. Load dataset
    if training_args.do_train:
        brain_ds = load_things_brain_dataset(
            data_directory=data_args.data_directory,
            split="train",
            subject_ids=data_args.subject_ids,
            brain_key=data_args.brain_key,
            avg_trials=data_args.avg_trials,
            selected_channels=data_args.selected_channels,
        )
        emb_dss = load_embedding_datasets(
            embedding_directory=data_args.embedding_directory,
            dataset_key="things",
            split="train",
            model_keys=list(model_args.proj_meta.keys()),
            cache_dir=data_args.cache_dir,
        )
        train_dataset = build_brain_with_emb_dataset(brain_ds=brain_ds, emb_dss=emb_dss)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        brain_ds = load_things_brain_dataset(
            data_directory=data_args.data_directory,
            split="test",
            subject_ids=data_args.eval_subject_ids,
            brain_key=data_args.brain_key,
            avg_trials=data_args.avg_trials,
            selected_channels=data_args.selected_channels,
        )
        emb_dss = load_embedding_datasets(
            embedding_directory=data_args.embedding_directory,
            dataset_key="things",
            split="test",
            model_keys=list(model_args.proj_meta.keys()),
            cache_dir=data_args.cache_dir,
        )
        eval_dataset = build_brain_with_emb_dataset(brain_ds=brain_ds, emb_dss=emb_dss)
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    def _stack_or_tensorize(vals, *, dtype=torch.float32):
        if isinstance(vals[0], torch.Tensor):
            return torch.stack(vals, dim=0).contiguous()
        return torch.tensor(vals, dtype=dtype).contiguous()

    def collate_fn(batch):
        # brain: [B,C,T]
        emb_prefix = "emb_"
        brain_vals = [b[data_args.brain_key] for b in batch]
        brain = _stack_or_tensorize(brain_vals, dtype=torch.float32)

        if data_args.time_slice is not None:
            t0, t1 = data_args.time_slice
            brain = brain[:, :, t0:t1].contiguous()

        # multi-emb
        emb_cols = [k for k in batch[0].keys() if k.startswith(emb_prefix)]
        if not emb_cols:
            raise KeyError(f"No embedding columns found with prefix '{emb_prefix}'")

        embs = {}
        for col in emb_cols:
            name = col[len(emb_prefix) :]
            embs[name] = _stack_or_tensorize(
                [b[col] for b in batch], dtype=torch.float32
            )  # [B,D]

        subject_ids = torch.tensor(
            [b["subject_id"] for b in batch], dtype=torch.long
        ).contiguous()
        image_ids = [b["image_id"] for b in batch]

        out = {
            "brain_signals": brain,
            "embs": embs,
            "subject_ids": subject_ids,
            "image_ids": image_ids,
        }
        out.update({"return_loss": True})
        return out

    def metric_fn(predictions: EvalPrediction):
        logits_per_brain = predictions.predictions[1]
        logits = np.asarray(logits_per_brain)

        N = logits.shape[0]

        labels = np.arange(N, dtype=np.int64)

        # top-k indices (descending by logit)
        top1_idx = np.argmax(logits, axis=1)
        top5_idx = np.argsort(-logits, axis=1)[:, :5]

        top1_acc = float(np.mean(top1_idx == labels))
        top5_acc = float(np.mean(np.any(top5_idx == labels[:, None], axis=1)))

        return {"top1_acc": top1_acc, "top5_acc": top5_acc}

    # 6. Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=metric_fn if training_args.do_eval else None,
    )

    # 7. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 8. Evaluation
    if training_args.do_eval and training_args.do_predict:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
