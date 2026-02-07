from typing import Optional, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import ModelOutput, logging
from transformers.models.clip.modeling_clip import clip_loss

from .models_brain import BrainEncoder
from .layers import FusionEncoder

logger = logging.get_logger(__name__)


@dataclass
class CLIPOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits_per_image: Optional[torch.Tensor] = None
    logits_per_brain: Optional[torch.Tensor] = None
    brain_embeds: Optional[torch.Tensor] = None  # normed
    image_embeds: Optional[torch.Tensor] = None  # Fused image embeds, normed


class CLIPConfig(PretrainedConfig):
    model_type = "clip"

    def __init__(
        self,
        # common
        hidden_size: int = 1024,
        intermediate_size: int = 1024,
        logit_scale_init_value: float = 2.6592,
        dropout: float = 0.0,
        # brain
        brain_backbone: str = "brain_mlp",
        brain_channels: int = 17,
        brain_sequence_length: int = 250,
        brain_embedding_dim: int = 1440,
        # vision
        proj_meta: Dict[str, int] = {},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = int(hidden_size)
        self.logit_scale_init_value = float(logit_scale_init_value)
        self.dropout = float(dropout)

        self.brain_backbone = str(brain_backbone)
        self.brain_channels = int(brain_channels)
        self.brain_sequence_length = int(brain_sequence_length)
        self.brain_embedding_dim = int(brain_embedding_dim)

        self.proj_meta = dict(proj_meta)
        self.intermediate_size = intermediate_size


class BrainCLIPModel(PreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "brain_signals"
    _supports_flash_attn = False

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.brain_model = BrainEncoder(
            backbone=config.brain_backbone,
            brain_channels=config.brain_channels,
            brain_sequence_length=config.brain_sequence_length,
            hidden_size=config.hidden_size,
            embedding_dim=config.brain_embedding_dim,
            dropout=config.dropout,
        )

        self.vision_model = FusionEncoder(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=1,
            dropout=config.dropout,
            proj_meta=config.proj_meta,
        )

        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        self.post_init()

    def get_image_features(self, embs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        return self.vision_model(**embs)

    def get_brain_features(self, brain_signals: torch.Tensor, **kwargs):
        additional_brain_kwargs = {}
        if (
            self.config.brain_backbone == "atm"
            and (subject_ids := kwargs.pop("subject_ids", None)) is not None
        ):
            # for atm compatible
            additional_brain_kwargs.update(subject_ids=subject_ids)

        return self.brain_model(brain_signals, **additional_brain_kwargs)

    def forward(
        self, brain_signals: torch.Tensor, embs: Dict[str, torch.Tensor], **kwargs
    ) -> CLIPOutput:
        return_loss = kwargs.pop("return_loss", False)

        brain_embeds = self.get_brain_features(brain_signals, **kwargs)
        brain_embeds = brain_embeds / brain_embeds.norm(p=2, dim=-1, keepdim=True)

        image_embeds = self.get_image_features(embs)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_brain = torch.matmul(
            brain_embeds, image_embeds.t().to(brain_embeds.device)
        )
        logits_per_brain = logits_per_brain * self.logit_scale.exp().to(
            brain_embeds.device
        )

        logits_per_image = logits_per_brain.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_brain)

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_brain=logits_per_brain,
            brain_embeds=brain_embeds,
            image_embeds=image_embeds,
        )
