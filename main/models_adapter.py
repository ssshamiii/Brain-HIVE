from typing import Dict, List, Tuple, Union, Optional


import torch
import safetensors
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from .layers import FusionEncoder, ResNet1DLayer

logger = logging.get_logger(__name__)


class IPAttnProcessor2_0(nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        sequence_len (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        scale: Optional[float] = None,
    ):
        super().__init__()

        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale if scale is not None else 1.0

        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        ip_hidden_states: Tuple[torch.Tensor, torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            with torch.no_grad():
                self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPProjectionModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        embed_dim: int = 1024,
        cross_attention_dim: int = 2048,
        sequence_length: int = 4,
    ):
        super().__init__()

        self.proj = nn.Linear(embed_dim, cross_attention_dim * sequence_length)
        self.norm = nn.LayerNorm(cross_attention_dim)

        self.cross_attention_dim = cross_attention_dim

    @classmethod
    def from_ip_adapter(cls, ip_adapter_model_path: str, **kwargs):
        adapter_weights = safetensors.torch.load_file(ip_adapter_model_path)

        proj_weights = {
            "weight": adapter_weights["image_proj.proj.weight"],
            "bias": adapter_weights["image_proj.proj.bias"],
        }

        norm_weights = {
            "weight": adapter_weights["image_proj.norm.weight"],
            "bias": adapter_weights["image_proj.norm.bias"],
        }

        cross_attention_dim = norm_weights["weight"].shape[0]
        embed_dim = proj_weights["weight"].shape[1]
        sequence_length = int(proj_weights["bias"].shape[0] / cross_attention_dim)

        model = cls(
            embed_dim=embed_dim,
            cross_attention_dim=cross_attention_dim,
            sequence_length=sequence_length,
        )
        model.proj.load_state_dict(proj_weights)
        model.norm.load_state_dict(norm_weights)

        model.eval()

        del proj_weights, norm_weights

        return model

    def forward(self, x: torch.Tensor):
        bsz = x.shape[0]

        x = self.proj(x)
        x = self.norm(x.reshape(bsz, -1, self.cross_attention_dim))

        return x


class IPAttnAdapterModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        cross_attention_dim: int = 768,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        attn_processor_keys: Optional[Union[Tuple[str], List[str]]] = None,
        scale: Optional[Union[float, int]] = 1.0,
    ):
        super().__init__()
        if not isinstance(block_out_channels, tuple):
            block_out_channels = tuple(block_out_channels)
        self.block_out_channels = block_out_channels

        attn_processor_keys = tuple(attn_processor_keys)
        self.attn_processor_keys = attn_processor_keys

        self.scale = scale
        self.attn_processors = nn.ModuleDict()

        for index, name in enumerate(attn_processor_keys):
            # skip self attention
            if name.endswith("attn1.processor"):
                continue

            if name.startswith("mid_block"):
                hidden_size = block_out_channels[-1]
                block_id = None
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(block_out_channels))[block_id]

            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = block_out_channels[block_id]

            self.attn_processors[str(index)] = IPAttnProcessor2_0(
                hidden_size, cross_attention_dim, scale=scale
            )

    @classmethod
    def from_ip_adapter(
        cls,
        ip_adapter_model_path: str,
        block_out_channels: Tuple[int] = (320, 640, 1280),
        attn_processor_keys: Optional[Union[Tuple[str], List[str]]] = None,
        scale: Optional[Union[float, int]] = 1.0,
        **kwargs,
    ):
        adapter_weights = safetensors.torch.load_file(ip_adapter_model_path)

        cross_attention_dim = adapter_weights["image_proj.norm.weight"].shape[0]

        model = cls(
            cross_attention_dim=cross_attention_dim,
            block_out_channels=block_out_channels,
            attn_processor_keys=attn_processor_keys,
            scale=scale,
        )

        # load weights from ip adapter modules
        for index, name in enumerate(attn_processor_keys):
            # skip self attention
            if name.endswith("attn1.processor"):
                continue

            to_k_weights = {
                "weight": adapter_weights[f"ip_adapter.{index}.to_k_ip.weight"]
            }

            to_v_weights = {
                "weight": adapter_weights[f"ip_adapter.{index}.to_v_ip.weight"]
            }

            model.attn_processors[str(index)].to_k_ip.load_state_dict(to_k_weights)
            model.attn_processors[str(index)].to_v_ip.load_state_dict(to_v_weights)

        model.eval()
        del adapter_weights

        return model

    def bind_unet(self, unet: UNet2DConditionModel) -> Dict:
        attn_processors = unet.attn_processors

        for index, name in enumerate(attn_processors.keys()):
            if str(index) in self.attn_processors.keys():
                attn_processor = self.attn_processors[str(index)]
                attn_processors[name] = attn_processor

        unet.set_attn_processor(attn_processors)

        return attn_processors

    def unbind_unet(self, unet: UNet2DConditionModel) -> Dict:
        unet.set_attn_processor(AttnProcessor2_0())


class FusionEncoderModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        proj_meta: Dict[str, int] = None,
        num_layers: int = 1,
    ):
        super().__init__()

        self.vision_model = FusionEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            proj_meta=proj_meta,
            num_layers=num_layers,
        )

        self.adapter = ResNet1DLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            norm=False,
        )

    def forward(self, embs: Dict[str, torch.Tensor]):
        image_embeds = self.vision_model(**embs)
        return self.adapter(image_embeds)
