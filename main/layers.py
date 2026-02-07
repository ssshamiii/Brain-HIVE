from typing import List, Union, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BaseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.w1(x))
        return self.w2(self.dropout(h))


class ResNet1DLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        norm: bool = True,
    ):
        super().__init__()
        self.ff = BaseMLP(hidden_size, intermediate_size, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff(x) + x
        x = self.norm(x)
        return x


class ResNet1DBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResNet1DLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FusionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 1024,
        proj_meta: Dict[str, Union[int, List[int], Tuple[int, ...]]] = {},
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_meta = proj_meta

        self.proj_maps = nn.ModuleDict()

        for key, meta in proj_meta.items():
            self.proj_maps.update({key: nn.Linear(meta, hidden_size)})

        self.block = ResNet1DBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            num_layers=num_layers,
        )

    def forward(self, **x: torch.Tensor):
        results = []

        for key, proj in self.proj_maps.items():
            emb = x[key]
            # flatten vae latents
            if emb.ndim > 2:
                bsz = emb.shape[0]
                emb = emb.reshape(bsz, -1)

            proj_result = proj(emb)
            results.append(proj_result)

        x = torch.stack(results, dim=0).sum(dim=0)

        x = self.block(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, c_num: int = 17):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, c_num=17, **kwargs):
        super().__init__(PatchEmbedding(emb_size, c_num), FlattenHead())


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )
