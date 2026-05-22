from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from capybara.layers import DownResBlock, UCountHead


class ProCapNet(nn.Module):
    """BPNet-style ProCapNet architecture used for PRO-cap profile/count prediction."""

    def __init__(
        self,
        *,
        n_filters: int = 512,
        n_layers: int = 8,
        n_outputs: int = 2,
        trimming: int = (2114 - 1000) // 2,
    ) -> None:
        super().__init__()
        self.n_filters = int(n_filters)
        self.n_layers = int(n_layers)
        self.n_outputs = int(n_outputs)
        self.trimming = int(trimming)

        self.iconv = nn.Conv1d(4, self.n_filters, kernel_size=21, padding=10)
        self.rconvs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.n_filters,
                    self.n_filters,
                    kernel_size=3,
                    padding=2**i,
                    dilation=2**i,
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(self.n_layers + 1)])
        self.profile_kernel_size = 75
        self.fconv = nn.Conv1d(self.n_filters, self.n_outputs, kernel_size=self.profile_kernel_size)
        self.linear = nn.Linear(self.n_filters, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, 4, length), got {tuple(x.shape)}")
        if x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 sequence channels, got shape {tuple(x.shape)}")

        start = self.trimming
        end = x.shape[2] - self.trimming

        x = self.relus[0](self.iconv(x))
        for i, conv in enumerate(self.rconvs):
            x_conv = self.relus[i + 1](conv(x))
            x = x + x_conv

        profile_features = x[:, :, start - self.profile_kernel_size // 2 : end + self.profile_kernel_size // 2]
        profile_logits = self.fconv(profile_features)
        log_counts = self.linear(profile_features.mean(dim=2)).reshape(x.shape[0], 1)
        return profile_logits, log_counts


class ProCapNetDownRes(nn.Module):
    """ProCapNet heads with a CAPY DownResBlock trunk and no pooling."""

    def __init__(
        self,
        *,
        n_filters: int = 512,
        n_layers: int = 8,
        n_outputs: int = 2,
        trimming: int = (2114 - 1000) // 2,
        trunk_channels: list[int] | None = None,
        norm_type: str = "batch",
        num_groups: int = 8,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_filters = int(n_filters)
        self.n_layers = int(n_layers)
        self.n_outputs = int(n_outputs)
        self.trimming = int(trimming)
        self.trunk_channels = [self.n_filters] * self.n_layers if trunk_channels is None else [int(ch) for ch in trunk_channels]
        if len(self.trunk_channels) == 0:
            raise ValueError("trunk_channels must contain at least one channel size.")

        self.iconv = nn.Conv1d(4, self.n_filters, kernel_size=21, padding=10)
        self.relu = nn.ReLU()

        blocks: list[nn.Module] = []
        in_channels = self.n_filters
        for out_channels in self.trunk_channels:
            if out_channels < in_channels:
                raise ValueError("ProCapNetDownRes trunk_channels must be non-decreasing.")
            blocks.append(
                DownResBlock(
                    in_channels,
                    out_channels,
                    norm_type=norm_type,
                    num_groups=int(num_groups),
                    activation=activation,
                    dropout=float(dropout),
                )
            )
            in_channels = out_channels
        self.downres_blocks = nn.ModuleList(blocks)

        self.profile_kernel_size = 75
        self.fconv = nn.Conv1d(in_channels, self.n_outputs, kernel_size=self.profile_kernel_size)
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, 4, length), got {tuple(x.shape)}")
        if x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 sequence channels, got shape {tuple(x.shape)}")

        start = self.trimming
        end = x.shape[2] - self.trimming

        x = self.relu(self.iconv(x))
        for block in self.downres_blocks:
            x = block(x)

        profile_features = x[:, :, start - self.profile_kernel_size // 2 : end + self.profile_kernel_size // 2]
        profile_logits = self.fconv(profile_features)
        log_counts = self.linear(profile_features.mean(dim=2)).reshape(x.shape[0], 1)
        return profile_logits, log_counts


class ProCapNetUCountHead(ProCapNet):
    """Dilated ProCapNet profile model with CAPY's configurable UCountHead."""

    def __init__(
        self,
        *,
        n_filters: int = 512,
        n_layers: int = 8,
        n_outputs: int = 2,
        trimming: int = (2114 - 1000) // 2,
        count_head: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(n_filters=n_filters, n_layers=n_layers, n_outputs=n_outputs, trimming=trimming)
        del self.linear
        count_cfg = {
            "conv_hidden_dims": [],
            "mlp_hidden_dims": [],
            "kernel_size": 5,
            "dropout": 0.0,
            "norm_type": "batch",
            "activation": "gelu",
            "mlp_norm_type": "layer",
            "num_groups": 8,
            "global_pool": "avg",
        }
        if count_head is not None:
            count_cfg.update(count_head)
        self.count_head = UCountHead(
            self.n_filters,
            conv_hidden_dims=list(count_cfg["conv_hidden_dims"]),
            mlp_hidden_dims=list(count_cfg["mlp_hidden_dims"]),
            kernel_size=int(count_cfg["kernel_size"]),
            dropout=float(count_cfg["dropout"]),
            norm_type=str(count_cfg["norm_type"]),
            activation=str(count_cfg["activation"]),
            mlp_norm_type=str(count_cfg["mlp_norm_type"]),
            num_groups=int(count_cfg["num_groups"]),
            global_pool=str(count_cfg["global_pool"]),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, 4, length), got {tuple(x.shape)}")
        if x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 sequence channels, got shape {tuple(x.shape)}")

        start = self.trimming
        end = x.shape[2] - self.trimming

        x = self.relus[0](self.iconv(x))
        for i, conv in enumerate(self.rconvs):
            x_conv = self.relus[i + 1](conv(x))
            x = x + x_conv

        profile_features = x[:, :, start - self.profile_kernel_size // 2 : end + self.profile_kernel_size // 2]
        profile_logits = self.fconv(profile_features)
        log_counts = self.count_head(profile_features)
        return profile_logits, log_counts


def build_procapnet_model(model_params: dict) -> nn.Module:
    params = dict(model_params)
    model_type = str(params.pop("model_type", "procapnet")).lower()
    if model_type in {"procapnet", "dilated"}:
        return ProCapNet(**params)
    if model_type in {"procapnet_downres", "downres"}:
        return ProCapNetDownRes(**params)
    if model_type in {"procapnet_ucount", "ucount"}:
        return ProCapNetUCountHead(**params)
    raise ValueError(f"Unsupported ProCapNet model_type: {model_type}")
