from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY


class BiasFactorizedCAPY(nn.Module):
    """ChromBPNet's bias-factorized composition, with CAPY in both slots.

    Mirrors ``training/models/chrombpnet_with_bias_model.py``:

    * ``profile_logits = acc_logits + bias_logits``  (element-wise add of logits)
    * ``log_counts     = logsumexp([acc_logcount, bias_logcount])``  (counts add in
      linear space)

    The (depth-scaled) bias branch is **frozen**: its parameters get
    ``requires_grad=False`` and it is held in ``.eval()`` at all times so its
    BatchNorm running statistics never update while the accessibility branch
    trains. ``forward`` returns the same ``(profile_logits, log_counts)`` 2-tuple
    as a bare :class:`CAPY`, so it drops into ``compute_losses`` / ``validate`` /
    ``run_model`` unchanged.
    """

    def __init__(self, accessibility: CAPY, bias: CAPY) -> None:
        super().__init__()
        self.accessibility = accessibility
        self.bias = bias
        for parameter in self.bias.parameters():
            parameter.requires_grad = False
        self.bias.eval()

    def train(self, mode: bool = True) -> "BiasFactorizedCAPY":
        """Set train/eval mode on the accessibility branch only; bias stays eval."""
        self.training = mode
        self.accessibility.train(mode)
        self.bias.eval()
        return self

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        acc_logits, acc_logcount = self.accessibility(x)
        with torch.no_grad():
            bias_logits, bias_logcount = self.bias(x)

        profile_logits = acc_logits + bias_logits
        stacked = torch.stack([acc_logcount, bias_logcount], dim=-1)  # (B, 1, 2)
        log_counts = torch.logsumexp(stacked, dim=-1)                 # (B, 1)
        return profile_logits, log_counts


def _self_test() -> None:
    """Assert the composition math element-wise against the two branches."""
    import numpy as np

    torch.manual_seed(0)
    cfg = {
        "model": {
            "input_length": 2114,
            "output_length": 1000,
            "encoder_channels": [32, 48],
            "decoder_channels": [48, 32, 32],
            "dna_embedder_channels": 16,
            "output_embedder_channels": 16,
            "embedding_projector": {"enabled": False},
            "profile_head": {"source": "decoder", "num_outputs": 1, "kernel_size": 75},
            "count_head": {"type": "ynet", "source": "bottleneck", "conv_hidden_dims": [16], "mlp_hidden_dims": [16]},
            "bottleneck": {"type": "residual_conv", "depth": 1, "kernel_size": 5},
        },
        "dataset": {"input_length": 2114, "output_length": 1000},
    }
    acc = CAPY(cfg)
    bias = CAPY(cfg)
    model = BiasFactorizedCAPY(acc, bias).eval()

    x = torch.randn(4, 4, 2114)
    with torch.no_grad():
        acc_logits, acc_lc = acc(x)
        bias_logits, bias_lc = bias(x)
        prof, lc = model(x)

    assert torch.allclose(prof, acc_logits + bias_logits, atol=1e-5), "profile != acc+bias"
    expected_lc = torch.logsumexp(torch.stack([acc_lc, bias_lc], dim=-1), dim=-1)
    assert torch.allclose(lc, expected_lc, atol=1e-6), "logcount != logsumexp"
    assert prof.shape == (4, 1, 1000), f"unexpected profile shape {tuple(prof.shape)}"
    assert lc.shape == (4, 1), f"unexpected logcount shape {tuple(lc.shape)}"

    # frozen bias: no bias params should require grad; accessibility should.
    assert not any(p.requires_grad for p in model.bias.parameters()), "bias not frozen"
    assert any(p.requires_grad for p in model.accessibility.parameters()), "accessibility frozen"
    print("BiasFactorizedCAPY self-test passed:", {"profile": tuple(prof.shape), "logcount": tuple(lc.shape)})
    _ = np  # silence unused when numpy import is trimmed by linters


if __name__ == "__main__":
    _self_test()
