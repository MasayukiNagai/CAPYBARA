"""Scalarizer wrappers around a :class:`CAPY` model for attribution.

These reduce CAPY's two-head output to a single scalar per example, matching the
two attribution targets ChromBPNet explains (``evaluation/interpret/shap_utils.py``):

* **profile** — ``get_weightedsum_meannormed_logits``: mean-normalize the profile
  logits over the length axis, weight by the ``stop_gradient`` softmax over
  length, and sum. (Contribution to the *shape* of the profile.)
* **counts** — ``sum(log_counts)``. (Contribution to total predicted coverage.)

ATAC is single-track, so the profile output is ``(B, 1, L)``; we flatten to
``(B, L)`` and reduce over length. This is deliberately **not** ProCapNet's
dual-strand wrapper (which sums a softmax over both strands) — see the top-level
package docstring. Both wrappers return shape ``(B, 1)`` so a single ``target=0``
works for both ``tangermeme.deep_lift_shap`` and ``captum`` explainers.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ProfileScalarWrapper(nn.Module):
    """Weighted-sum mean-normalized profile logits -> scalar ``(B, 1)``."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        logits, _ = self.model(x)
        # (B, 1, L) -> (B, L); single-track ATAC.
        logits = logits.reshape(logits.shape[0], -1)
        mean_norm_logits = logits - logits.mean(dim=-1, keepdim=True)
        softmax_weights = torch.softmax(mean_norm_logits.detach(), dim=-1)
        weighted_sum = (mean_norm_logits * softmax_weights).sum(dim=-1, keepdim=True)
        return weighted_sum  # (B, 1)


class CountScalarWrapper(nn.Module):
    """Sum of log-count outputs -> scalar ``(B, 1)``."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        _, log_counts = self.model(x)
        return log_counts.sum(dim=-1, keepdim=True)  # (B, 1)


def _self_test() -> None:
    """Assert both wrappers equal a hand-computed scalar on a random batch."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from capybara import CAPY

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
    model = CAPY(cfg).eval()
    x = torch.randn(4, 4, 2114)

    with torch.no_grad():
        logits, log_counts = model(x)
        # Reference profile scalar.
        flat = logits.reshape(4, -1)
        mn = flat - flat.mean(dim=-1, keepdim=True)
        w = torch.softmax(mn, dim=-1)
        ref_profile = (mn * w).sum(dim=-1, keepdim=True)
        ref_count = log_counts.sum(dim=-1, keepdim=True)

        got_profile = ProfileScalarWrapper(model)(x)
        got_count = CountScalarWrapper(model)(x)

    assert got_profile.shape == (4, 1), got_profile.shape
    assert got_count.shape == (4, 1), got_count.shape
    assert torch.allclose(got_profile, ref_profile, atol=1e-5), "profile scalar mismatch"
    assert torch.allclose(got_count, ref_count, atol=1e-6), "count scalar mismatch"
    print("wrappers self-test passed:", {"profile": tuple(got_profile.shape), "count": tuple(got_count.shape)})


if __name__ == "__main__":
    _self_test()
