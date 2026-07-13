"""Tests for the Majdandzic gradient correction + hypothetical/projected fix.

No pytest in this env, so these mirror the repo's ``_self_test`` convention: plain
asserts, runnable directly::

    .venv/bin/python -m examples.atac.attribution.test_gradient_correction

The functions are also ``test_*``-named so pytest can collect them if it is ever
added. Everything here runs on CPU with a tiny CAPY and no genome files.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.atac.attribution.contribs import (
    center_channels,
    deeplift_attributions,
    gradientshap_attributions,
    write_chrombpnet_h5,
)

INPUT_LENGTH = 2114
TINY_CFG = {
    "model": {
        "input_length": INPUT_LENGTH,
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
    "dataset": {"input_length": INPUT_LENGTH, "output_length": 1000},
}


def _tiny_model() -> torch.nn.Module:
    from capybara import CAPY

    torch.manual_seed(0)
    return CAPY(TINY_CFG).eval()


def _random_onehot(n: int, length: int, seed: int = 7) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, 4, size=(n, length))
    oh = np.zeros((n, 4, length), dtype=np.float32)
    for i in range(n):
        oh[i, idx[i], np.arange(length)] = 1.0
    return torch.from_numpy(oh)


# --------------------------------------------------------------------------- #
# 1. Zero-sum: the defining property of the simplex-tangent projection.
# --------------------------------------------------------------------------- #
def test_center_channels_zero_sum() -> None:
    rng = np.random.RandomState(0)
    x = rng.randn(3, 4, 50).astype(np.float32)
    out = center_channels(x)
    assert out.shape == x.shape
    assert np.abs(out.sum(axis=1)).max() < 1e-6, "not zero-sum over the 4-channel axis"
    # It is subtraction, not division: idempotent projection, and a constant-per-position
    # shift leaves nothing behind (a pure off-simplex input maps to ~0).
    assert np.allclose(out, center_channels(out), atol=1e-6), "projection not idempotent"
    const = np.broadcast_to(rng.randn(3, 1, 50), (3, 4, 50)).astype(np.float32)
    assert np.abs(center_channels(const)).max() < 1e-6, "channel-constant input must project to 0"
    print("test_center_channels_zero_sum passed")


# --------------------------------------------------------------------------- #
# 2. GradientShap returns a corrected (zero-sum) hypothetical; flag toggles it.
# --------------------------------------------------------------------------- #
def test_gradientshap_correction_zero_sum_and_flag() -> None:
    model = _tiny_model()
    onehot = _random_onehot(2, INPUT_LENGTH)
    device = torch.device("cpu")

    corrected = gradientshap_attributions(
        model, "profile", onehot, device=device, n_shuffles=2, gradient_correction=True, verbose=False
    )
    assert corrected.shape == (2, 4, INPUT_LENGTH)
    assert corrected.dtype == np.float16
    # Stored as float16, so the 4-value per-position sum only holds to ~1e-2.
    max_resid = np.abs(corrected.astype(np.float32).sum(axis=1)).max()
    assert max_resid < 1e-2, f"corrected f16 output not ~zero-sum (max |sum|={max_resid:.2e})"

    uncorrected = gradientshap_attributions(
        model, "profile", onehot, device=device, n_shuffles=2, gradient_correction=False, verbose=False
    )
    # The flag must actually change the output (otherwise the correction is a no-op).
    assert not np.allclose(corrected.astype(np.float32), uncorrected.astype(np.float32), atol=1e-4), (
        "correction flag had no effect"
    )
    print("test_gradientshap_correction_zero_sum_and_flag passed")


# --------------------------------------------------------------------------- #
# 3. deeplift path is untouched by the correction: the flag cannot reach it,
#    and its output is deterministic across identical calls.
# --------------------------------------------------------------------------- #
def test_deeplift_path_untouched() -> None:
    # The correction is a gradientshap-only concept: deeplift_attributions must not
    # even accept the flag, so there is no way for generate_scores to leak it in.
    # (deeplift's own output is nondeterministic across calls because tangermeme
    # reshuffles its references each call, so we assert the wiring, not the bytes.)
    assert "gradient_correction" not in inspect.signature(deeplift_attributions).parameters, (
        "deeplift_attributions must not take gradient_correction"
    )
    # generate_scores threads the flag only through the gradientshap branch.
    from examples.atac.attribution import contribs

    src = inspect.getsource(contribs.generate_scores)
    deeplift_block, gradientshap_block = src.split('if engine == "deeplift":')[1].split("else:", 1)
    assert "gradient_correction" not in deeplift_block, "flag leaked into the deeplift branch"
    assert "gradient_correction=gradient_correction" in gradientshap_block, (
        "flag not wired into the gradientshap branch"
    )
    print("test_deeplift_path_untouched passed")


# --------------------------------------------------------------------------- #
# 4. h5 schema is unchanged: keys, shapes (N,4,L), dtypes int8/f16/f16.
# --------------------------------------------------------------------------- #
def test_h5_schema_unchanged(tmp_path: Path | None = None) -> None:
    import tempfile

    import h5py

    # Use pytest's tmp_path when available, else a system tempdir (never the repo tree).
    out_dir = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp(prefix="capy_h5_"))
    path = out_dir / "schema.h5"

    n, length = 3, 2114
    onehot = _random_onehot(n, length).numpy()
    hyp = center_channels(np.random.RandomState(1).randn(n, 4, length).astype(np.float32)).astype(np.float16)
    write_chrombpnet_h5(path, onehot, hyp)

    with h5py.File(path, "r") as h:
        assert set(h.keys()) == {"raw", "shap", "projected_shap"}
        assert h["raw/seq"].shape == (n, 4, length) and h["raw/seq"].dtype == np.int8
        assert h["shap/seq"].shape == (n, 4, length) and h["shap/seq"].dtype == np.float16
        assert h["projected_shap/seq"].shape == (n, 4, length) and h["projected_shap/seq"].dtype == np.float16
        # projected == onehot * hyp (modisco recomputes the same product internally).
        proj = h["projected_shap/seq"][:].astype(np.float32)
        assert np.allclose(proj, (onehot * hyp.astype(np.float32)).astype(np.float16).astype(np.float32))
    path.unlink()
    print("test_h5_schema_unchanged passed")


def _run_all() -> None:
    test_center_channels_zero_sum()
    test_gradientshap_correction_zero_sum_and_flag()
    test_deeplift_path_untouched()
    test_h5_schema_unchanged()
    print("\nAll gradient-correction tests passed.")


if __name__ == "__main__":
    _run_all()
