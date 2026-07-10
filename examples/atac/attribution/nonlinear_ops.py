"""Custom DeepLIFT/DeepSHAP nonlinearity handlers for CAPY's non-standard modules.

``tangermeme.deep_lift_shap`` attaches a Rescale-rule backward hook only to modules
whose *exact type* is a key in its internal ``_NON_LINEAR_OPS`` registry (ReLU, GELU,
Sigmoid, ``torch.nn.MaxPool1d``, Softmax, ...). Any nonlinearity whose module class is
not in that registry is **silently treated as linear** (plain autograd gradient),
which breaks the DeepLIFT summation-to-delta property and inflates convergence deltas.

CAPY's encoder pools with :class:`capybara.layers.SameMaxPool1d`, a *custom* class
that wraps ``F.max_pool1d`` with asymmetric same-padding. Because it is not
``torch.nn.MaxPool1d``, tangermeme's built-in ``_maxpool`` rule never fires for it.
:func:`_same_maxpool` re-implements that rule for ``SameMaxPool1d`` (accounting for the
internal pad), and :data:`BIAS_ADDITIONAL_NONLINEAR_OPS` wires it in via
``deep_lift_shap(..., additional_nonlinear_ops=...)``.

**Shared-module reuse.** ``SequenceEncoder`` calls a *single* ``SameMaxPool1d`` instance
at every resolution (``layers.py:310,315``). tangermeme's forward hooks store
``module.input``/``module.output`` as plain attributes, so a reused module keeps only the
*last* call's tensors — during backward the earlier (larger) resolutions read stale
tensors and the shapes mismatch. :func:`track_shared_maxpools` fixes this by pushing each
call's input onto a per-module stack (forward runs push in order; backward pops LIFO, the
exact reverse), so the handler always sees the correct per-call input and recomputes the
matching output. Use it as a context manager around the ``deep_lift_shap`` call.

The count-head ``LayerNorm`` is also nonlinear (input-dependent μ/σ) and not in tangermeme's
registry — so it is left as an **autograd-gradient passthrough**, which is exactly the
Captum/tangermeme default: neither library registers ``LayerNorm`` (Captum's
``SUPPORTED_NON_LINEAR`` covers only ReLU/ELU/LeakyReLU/Sigmoid/Tanh/Softplus, MaxPool1/2/3d,
and Softmax; tangermeme's ``_NON_LINEAR_OPS`` adds GELU/SiLU etc. but no norm layers). We do
**not** register it: the generic elementwise Rescale rule is the wrong tool for a
cross-dimensional op (Captum handles its one supported cross-dim op, Softmax, with a *dedicated*
renormalizing rule, not the generic one), and forcing it would only guarantee conservation
tautologically, not correct credit distribution. Consequence: the counts head keeps a ~0.46
convergence-delta gap — the standard-tool behavior, accepted here. The **profile** head (the
primary Tn5 QC) has no LayerNorm and is fully covered by the maxpool handler (delta ~0.13).

Remaining bias-model ops are already handled or genuinely linear: GELU is a real ``nn.GELU``
module in tangermeme's registry; BatchNorm1d in ``.eval()`` is affine (linear);
``adaptive_avg_pool1d`` is linear.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from capybara.layers import SameMaxPool1d


def _same_maxpool(module, grad_input, grad_output):
    """DeepLIFT Rescale rule for :class:`capybara.layers.SameMaxPool1d`.

    Adapted from tangermeme's ``_maxpool`` (ported from Captum). The stacked input
    ``[x, x_ref]`` (dim 0) comes from the per-call stack maintained by
    :func:`track_shared_maxpools` (falling back to ``module.input`` for a non-shared
    module). We mirror ``SameMaxPool1d.forward`` exactly — derive the same asymmetric
    zero-pad, pool with indices, recompute the output, scatter ``grad_output * delta_out``
    back with ``max_unpool1d``, then crop the pad so the result aligns with ``grad_input``.
    """
    kernel_size = module.kernel_size
    stride = module.stride

    stack = getattr(module, "_dl_in_stack", None)
    x = stack.pop() if stack else module.input  # (2B, C, L), un-padded per-call input
    input_len = x.shape[-1]
    out_len = (input_len + stride - 1) // stride
    pad_total = max((out_len - 1) * stride + kernel_size - input_len, 0)
    left = pad_total // 2
    right = pad_total - left

    with torch.no_grad():
        delta_in_ = torch.sub(*x.chunk(2))
        delta_in = torch.cat([delta_in_, delta_in_])

        x_pad = F.pad(x, (left, right)) if pad_total > 0 else x  # 0-pad, matches forward
        pooled, indices = F.max_pool1d(
            x_pad, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=True
        )
        output, output_ref = pooled.chunk(2)
        delta_out_xmax = torch.max(output, output_ref)
        delta_out = torch.cat([delta_out_xmax - output_ref, output - delta_out_xmax])

        unpool_ = F.max_unpool1d(
            grad_output[0] * delta_out, indices, kernel_size, stride, padding=0, output_size=list(x_pad.shape)
        )
        if pad_total > 0:
            unpool_ = unpool_[..., left : left + input_len]  # crop pad -> align with grad_input
        unpool_delta, unpool_ref_delta = torch.chunk(unpool_, 2)

    unpool_delta_ = unpool_delta + unpool_ref_delta
    unpool_delta = torch.cat([unpool_delta_, unpool_delta_])
    idxs = torch.abs(delta_in) < 1e-7

    new_grad_inp = torch.where(idxs, grad_input[0], unpool_delta / delta_in)
    return (new_grad_inp,)


def _push_input(module, inputs):
    module._dl_in_stack.append(inputs[0].detach())


@contextlib.contextmanager
def track_shared_maxpools(model):
    """Track each ``SameMaxPool1d`` call's input so the handler is reuse-safe.

    Registers a forward-pre-hook per ``SameMaxPool1d`` that pushes the call's input onto
    a per-module stack; the backward handler pops it. Forward pushes run in call order and
    backward pops are LIFO (reverse), so each backward firing gets its own call's input —
    correct even when one pool instance is reused at several resolutions. Only forward-pre
    hooks are added (``module._backward_hooks`` stays empty), so tangermeme still registers
    its own rescale hooks normally.

    A wrapper-level forward-pre-hook clears all pool stacks at the start of every forward.
    This is required because ``ProfileScalarWrapper`` computes the *full* model (both heads),
    so the count-head pool is pushed on every forward but is **off** the profile-scalar
    backward path — its entry is never popped and would otherwise accumulate one tensor per
    batch (a leak that OOMs a long run). Clearing per-forward bounds every stack to a single
    forward: on-path pools push/pop within the chunk (deep_lift_shap does one forward then one
    backward per chunk), and any off-path stale entry is dropped at the next forward. The
    parent (wrapper) forward-pre-hook fires before any child pool's push.
    """
    handles = []
    pools = [m for m in model.modules() if isinstance(m, SameMaxPool1d)]
    for m in pools:
        m._dl_in_stack = []
        handles.append(m.register_forward_pre_hook(_push_input))

    def _clear_stacks(module, inputs):
        for m in pools:
            m._dl_in_stack.clear()

    handles.append(model.register_forward_pre_hook(_clear_stacks))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
        for m in pools:
            if hasattr(m, "_dl_in_stack"):
                del m._dl_in_stack


# Registered whenever the DeepLIFT/DeepSHAP engine runs on a CAPY model (bias or nobias
# encoders both use SameMaxPool1d). Keyed on module class. LayerNorm is deliberately NOT
# registered — left at the Captum/tangermeme autograd-gradient default (see module docstring).
BIAS_ADDITIONAL_NONLINEAR_OPS = {SameMaxPool1d: _same_maxpool}
