from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from .model import CAPY


def _count_params(module: nn.Module | Iterable[nn.Module], *, trainable_only: bool = False) -> int:
    modules = [module] if isinstance(module, nn.Module) else list(module)
    parameters = []
    seen: set[int] = set()
    for current_module in modules:
        for parameter in current_module.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            if trainable_only and not parameter.requires_grad:
                continue
            parameters.append(parameter)
    return int(sum(parameter.numel() for parameter in parameters))


def _humanize(value: int) -> str:
    if value < 1_000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1_000:.1f}K"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value / 1_000_000_000:.1f}B"


def _mode(module: nn.Module | Iterable[nn.Module]) -> str:
    modules = [module] if isinstance(module, nn.Module) else list(module)
    training_states = {current_module.training for current_module in modules}
    if training_states == {True}:
        return "train"
    if training_states == {False}:
        return "eval"
    return "mixed"


def print_model_summary(model: CAPY) -> None:
    """Print a compact trainable/total parameter summary for a CAPY model."""
    base_modules = [
        model.encoder,
        model.bottleneck,
        model.decoder,
        model.encoder_projector,
        model.bottleneck_projector,
        model.output_embedder,
    ]
    rows = [
        (
            "base",
            "Backbone",
            _count_params(base_modules, trainable_only=True),
            _count_params(base_modules),
            _mode(base_modules),
        ),
        (
            "profile head",
            type(model.profile_head).__name__,
            _count_params(model.profile_head, trainable_only=True),
            _count_params(model.profile_head),
            _mode(model.profile_head),
        ),
        (
            "count head",
            type(model.count_head).__name__,
            _count_params(model.count_head, trainable_only=True),
            _count_params(model.count_head),
            _mode(model.count_head),
        ),
    ]

    name_w = max(4, *(len(row[0]) for row in rows))
    type_w = max(4, *(len(row[1]) for row in rows))
    trainable_w = max(9, *(len(_humanize(row[2])) for row in rows))
    total_w = max(5, *(len(_humanize(row[3])) for row in rows))
    mode_w = max(4, *(len(row[4]) for row in rows))

    header = (
        f"  | {'Name':<{name_w}} | {'Type':<{type_w}} | "
        f"{'Trainable':>{trainable_w}} | {'Total':>{total_w}} | {'Mode':<{mode_w}}"
    )
    separator = "-" * (len(header) + 3)
    print(header, flush=True)
    print(separator, flush=True)

    for name, typ, trainable, total, mode in rows:
        print(
            f"  | {name:<{name_w}} | {typ:<{type_w}} | "
            f"{_humanize(trainable):>{trainable_w}} | {_humanize(total):>{total_w}} | "
            f"{mode:<{mode_w}}",
            flush=True,
        )

    print(separator, flush=True)

    grand_trainable = sum(row[2] for row in rows)
    grand_total = sum(row[3] for row in rows)
    print(f"Trainable params: {_humanize(grand_trainable)}", flush=True)
    print(f"Total params:     {_humanize(grand_total)}", flush=True)
