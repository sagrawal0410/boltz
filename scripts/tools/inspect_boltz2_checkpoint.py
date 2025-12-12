#!/usr/bin/env python3
"""Inspect Boltz2 checkpoints and group weights by high-level module.

This utility is helpful when you need to understand which tensors belong to
the trunk, denoiser (structure module), confidence head, or ancillary blocks
before selectively freezing parts of the model.

Example
-------
python scripts/tools/inspect_boltz2_checkpoint.py \
    --checkpoint /path/to/boltz2.ckpt \
    --list-components trunk denoiser \
    --max-names 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import torch

from boltz.model.utils.components import (
    ALL_COMPONENTS,
    COMPONENT_BY_TOKEN,
    infer_component,
)


@dataclass
class ComponentSummary:
    """Track per-component statistics and parameter names."""

    tensor_count: int = 0
    param_count: int = 0
    names: list[str] = field(default_factory=list)

    def add(self, name: str, tensor: torch.Tensor) -> None:
        self.tensor_count += 1
        self.param_count += int(tensor.numel())
        self.names.append(name)


def summarize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, ComponentSummary]:
    """Aggregate tensors by component."""

    summary: dict[str, ComponentSummary] = {
        component: ComponentSummary() for component in ALL_COMPONENTS
    }

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            # Skip metadata entries (e.g., optimizer states)
            continue
        component = infer_component(name)
        summary.setdefault(component, ComponentSummary()).add(name, tensor)
    return summary


def format_human_readable(summary: Mapping[str, ComponentSummary], max_names: int) -> str:
    """Return a textual report."""

    lines = []
    total_params = sum(item.param_count for item in summary.values())
    for component in ALL_COMPONENTS:
        item = summary.get(component)
        if item is None or item.tensor_count == 0:
            continue
        pct = (item.param_count / total_params * 100.0) if total_params else 0.0
        lines.append(
            f"[{component}] tensors={item.tensor_count:,} params={item.param_count:,} ({pct:.2f}% of total)"
        )
        if max_names > 0:
            preview = item.names[:max_names]
            for name in preview:
                lines.append(f"  - {name}")
            if len(item.names) > max_names:
                lines.append(f"  ... (+{len(item.names) - max_names} more)")
    return "\n".join(lines)


def save_json(summary: Mapping[str, ComponentSummary], output_path: Path) -> None:
    """Persist summary to JSON."""

    serializable = {
        component: {
            "tensor_count": item.tensor_count,
            "param_count": item.param_count,
            "names": item.names,
        }
        for component, item in summary.items()
        if item.tensor_count
    }
    output_path.write_text(json.dumps(serializable, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a Boltz2 Lightning checkpoint (e.g., boltz2.ckpt).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for loading the checkpoint.",
    )
    parser.add_argument(
        "--list-components",
        nargs="*",
        default=[],
        choices=ALL_COMPONENTS,
        help="Restrict detailed name listing to the selected components.",
    )
    parser.add_argument(
        "--max-names",
        type=int,
        default=10,
        help="Number of parameter names to display per component (when listed).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save the full summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    summary = summarize_state_dict(state_dict)

    if args.list_components:
        filtered_summary = {
            component: summary.get(component, ComponentSummary())
            for component in args.list_components
        }
        report = format_human_readable(filtered_summary, args.max_names)
    else:
        report = format_human_readable(summary, max_names=0)
    print(report)

    if args.json_out:
        save_json(summary, args.json_out.expanduser().resolve())
        print(f"\nSummary written to {args.json_out}")


if __name__ == "__main__":
    main()



