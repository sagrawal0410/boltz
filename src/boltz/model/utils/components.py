"""Utilities for grouping model parameters into logical components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch.nn as nn

__all__ = [
    "COMPONENT_BY_TOKEN",
    "ALL_COMPONENTS",
    "infer_component",
    "freeze_components",
]

# Shared mapping between parameter-name tokens and semantic components.
COMPONENT_BY_TOKEN: Mapping[str, str] = {
    # Trunk / shared representation stack
    "input_embedder": "trunk",
    "s_init": "trunk",
    "z_init_1": "trunk",
    "z_init_2": "trunk",
    "rel_pos": "trunk",
    "token_bonds": "trunk",
    "token_bonds_type": "trunk",
    "contact_conditioning": "trunk",
    "s_norm": "trunk",
    "z_norm": "trunk",
    "s_recycle": "trunk",
    "z_recycle": "trunk",
    "template_module": "trunk",
    "msa_module": "trunk",
    "pairformer_module": "trunk",
    "distogram_module": "trunk",
    "bfactor_module": "trunk",
    # Denoiser / structure predictor
    "diffusion_conditioning": "denoiser",
    "structure_module": "denoiser",
    # Confidence head
    "confidence_module": "confidence",
    # Affinity head(s)
    "affinity_module": "affinity",
    "affinity_module1": "affinity",
    "affinity_module2": "affinity",
    # EMA helper
    "ema": "ema",
}

ALL_COMPONENTS = (
    "trunk",
    "denoiser",
    "confidence",
    "affinity",
    "ema",
    "other",
)


def infer_component(param_name: str) -> str:
    """Infer which high-level component a parameter belongs to."""

    for token in param_name.split("."):
        if token in COMPONENT_BY_TOKEN:
            return COMPONENT_BY_TOKEN[token]
    return "other"


def freeze_components(module: nn.Module, components: Iterable[str]) -> None:
    """Disable gradients for all parameters that belong to the given components."""

    components = set(components)
    if not components:
        return

    for name, param in module.named_parameters():
        component = infer_component(name)
        if component in components:
            param.requires_grad = False


