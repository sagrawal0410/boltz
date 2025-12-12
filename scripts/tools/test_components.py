#!/usr/bin/env python3
"""Test script for components.py utilities."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from boltz.model.utils.components import (
    ALL_COMPONENTS,
    COMPONENT_BY_TOKEN,
    freeze_components,
    infer_component,
)


class MockModel(nn.Module):
    """Mock model with parameters matching component tokens."""

    def __init__(self):
        super().__init__()
        # Trunk components
        self.input_embedder = nn.Linear(10, 20)
        self.s_init = nn.Linear(20, 20)
        self.z_init_1 = nn.Linear(20, 20)
        self.z_init_2 = nn.Linear(20, 20)
        self.rel_pos = nn.Linear(20, 20)
        self.token_bonds = nn.Linear(1, 20)
        self.contact_conditioning = nn.Linear(20, 20)
        self.s_norm = nn.LayerNorm(20)
        self.z_norm = nn.LayerNorm(20)
        self.s_recycle = nn.Linear(20, 20)
        self.z_recycle = nn.Linear(20, 20)
        self.msa_module = nn.Linear(20, 20)
        self.pairformer_module = nn.Linear(20, 20)
        self.distogram_module = nn.Linear(20, 20)
        self.bfactor_module = nn.Linear(20, 20)

        # Denoiser components
        self.diffusion_conditioning = nn.Linear(20, 20)
        self.structure_module = nn.Linear(20, 20)

        # Confidence component
        self.confidence_module = nn.Linear(20, 20)

        # Affinity components
        self.affinity_module = nn.Linear(20, 20)
        self.affinity_module1 = nn.Linear(20, 20)
        self.affinity_module2 = nn.Linear(20, 20)

        # Other (not in mapping)
        self.other_layer = nn.Linear(20, 20)


def test_infer_component():
    """Test component inference from parameter names."""
    print("Testing infer_component()...")

    test_cases = [
        ("input_embedder.weight", "trunk"),
        ("s_init.bias", "trunk"),
        ("msa_module.weight", "trunk"),
        ("pairformer_module.weight", "trunk"),
        ("distogram_module.weight", "trunk"),
        ("diffusion_conditioning.weight", "denoiser"),
        ("structure_module.weight", "denoiser"),
        ("confidence_module.weight", "confidence"),
        ("affinity_module.weight", "affinity"),
        ("affinity_module1.weight", "affinity"),
        ("affinity_module2.weight", "affinity"),
        ("other_layer.weight", "other"),
        ("unknown_component.weight", "other"),
    ]

    all_passed = True
    for param_name, expected_component in test_cases:
        result = infer_component(param_name)
        if result == expected_component:
            print(f"  ✓ '{param_name}' -> {result}")
        else:
            print(f"  ✗ '{param_name}' -> {result} (expected {expected_component})")
            all_passed = False

    return all_passed


def test_freeze_components():
    """Test freezing components."""
    print("\nTesting freeze_components()...")

    model = MockModel()

    # Ensure all parameters are trainable initially
    for param in model.parameters():
        param.requires_grad = True

    # Freeze trunk and confidence
    freeze_components(model, ["trunk", "confidence"])

    # Check that trunk parameters are frozen
    trunk_params = [
        "input_embedder.weight",
        "s_init.weight",
        "msa_module.weight",
        "pairformer_module.weight",
        "distogram_module.weight",
    ]
    all_passed = True

    for name, param in model.named_parameters():
        component = infer_component(name)
        if component in ["trunk", "confidence"]:
            if param.requires_grad:
                print(f"  ✗ '{name}' (component: {component}) should be frozen but requires_grad=True")
                all_passed = False
            else:
                print(f"  ✓ '{name}' (component: {component}) correctly frozen")
        elif component == "denoiser":
            if not param.requires_grad:
                print(f"  ✗ '{name}' (component: {component}) should be trainable but requires_grad=False")
                all_passed = False
            else:
                print(f"  ✓ '{name}' (component: {component}) correctly trainable")
        elif component == "other":
            if not param.requires_grad:
                print(f"  ✗ '{name}' (component: {component}) should be trainable but requires_grad=False")
                all_passed = False
            else:
                print(f"  ✓ '{name}' (component: {component}) correctly trainable")

    return all_passed


def test_freeze_empty_list():
    """Test that freezing empty list doesn't break anything."""
    print("\nTesting freeze_components() with empty list...")

    model = MockModel()
    for param in model.parameters():
        param.requires_grad = True

    # Should not raise an error
    freeze_components(model, [])
    freeze_components(model, set())

    # All should still be trainable
    all_trainable = all(p.requires_grad for p in model.parameters())
    if all_trainable:
        print("  ✓ Empty freeze list leaves all parameters trainable")
        return True
    else:
        print("  ✗ Empty freeze list incorrectly froze some parameters")
        return False


def test_component_statistics():
    """Print statistics about component distribution in a model."""
    print("\nComponent statistics for MockModel:")

    model = MockModel()
    component_counts = {comp: 0 for comp in ALL_COMPONENTS}

    for name, param in model.named_parameters():
        component = infer_component(name)
        component_counts[component] += 1

    for component, count in sorted(component_counts.items()):
        if count > 0:
            print(f"  {component}: {count} parameters")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing components.py utilities")
    print("=" * 60)

    results = []

    results.append(("infer_component", test_infer_component()))
    results.append(("freeze_components", test_freeze_components()))
    results.append(("freeze_empty_list", test_freeze_empty_list()))
    test_component_statistics()

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
        return 0
    else:
        print("Some tests FAILED! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())


