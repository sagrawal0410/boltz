# test_memory_bank.py
import pytest
import torch

from memory_bank import MemoryBank  # adjust if your class lives elsewhere

def make_nested_batch(B=4, device="cpu"):
    """
    Create a nested dict-of-dicts with tensor leaves (each [B, ...]).
    Two branches with different shapes/dtypes to stress the code.
    """
    return {
        "img": {
            "low_res":  torch.randn(B, 3, 8, 8, device=device, dtype=torch.float32),
            "high_res": torch.randn(B, 3, 16, 16, device=device, dtype=torch.float32),
        },
        "meta": {
            "token_ids": torch.randint(0, 100, (B, 5), device=device, dtype=torch.int64),
            "score":     torch.randn(B, 1, device=device, dtype=torch.float32),
        },
    }

def assert_same_structure(a, b):
    assert isinstance(a, dict) and isinstance(b, dict)
    assert set(a.keys()) == set(b.keys())
    for k in a:
        if isinstance(a[k], dict):
            assert isinstance(b[k], dict)
            assert_same_structure(a[k], b[k])
        else:
            # leaf tensors only (by contract)
            assert isinstance(a[k], torch.Tensor)
            assert isinstance(b[k], torch.Tensor)

def test_add_and_sample_shapes_dtypes_cpu():
    torch.manual_seed(0)
    B = 3
    bank = MemoryBank(num_classes=5, max_size=10, device="cpu")

    samples = make_nested_batch(B=B, device="cpu")
    labels = torch.tensor([1, 1, 2], dtype=torch.long)  # two go to class 1, one to class 2
    bank.add(samples, labels)

    # Sample 4 per label in a batch of size 3
    out = bank.sample(labels=torch.tensor([1, 2, 1], dtype=torch.long), n_samples=4)

    # Structure should mirror the input nests
    assert_same_structure(samples, out)

    # Check shapes/dtypes on a few leaves
    assert out["img"]["low_res"].shape == (3, 4, 3, 8, 8)
    assert out["img"]["low_res"].dtype == torch.float32
    assert out["meta"]["token_ids"].shape == (3, 4, 5)
    assert out["meta"]["token_ids"].dtype == torch.int64
    assert out["meta"]["score"].shape == (3, 4, 1)

def test_unseen_label_returns_zeros():
    torch.manual_seed(0)
    B = 2
    bank = MemoryBank(num_classes=3, max_size=5, device="cpu")

    # Only add for class 0
    samples = make_nested_batch(B=B)
    bank.add(samples, labels=torch.tensor([0, 0], dtype=torch.long))

    # Sample for a completely unseen class 2
    out = bank.sample(labels=torch.tensor([2, 2], dtype=torch.long), n_samples=3)

    # Expect zeros with correct shapes/dtypes
    low = out["img"]["low_res"]
    tok = out["meta"]["token_ids"]
    assert low.shape == (2, 3, 3, 8, 8)
    assert tok.shape == (2, 3, 5)
    assert torch.all(low == 0)
    assert torch.all(tok == 0)

def test_sampling_with_replacement_when_insufficient():
    torch.manual_seed(0)
    bank = MemoryBank(num_classes=2, max_size=3, device="cpu")

    # Add just one item to class 1
    samples = make_nested_batch(B=1)
    bank.add(samples, labels=torch.tensor([1], dtype=torch.long))

    # Ask for 5 samples -> must sample with replacement
    out = bank.sample(labels=torch.tensor([1], dtype=torch.long), n_samples=5)
    assert out["img"]["high_res"].shape == (1, 5, 3, 16, 16)

    # All rows should be equal because only one prototype exists
    hr = out["img"]["high_res"][0]  # [5, 3, 16, 16]
    diffs = (hr - hr[0]).abs().max()
    assert diffs.item() == 0.0

def test_max_size_eviction_fifo_like():
    torch.manual_seed(0)
    max_size = 4
    bank = MemoryBank(num_classes=1, max_size=max_size, device="cpu")

    # Insert 6 unique items into class 0; oldest two should be evicted
    for val in range(6):
        s = {
            "branch": {
                "leaf": torch.full((1, 2), float(val)),  # [B=1, 2]
            }
        }
        bank.add(s, labels=torch.tensor([0], dtype=torch.long))

    # Sample many times; we should never see values 0 or 1 if eviction occurred
    out = bank.sample(labels=torch.tensor([0, 0, 0], dtype=torch.long), n_samples=20)
    leaf = out["branch"]["leaf"]  # [B=3, 20, 2]
    unique_vals = torch.unique(leaf).tolist()
    # Because values are floats encoded as int-like, compare numerically
    for old in [0.0, 1.0]:
        assert old not in unique_vals
    # The most recent 4 values should be present among samples
    for keep in [2.0, 3.0, 4.0, 5.0]:
        assert keep in unique_vals

def test_error_when_sample_before_add():
    bank = MemoryBank(num_classes=2, max_size=3, device="cpu")
    with pytest.raises(RuntimeError):
        bank.sample(labels=torch.tensor([0], dtype=torch.long), n_samples=1)

def test_mismatched_batch_size_raises():
    bank = MemoryBank(num_classes=2, max_size=3, device="cpu")
    # Create mismatched batch sizes on purpose
    bad = {
        "a": {"x": torch.randn(3, 4)},  # B=3
        "b": {"y": torch.randn(2, 5)},  # B=2  <-- mismatch
    }
    with pytest.raises(ValueError):
        bank.add(bad, labels=torch.tensor([0, 0, 0], dtype=torch.long))

def test_multi_label_batch_and_output():
    torch.manual_seed(0)
    bank = MemoryBank(num_classes=3, max_size=10, device="cpu")
    # Add two classes
    s1 = make_nested_batch(B=2); bank.add(s1, labels=torch.tensor([0, 0]))
    s2 = make_nested_batch(B=3); bank.add(s2, labels=torch.tensor([1, 1, 1]))

    # Request B=4 with mixed labels
    req_labels = torch.tensor([1, 0, 1, 1], dtype=torch.long)
    out = bank.sample(labels=req_labels, n_samples=3)

    # Check batch dim = 4 and shapes are consistent
    assert out["img"]["low_res"].shape[:2] == (4, 3)
    assert out["meta"]["token_ids"].shape[:2] == (4, 3)

    # Ensure dtype/device consistency
    assert out["meta"]["token_ids"].dtype == torch.int64
    assert out["img"]["low_res"].dtype == torch.float32

@pytest.mark.cuda
def test_cuda_device_roundtrip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    bank = MemoryBank(num_classes=2, max_size=4, device=device)
    s = make_nested_batch(B=2, device=device)
    bank.add(s, labels=torch.tensor([1, 1], device=device))
    out = bank.sample(labels=torch.tensor([1, 1], device=device), n_samples=2)

    # On GPU and with correct shapes
    assert out["img"]["high_res"].is_cuda
    assert out["img"]["high_res"].shape == (2, 2, 3, 16, 16)

if __name__ == "__main__":
    import sys, pytest as _pytest
    sys.exit(_pytest.main([__file__]))
