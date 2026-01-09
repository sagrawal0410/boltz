"""Alignment utilities for Boltz.

Contains a weighted Kabsch alignment so predicted coordinates can be
super-imposed on ground-truth coordinates before computing energy loss.
"""
from __future__ import annotations

from typing import Tuple

import torch


def weighted_rigid_align_centered(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align *pred_coords* onto *true_coords* using a weighted Kabsch algorithm.

    Parameters
    ----------
    true_coords : torch.Tensor, shape [B, N, 3]
        Reference atomic coordinates.
    pred_coords : torch.Tensor, shape [B, N, 3]
        Coordinates predicted by the model.
    weights : torch.Tensor, shape [B, N]
        Per-atom weights ( e.g. 1 for heavy atoms, 0 for padding ).
    mask : torch.Tensor, shape [B, N]
        Boolean / 0-1 mask selecting atoms to consider in the alignment.

    Returns
    -------
    pred_aligned : torch.Tensor, shape [B, N, 3]
        *pred_coords* after optimal rotation/translation onto *true_coords*.
    true_coords : torch.Tensor, shape [B, N, 3]
        Returned unchanged so callers can do ``aligned_pred, aligned_true = align_fn(...)``.
    """
    if true_coords.shape != pred_coords.shape:
        raise ValueError("pred_coords and true_coords must share shape [B, N, 3].")

    batch_size, num_points, dim = true_coords.shape
    if dim != 3:
        raise ValueError("Coordinates must have last dimension = 3.")

    weights = (mask * weights).unsqueeze(-1)  # [B, N, 1]

    # Weighted centroids
    w_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / w_sum
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / w_sum

    # Centered coords
    true_centered = true_coords - true_centroid
    pred_centered = pred_coords - pred_centroid

    # Covariance
    cov = torch.einsum("b n i, b n j -> b i j", weights * pred_centered, true_centered)

    # SVD (on float32 for numerical stability)
    cov_f32 = cov.to(dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(cov_f32, full_matrices=False)
    V = Vh.mH  # transpose conjugate

    # Build rotation ensuring det = +1
    det_sign = torch.sign(torch.det(torch.einsum("b i j, b k j -> b i k", U, V)))
    F = torch.eye(dim, device=cov.device, dtype=cov_f32.dtype).expand(batch_size, -1, -1).clone()
    F[:, -1, -1] = det_sign
    R = torch.einsum("b i j, b j k, b l k -> b i l", U, F, V)
    R = R.to(dtype=true_coords.dtype)

    # Apply rotation + translation to pred
    pred_aligned = torch.einsum("b n i, b i j -> b n j", pred_centered, R) + true_centroid

    return pred_aligned, true_coords

