"""Energy-style loss for Boltz denoiser using boltz.energy attention losses."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

# Import from boltz.energy.energy_loss (local copy, no external dependency)
from src.boltz.energy.energy_loss import attn_loss_new, attn_contra_loss


class BoltzEnergyLoss(nn.Module):
    """Energy-style loss for Boltz denoiser using nn_flow attention losses.

    This loss encourages the denoiser to produce coordinates that are 
    attracted to positive samples (ground truth) and repelled from 
    negative samples (decoys, if provided).

    Parameters
    ----------
    align_fn : callable or None
        Function `(pred_xyz, ref_xyz, weights, mask) -> (pred_aligned, ref_aligned)` 
        that aligns the predicted coordinates to the reference before loss computation.
        If *None*, alignment is skipped.
    use_contra : bool, default False
        If *True*, use `attn_contra_loss` (contrastive). Otherwise use
        `attn_loss_new` (attractive with optional repulsion).
    **loss_kwargs
        Extra keyword arguments forwarded to the chosen nn_flow loss
        implementation (e.g. R_list, target_ratio, n_sinkhorn_steps, etc).
    """

    def __init__(
        self,
        align_fn: Optional[Callable] = None,
        *,
        use_contra: bool = False,
        **loss_kwargs,
    ) -> None:
        super().__init__()
        self.align_fn = align_fn
        self.use_contra = use_contra
        self.loss_kwargs: Dict = loss_kwargs

    def forward(
        self,
        gen_xyz: Optional[torch.Tensor] = None,  # [B, N, 3] or [B, R, N, 3]
        ref_xyz: Optional[torch.Tensor] = None,  # [B, N, 3]
        neg_xyz: Optional[torch.Tensor] = None,  # [B, M, 3]
        # Alternative interface (for compatibility with existing code)
        target: Optional[torch.Tensor] = None,   # [B, N, 3] - alias for ref_xyz
        recon: Optional[torch.Tensor] = None,    # [B, N, 3] or [B, R, N, 3] - alias for gen_xyz
        fixed_neg: Optional[torch.Tensor] = None,# alias for neg_xyz
        atom_mask: Optional[torch.Tensor] = None,# [B, N] mask for valid atoms
        scale_override: Optional[torch.Tensor] = None,  # not used, for interface compatibility
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the energy loss.

        Supports two interfaces:
        1. New interface: gen_xyz, ref_xyz, neg_xyz
        2. Legacy interface: target, recon, fixed_neg, atom_mask

        Parameters
        ----------
        gen_xyz / recon : torch.Tensor
            Predicted coordinates from denoiser. Shape [B, N, 3] or [B, R, N, 3] 
            where R is the multiplicity (number of samples per batch item).
        ref_xyz / target : torch.Tensor
            Ground truth reference coordinates. Shape [B, N, 3].
        neg_xyz / fixed_neg : torch.Tensor, optional
            Negative samples for repulsion. Shape [B, M, 3].
        atom_mask : torch.Tensor, optional
            Boolean or 0/1 mask for valid atoms. Shape [B, N].

        Returns
        -------
        loss : torch.Tensor
            Scalar loss (mean over batch and multiplicity).
        info : dict
            Diagnostics returned by the underlying nn_flow loss.
        """
        # Handle both interfaces
        if gen_xyz is None and recon is not None:
            gen_xyz = recon
        if ref_xyz is None and target is not None:
            ref_xyz = target
        if neg_xyz is None and fixed_neg is not None:
            neg_xyz = fixed_neg

        if gen_xyz is None or ref_xyz is None:
            raise ValueError("Must provide either (gen_xyz, ref_xyz) or (recon, target)")

        if gen_xyz.shape[-1] != 3 or ref_xyz.shape[-1] != 3:
            raise ValueError("Inputs must have last dimension = 3 (xyz coordinates).")

        # Handle multiplicity: if gen_xyz is [B, R, N, 3], process each sample
        if gen_xyz.dim() == 4:
            B, R, N, _ = gen_xyz.shape
            # Flatten multiplicity into batch dimension for loss computation
            gen_xyz_flat = gen_xyz.reshape(B * R, N, 3)
            # Repeat ref_xyz for each sample
            ref_xyz_rep = ref_xyz.unsqueeze(1).expand(-1, R, -1, -1).reshape(B * R, N, 3)
            
            # Handle mask
            if atom_mask is not None:
                atom_mask_rep = atom_mask.unsqueeze(1).expand(-1, R, -1).reshape(B * R, N)
            else:
                atom_mask_rep = None
            
            # Compute loss on flattened batch
            loss, info = self._compute_loss(gen_xyz_flat, ref_xyz_rep, neg_xyz, atom_mask_rep)
            
            # Reshape loss back to [B, R] then mean
            if loss.dim() > 0 and loss.numel() == B * R:
                loss = loss.view(B, R).mean()
            else:
                loss = loss.mean()
            
            return loss, info
        else:
            # Standard [B, N, 3] case
            return self._compute_loss(gen_xyz, ref_xyz, neg_xyz, atom_mask)

    def _compute_loss(
        self,
        gen_xyz: torch.Tensor,  # [B, N, 3]
        ref_xyz: torch.Tensor,  # [B, N, 3]
        neg_xyz: Optional[torch.Tensor],
        atom_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """Core loss computation."""
        
        # Create weights from mask (for alignment and potentially for loss weighting)
        if atom_mask is not None:
            weights = atom_mask.float()
            mask = atom_mask.float()
        else:
            weights = torch.ones(gen_xyz.shape[:-1], device=gen_xyz.device, dtype=gen_xyz.dtype)
            mask = weights

        # Optional alignment (no gradients for alignment transform)
        if self.align_fn is not None:
            # The align function signature: (true_coords, pred_coords, weights, mask) -> (aligned_pred, true)
            # We detach alignment so gradients flow through the original coordinates
            gen_xyz_aligned, _ = self.align_fn(ref_xyz.detach(), gen_xyz, weights, mask)
            gen_xyz = gen_xyz_aligned

        # Apply mask to coordinates (zero out invalid atoms)
        if atom_mask is not None:
            gen_xyz = gen_xyz * atom_mask.unsqueeze(-1)
            ref_xyz = ref_xyz * atom_mask.unsqueeze(-1)

        # nn_flow expects [B, C, S]. We use C = number of atoms, S = 3 (xyz).
        gen_feat = gen_xyz  # [B, N, 3] - already in correct shape
        pos_feat = ref_xyz  # [B, N, 3]

        neg_feat = None
        if neg_xyz is not None:
            neg_feat = neg_xyz  # [B, M, 3]

        # Compute loss
        if self.use_contra:
            loss, info = attn_contra_loss(
                target=pos_feat,
                recon=gen_feat,
                return_info=True,
                **self.loss_kwargs,
            )
        else:
            loss, info = attn_loss_new(
                gen=gen_feat,
                fixed_pos=pos_feat,
                fixed_neg=neg_feat,
                return_info=True,
                **self.loss_kwargs,
            )

        return loss.mean(), info
