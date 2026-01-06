import torch 
import torch.nn as nn
import einops
from einops import rearrange, repeat
from math import prod
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .persistence import persistent_class
from .misc import sg, custom_compile

from utils.boltz_utils import weighted_rigid_align_centered

def cdist(x, y, eps=1e-8):
    '''
    Args:
        x: [B, C1, D]
        y: [B, C2, D]
    Returns: [B, C1, C2]

    Same effect as torch.cdist, but faster. 
    '''
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    return (xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot).clamp(min=eps).sqrt()


def kernel(x, kernel_type="log"):
    """
    Kernel function for the contrastive loss
    x: the square of euclidean distance
    """
    if kernel_type == "log":
        return (x + 1e-3**2).sqrt().log()
    elif kernel_type == "sqrt":
        return (x + 1e-6).sqrt().sqrt()
    elif kernel_type == "dist":
        return (x + 1e-6).sqrt()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

# @torch.compile(dynamic=True)
def attn_loss_new(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=None,
    transpose_aff=False,
    old_gen=None,
    step_size=1.0,
    temp_isqrt_d=False,
    same_group_mask=None,
    target_ratio=None, 
    n_sinkhorn_steps=0,
    has_repulsion=False,
    exp_affinity=False,
    no_ratio=False,
    proj_dim=0,
    scale_override=None,
    **kwargs,
):
    """
    Args:
        gen: [B, C1, S]
        fixed_pos: [B, C2, S]
        fixed_neg: [B, C3, S] (optional, can be None)
        weight_gen: [B, C1] (optional; if None: weight is 1)
        weight_pos: [B, C2] (optional; if None: weight is 1)
        weight_neg: [B, C3] (optional; if None: weight is 1)
        R_list: a list of R values to use for the kernel function
        transpose_aff: whether to transpose the affinity matrix; if false: softmax on the targets; if true: average softmax on gen & targets.
        old_gen: [B, C1, S] (optional); if provided, use this for target computation.
        temp_isqrt_d: whether to use isqrt(d) to scale R
        same_group_mask: [B, C1, C1] (optional); if provided, mask out the corresponding pairs in gen.
        target_ratio: float; if specified, want max / mean_ratio to be target_ratio.
        n_sinkhorn_steps: int; number of additional sinkhorn steps to take. Odd steps: each gen normed to 1; even steps: each target normed to 1.
        proj_dim: int; if specified, project the features to the given dimension.
    Returns:
        loss: [batch_size]
        (optional) info: a dict with entries:

        # TODO: per sample normalization of output; per sample normalization of force.
    """
    # print("Owo")
    if R_list is None:
        R_list = [0.2]
    if len(kwargs) > 0:
        print("Additional keys:", kwargs.keys())
    old_dtype = fixed_pos.dtype
    # convert everything to float32

    realignment = kwargs.get('realignment', False)
    
    atom_mask = kwargs.get('atom_mask', None)

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(fixed_pos[:, :0, :])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])
    if old_gen is None:
        old_gen = gen
    
    # Convert everything to float32
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    gen = gen.float()
    old_gen = old_gen.float()
    weight_pos = weight_pos.float()
    weight_gen = weight_gen.float()
    weight_neg = weight_neg.float()
    
    old_gen = old_gen.detach()

    B, C_g, S = old_gen.shape
    B, C_p, S = fixed_pos.shape
    B, C_n, S = fixed_neg.shape
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat(
        [weight_gen, weight_neg, weight_pos], dim=1
    )  # [B, C_g + C_n + C_p]

    if proj_dim > 0:
        proj_matrix = torch.randn(B, S, proj_dim).to(old_gen)
        old_gen = torch.einsum("bnd,bdm->bnm", old_gen, proj_matrix)
        gen = torch.einsum("bnd,bdm->bnm", gen, proj_matrix)
        targets = torch.einsum("bnd,bdm->bnm", targets, proj_matrix)
    info = dict()

    with torch.no_grad():
        if not realignment:
            dist = cdist(old_gen, targets)  # [B, C_g, C_g + C_n + C_p]
            targets = targets.repeat_interleave(C_g, dim=0)  # [B * C_g, C_g + C_n + C_p, S]
            targets = einops.rearrange(targets, "(b cg) t s -> b cg t s", b=B, cg=C_g, t=C_g + C_n + C_p)  # [B, C_g, C_g + C_n + C_p, S]
        else:
            # Do weighted_rigid_alignment first for each C_g to targets

            assert targets.shape[-1] % 3 == 0, "Coordinates should be in 3D for realignment."

            targets_aligned = targets.repeat_interleave(C_g, dim=0)  # [B * C_g, C_g + C_n + C_p, S]
            # print(f"{targets_aligned.shape=}, b={B}, cg={C_g}, t={C_g + C_n + C_p}")
            targets_aligned = einops.rearrange(targets_aligned, "(b cg) t (n c) -> (b cg t) n c", b=B, cg=C_g, t=C_g + C_n + C_p, c=3)  # [(B * C_g * (C_g + C_n + C_p)), n, 3]
            gen_expanded = old_gen.unsqueeze(2).repeat_interleave(C_g + C_n + C_p, dim=2)  # [B, C_g, C_g + C_n + C_p, S]
            gen_expanded = einops.rearrange(gen_expanded, "b cg t (n c) -> (b cg t) n c", b=B, cg=C_g, t=C_g + C_n + C_p, c=3)  # [(B * C_g * (C_g + C_n + C_p)), n, 3]

            if atom_mask is not None: # [B, N]
                atom_mask = atom_mask.repeat_interleave(C_g * (C_g + C_n + C_p), dim=0)  # [(B * C_g * (C_g + C_n + C_p)), N]

            targets_aligned = weighted_rigid_align_centered(
                targets_aligned,
                gen_expanded,
                weights=torch.ones_like(atom_mask) if atom_mask is not None else None,
                mask=atom_mask,
            )  # [(B * C_g * (C_g + C_n + C_p)), n, 3]
            targets_aligned = einops.rearrange(
                targets_aligned, "(b cg t) n c -> (b cg) t (n c)", b=B, cg=C_g, t=C_g + C_n + C_p, c=3
            )  # [B * C_g, C_g + C_n + C_p, S]
            gen_aligned = einops.rearrange(
                old_gen, "b cg s -> (b cg) 1 s"
            )  # [B * C_g, 1, S]
            dist = cdist(gen_aligned, targets_aligned)  # [B * C_g, 1, C_g + C_n + C_p]
            dist = einops.rearrange(
                dist, "(b cg) 1 t -> b cg t", b=B, cg=C_g, t=C_g + C_n + C_p
            )  # [B, C_g, C_g + C_n + C_p]

            targets = einops.rearrange(
                targets_aligned, "(b cg) t s -> b cg t s", b=B, cg=C_g, t=C_g + C_n + C_p
            ) # [B, C_g, C_g + C_n + C_p, S]

        # Split the weights into (gen + neg) and pos parts
        w_nonpos = targets_w[:, : C_g + C_n]      # [B, C_g + C_n]
        w_pos    = targets_w[:, C_g + C_n :]      # [B, C_p]

        # For each batch element, rescale pos weights so that:
        #   sum_pos_scaled  == sum_nonpos
        sum_nonpos = w_nonpos.sum(dim=-1, keepdim=True)        # [B, 1]
        sum_pos    = w_pos.sum(dim=-1, keepdim=True) + 1e-8    # [B, 1]
        pos_scale  = sum_nonpos / sum_pos                      # [B, 1]

        w_pos_balanced = w_pos * pos_scale                     # [B, C_p]

        # Use these balanced weights only for computing the normalization scale
        targets_w_scale = torch.cat([w_nonpos, w_pos_balanced], dim=-1)  # [B, C_g + C_n + C_p]

        # Scale based on distances
        if scale_override is None:
            scale = (dist * targets_w_scale[:, None, :]).mean() / targets_w_scale.mean()
        else:
            scale = scale_override

        # e.g. mean pairwise distance in data space       # scalar

        info["scale"] = scale.mean()

        scale_inputs = (scale / math.sqrt(S)).clamp_min(1e-3)  # scale to make sure coords are N[0,1]
        dist = dist / scale.clamp_min(1e-3)                    # dist_map: elts of order 1

        dist[:, :C_g, :C_g] = (
            dist[:, :C_g, :C_g] + torch.eye(C_g, device=dist.device) * 100
        )
        if same_group_mask is not None:
            dist[:, :C_g, :C_g][same_group_mask] = 100

    # with torch.no_grad():
    #     dist = cdist(old_gen, targets)  # [B, C_g, C_g + C_n + C_p]
    #     scale = (dist * targets_w[:, None, :]).mean() / targets_w.mean()
    #     info["scale"] = scale.mean()
    #     scale_inputs = (scale / (math.sqrt(S))).clamp_min(
    #         1e-3
    #     )  # scale to make sure coords are N[0,1]
    #     dist = dist / scale.clamp_min(1e-3)  # dist_map: elts of order 1
    #     dist[:, :C_g, :C_g] = (
    #         dist[:, :C_g, :C_g] + torch.eye(C_g, device=dist.device) * 100
    #     )
    #     if same_group_mask is not None:
    #         dist[:, :C_g, :C_g][same_group_mask] = 100

    old_gen, targets, gen = (
        old_gen / scale_inputs,
        targets / scale_inputs,
        gen / scale_inputs,
    )
    with torch.no_grad():
        force_across_R = torch.zeros_like(old_gen)
        for R in R_list:
            temp = R if not temp_isqrt_d else R / math.sqrt(S)
            # Increase temperature
            # temp = temp * 2.0
            
            R_coeff = torch.zeros_like(dist)  # [B, C_g, C_g + C_n + C_p];
            affinity = (-dist / temp).softmax(dim=-1)  # [B, C_g, C_g + C_n + C_p]
            entropy = (-affinity * (affinity + 1e-8).log()).sum(dim=-1)
            info[f'temp_{R}'] = temp
            info[f'effective_samples_{R}'] = (entropy.mean()).exp()
            if transpose_aff:
                affinity = (
                    (affinity * (-dist / temp).softmax(dim=-2)).clamp_min(0.0).sqrt()
                )
            for i in range(n_sinkhorn_steps):
                if i % 2 == 0:
                    affinity = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-3)
                else:
                    affinity = affinity / (affinity.sum(dim=-2, keepdim=True) + 1e-3)
            if exp_affinity:
                affinity = (-dist / temp).exp()
            
            info[f'mean_affinity_{R}'] = affinity.mean()
            info[f'max_affinity_{R}'] = affinity.max(dim=-1).values.mean()
            ratio = affinity.max(dim=-1).values.mean() / affinity.mean()
            if target_ratio is not None:
                k = math.log(target_ratio) / math.log(ratio)
                affinity = affinity.pow(k)
            info[f'max_to_mean_affinity_{R}'] = affinity.max(dim=-1).values.mean() / affinity.mean()

            # weight by the weights
            # affinity: [B, C_g, C_g + C_n + C_p]
            # targets_w[:, None, :] : [B, 1, C_g + C_n + C_p]
            affinity = affinity * targets_w[:, None, :]  # [B, C_g, C_g + C_n + C_p]

            # Make sure pos_ker ~= neg_ker, for normalization of forces
            info[f"pos_ker_{R}"] = affinity[:, :, C_g + C_n :].sum(dim=-1).mean()
            info[f"neg_ker_{R}"] = affinity[:, :, : C_g + C_n].sum(dim=-1).mean()
            ratio = info[f"neg_ker_{R}"] / (info[f"pos_ker_{R}"] + 1e-3)
            if no_ratio:
                ratio = torch.ones_like(ratio)
            info[f"ratio_{R}"] = ratio.mean()
            affinity[:, :, C_g + C_n :] = (
                affinity[:, :, C_g + C_n :] * ratio
            )  # make sure: pos sum ~= neg sum
            R_coeff[:, :, C_g + C_n :] = affinity[:, :, C_g + C_n :] * (
                affinity[:, :, : C_g + C_n].sum(dim=-1, keepdim=True)
            )
            R_coeff[:, :, : C_g + C_n] = -affinity[:, :, : C_g + C_n] * (
                affinity[:, :, C_g + C_n :].sum(dim=-1, keepdim=True)
            )
            if has_repulsion:
                R_coeff[:, :, C_g + C_n :] = affinity[:, :, C_g + C_n :]
                R_coeff[:, :, : C_g + C_n] = -affinity[:, :, : C_g + C_n]

            # estimation of force, when **no cancellation**.
            norm_est = (
                ((affinity * dist) ** 2)
                .mean(dim=(-1, -2), keepdim=True)
                .clamp_min(1e-8)
                .sqrt()
            )
            info[f"norm_est_{R}"] = norm_est.mean()

            # Weird scaling things are happening here
            # R_coeff = R_coeff / norm_est

            total_force_R = torch.einsum("biy,biyx->bix", R_coeff, targets)
            if has_repulsion:
                total_force_R = total_force_R - R_coeff.sum(dim=-1,keepdim=True) * old_gen
            info[f"f_norm_{R}"] = (total_force_R**2).mean()

            # force_across_R = (
            #     force_across_R
            #     + total_force_R / (total_force_R**2).mean().clamp_min(1e-8).sqrt()
            # )
            
            # We disable RMS normalization here to avoid instability.
            force_across_R = force_across_R + total_force_R

        goal = (old_gen + force_across_R * step_size).detach()

    loss = ((gen - goal) ** 2).mean(dim=(-1, -2)).to(old_dtype)
    info["diff_base"] = ((gen - old_gen) ** 2).mean()
    return loss, info


@custom_compile(dynamic=True)
def attn_contra_loss(target, recon, return_info=False, sample_norm=True, weight_r = None, scale_dist=False, no_R_norm=False, no_global_norm=False, new_R_norm=False, scale_dist_normed=True, R_list = [0.2], coord_norm=False, softmax=True, gen_attend_data=True,data_attend_gen=True,softmax_p=0.5,norm_R_force=False,scale_by_pos=False):
    '''
    Best recommendation:
        sample_norm: True
        scale_disr_normed: True
        Other: False
        This will align different features. Other settings are mostly for exploration.
    Args:
        target: [batch_size, C1, S]
        recon: [batch_size, C2, S]
        return_info: whether to return the info dict
        sample_norm: whether to normalize the loss by the sample norm; 
            if enabled: loss will have shape (B, )
        weight_r: 
            The weight for negative samples. None or shape (B, C2). 
            When enabled: all repulsions will be weighted by weight_r. 
        softmax: whether to use softmax to normalize the affinity matrix. 
        scale_by_pos: whether to scale the dist only by distance of positive samples. 
    Returns:
        loss: [batch_size]
        (optional) info: a dict with entries:
            force_norm: the norm of the force.  
            prec: the precision of the target. 

    '''
    B, C1, S = target.shape
    B, C2, S = recon.shape
    if coord_norm:
        # normalize, to make sure every coordinate has mean 0 & var 1
        coord_mean = torch.cat([target, recon], dim=1).mean(dim=(0, 1)).detach()
        coord_std = torch.cat([target, recon], dim=1).std(dim=(0, 1)).detach()
        target = (target - coord_mean) / (coord_std + 1e-3)
        recon = (recon - coord_mean) / (coord_std + 1e-3)

    with torch.no_grad():
        pos_neg = torch.cat([target, recon], dim=1)
        dist = cdist(pos_neg, pos_neg) # [B, C1 + C2, C1 + C2]
        # assert target.shape[1] == 1
        scale = dist.mean()
        if sample_norm:
            scale = dist.mean(dim=(-1, -2), keepdim=True)
        if scale_by_pos:
            scale = dist[:, :C1, :C1].mean()
            if sample_norm:
                scale = dist[:, :C1, :C1].mean(dim=(-1, -2), keepdim=True)
    if scale_dist:
        target, recon, pos_neg = target / scale, recon / scale, pos_neg / scale
    if scale_dist_normed:
        assert not scale_dist
        scale_2 = scale / (math.sqrt(S))
        target, recon, pos_neg = target / scale_2, recon / scale_2, pos_neg / scale_2
    with torch.no_grad():
        dist = dist + torch.eye(C1 + C2, device=dist.device) * 100 * scale

        info = {"nn_data": dist[:, :C1].min(dim=-1).values.mean(), "nn_samples": dist[:, C1:].min(dim=-1).values.mean(), "scale": scale.mean(), "scale_normed": scale.mean() / (math.sqrt(S))}
        
        dist = dist / scale
        info['dist_std'] = (dist - torch.eye(C1 + C2, device=dist.device) * 100).std(dim=(-1,-2)).mean()
        
        total_pos = torch.zeros_like(dist)

        for R in R_list:
            if softmax:
                affinity = torch.ones_like(dist)
                if gen_attend_data:
                    affinity = affinity * torch.pow((-dist / R).softmax(dim=-1), softmax_p)
                if data_attend_gen:
                    affinity = affinity * torch.pow((-dist / R).softmax(dim=-2), softmax_p)
            else:
                # mean_dist = (-dist).logsumexp(dim=(-1, -2), keepdim=True)
                # affinity = ((-mean_dist - dist) / R).sigmoid()
                affinity = (-dist / R).exp()

            if weight_r is not None:
                affinity[:, :, C1:] = affinity[:, :, C1:] * weight_r[:, None, :]
            
            if sample_norm:
                norm_est = ((affinity * dist) ** 2).mean(dim=(-1, -2), keepdim=True).clamp_min(1e-8).sqrt()
                if norm_R_force:
                    norm_est = norm_est * affinity.mean(dim=(-1,-2),keepdim=True)
            cur_force = torch.zeros_like(dist)
            info[f'pos_ker_{R}'] = affinity[:, C1:, :C1].sum(dim=-1).mean()
            info[f'neg_ker_{R}'] = affinity[:, C1:, C1:].sum(dim=-1).mean()
            cur_force[:, C1:, C1:] = -affinity[:, C1:, C1:] * (affinity[:, C1:, :C1].sum(dim=-1, keepdim=True))
            cur_force[:, C1:, :C1] = affinity[:, C1:, :C1] * (affinity[:, C1:, C1:].sum(dim=-1, keepdim=True))
            if not sample_norm:
                norm_est = ((cur_force * dist) ** 2).mean().clamp_min(1e-8).sqrt()
            
            if new_R_norm:
                norm_est = (affinity * dist).mean().clamp_min(1e-8)
                if norm_R_force:
                    norm_est = norm_est * affinity.mean(dim=(-1,-2),keepdim=True)
            
            if not no_R_norm:
                cur_force = cur_force / norm_est
            total_pos = total_pos + cur_force
            info[f'norm_{R}'] = norm_est.mean()
            info[f'norm_{R}_std'] = norm_est.std() if (len(norm_est.shape) > 0 and norm_est.shape[0] > 1) else 0
        
        sum_forces = torch.einsum("biy,byx->bix", total_pos[:, C1:], pos_neg)
        info['f_norm'] = ((sum_forces ** 2).mean())
        if not no_global_norm:
            sum_forces = sum_forces / ((sum_forces ** 2).mean().clamp_min(1e-8).sqrt())
        goal = sg(recon + sum_forces)
    
    loss = ((recon - goal) ** 2).mean(dim=(-1, -2))
    if return_info:
        return loss, info
    return loss

def group_contra_loss(
    pos,
    gen,
    neg=None,
    pos_w=None,
    gen_w=None,
    neg_w=None,
    scale_override=None,
    kernel_type="log",
    return_info=True,
    **contra_dict
):
    """
    Args:
        pos: [batch_size, C1, S]
        gen: [batch_size, C2, S]
        neg: [batch_size, C3, S]
        pos_w: [batch_size, C1] | None; default to 1
        gen_w: [batch_size, C2] | None; default to 1
        neg_w: [batch_size, C3] | None; default to 1
        use_base: whether to use the base loss
        kernel_type: the type of kernel function
        use_scale: whether to use the scale loss
        return_info: whether to return the info dict
    Returns:
        loss: [batch_size] or scalar
        (optional) info: a dict
    """
    if kernel_type == "attn":
        # gen and neg are repelled from each other and attracted to pos.
        # loss is only on gen.
        recon = gen
        if neg is not None:
            recon = torch.cat([gen, neg], dim=1)
        
        if gen_w is None:
            gen_w = torch.ones_like(gen[:, :, 0])
        if neg is not None and neg_w is None:
            neg_w = torch.ones_like(neg[:, :, 0])

        weight_r = gen_w
        if neg is not None:
            weight_r = torch.cat([gen_w, neg_w], dim=1)

        return attn_contra_loss(pos, recon, weight_r=weight_r, **contra_dict, return_info=return_info)
        
    elif kernel_type == "attn_new":
        # note: won't have weight_r here...
        return attn_loss_new(gen=gen, fixed_pos=pos, fixed_neg=neg, weight_gen=gen_w, weight_pos=pos_w, weight_neg=neg_w, scale_override=scale_override, **contra_dict)

    if len(pos.shape) == 2:
        pos = pos.unsqueeze(1)  # [B, C, S]
    
    target, recon = pos, gen
    
    use_scale = contra_dict.get('use_scale', False)
    
    if use_scale:
        scale = target.std() + 1e-2
        target = target / scale
        recon = recon / scale
    else:
        scale = 1

    B, C, S = recon.shape
    dist_tar_recon = cdist(target, recon) ** 2 / S  # [B, C1, C2]
    dist_recon_recon = cdist(recon, recon) ** 2 / S  # [B, C2, C2]

    use_base=contra_dict.get('use_base', True)
    base = (dist_tar_recon.mean() + dist_recon_recon.mean()) / 100 if use_base else 0
    loss_recon = kernel(
        dist_tar_recon + base, kernel_type
    )  # [B, C1, C2]
    loss_contra = kernel(
        dist_recon_recon + 2 * base,
        kernel_type,
    )  # [B, C2, C2]
    loss_contra = loss_contra.masked_fill(
        torch.eye(loss_contra.shape[-1], device=loss_contra.device).bool(), 0
    )
    loss = loss_recon.mean(dim=(1, 2)) * 2 - loss_contra.mean(dim=(1, 2)) * C / (C - 1)

    info = dict(l2_recon=dist_tar_recon.mean(), l2_contra=dist_recon_recon.mean() * C / (C - 1), base=base)

    if return_info:
        return loss, info   
    return loss

@persistent_class
class FeatureExtractor(nn.Module):
    """
    Generic feature extractor for non-image data.

    For this project we use it only for coordinates [B, N, 3].
    Subclasses implement `f_map(x)` and optionally `feature_names()`.
    """
    def __init__(self):
        super().__init__()
        self.realignment = False

    def extract_features(self, x: torch.Tensor):
        """
        x: arbitrary shape; subclasses know how to interpret it.
        Returns: list of features, each with shape [B, F, D].
        """
        f_result = self.f_map(x)
        if not isinstance(f_result, list):
            f_result = [f_result]
        return f_result

    def feature_names(self):
        return []


def coords_to_vec(coords: torch.Tensor, atom_mask: torch.Tensor | None = None):
    """
    coords:
      - [B, N, 3]          (single feature per sample)
      - or [B, K, N, 3]    (K negatives per sample)

    atom_mask:
      - [B, N] or [B, K, N], matching coords except last dim.

    returns:
      - [B, 1, 3N]   if coords.ndim == 3
      - [B, K, 3N]   if coords.ndim == 4
    """
    if atom_mask is not None:
        # broadcast mask to coords shape (except last dim)
        while atom_mask.ndim < coords.ndim - 1:
            atom_mask = atom_mask.unsqueeze(1)
        coords = coords * atom_mask.unsqueeze(-1)

    if coords.ndim == 3:
        B, N, _ = coords.shape
        vec = coords.reshape(B, 1, N * 3)
    elif coords.ndim == 4:
        B, K, N, _ = coords.shape
        vec = coords.reshape(B, K, N * 3)
    else:
        raise ValueError(f"Unexpected coords shape {coords.shape}")

    return vec


@persistent_class
class CoordGlobal(FeatureExtractor):
    """
    Feature extractor for coordinate tensors [B, N, 3].

    Produces a single global feature [B, 1, 3N] suitable for group_contra_loss.
    """
    def __init__(self):
        super().__init__()
        self.realignment = True

    def f_map(self, x: torch.Tensor, atom_mask: torch.Tensor | None = None):
        # x: [B, N, 3] or [B, K, N, 3]
        return coords_to_vec(x, atom_mask=atom_mask)

    def name(self):
        return "coords_global"

    def feature_names(self):
        return ["coords_global"]


@persistent_class
class CoordLocal(FeatureExtractor):
    """
    Feature extractor that computes local coordinates.

    Produce a cloud of local coordinates [B, N, 3] or [B, K, N, 3] centered at each atom.

    Intuitively, this should help the model learn individual atom's distribution
    """

    def __init__(self):
        super().__init__()

    def f_map(self, x: torch.Tensor, atom_mask: torch.Tensor | None = None):
        # x: [B, N, 3] or [B, K, N, 3]
        if x.ndim == 4:
            local_coords = einops.rearrange(x, "b k n c -> (b n) k c")  # [B * N, K, 3]
            return local_coords  # [B * N, K, 3]
        elif x.ndim == 3:
            local_coords = einops.rearrange(x, "b n c -> (b n) 1 c")  # [B * N, 1, 3]
            return local_coords  # [B * N, 1, 3]
        else:
            raise ValueError(f"Unexpected x shape {x.shape}")

    def name(self):
        return "coords_local"

    def feature_names(self):
        return ["coords_local"]

class GlobalPatches(FeatureExtractor):
    """
    Feature extract that segments the coordinates into patches 
    and calculate the patches' centers' pairwise distance.
    """

    def __init__(self, patch_size=32):
        super().__init__()
        self.patch_size = patch_size

    def get_patch_centers_dists(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, N, 3]
        Returns:
            patch_dists: [B, num_patches * num_patches]
        """
        B, N, _ = x.shape
        device = x.device
        num_patches = N // self.patch_size

        x_patches = x[:, : num_patches * self.patch_size, :].reshape(
            B, num_patches, self.patch_size, 3
        )  # [B, num_patches, patch_size, 3]
        patch_centers = x_patches.mean(dim=2)  # [B, num_patches, 3]
        patch_dists = torch.cdist(
            patch_centers, patch_centers
        )  # [B, num_patches, num_patches]
        patch_dists = einops.rearrange(
            patch_dists, "b p1 p2 -> b (p1 p2)"
        )  # [B, num_patches * num_patches]

        return patch_dists

    def f_map(self, x: torch.Tensor, atom_mask: torch.Tensor | None = None):
        """
        x : [B, N, 3] or [B, K, N, 3]
        Returns:
            patch_dists: [B, 1, num_patches * num_patches] or [B, K, num_patches * num_patches]
        """
        if x.ndim == 4:
            B, K, N, _ = x.shape
            x = einops.rearrange(x, "b k n c -> (b k) n c")

            patch_dists = self.get_patch_centers_dists(x)  # [(B*K), num_patches * num_patches]
            patch_dists = einops.rearrange(
                patch_dists, "(b k) d -> b k d", b=B, k=K
            )  # [B, K, num_patches * num_patches]
            return patch_dists  # [B, K, num_patches * num_patches]
        elif x.ndim == 3:
            B, N, _ = x.shape

            patch_dists = self.get_patch_centers_dists(x)  # [B, num_patches * num_patches]
            patch_dists = einops.rearrange(
                patch_dists, "b d -> b 1 d"
            )  # [B, 1, num_patches * num_patches]
            return patch_dists  # [B, 1, num_patches * num_patches]
        else:
            raise ValueError(f"Unexpected x shape {x.shape}")

    def name(self):
        return f"coords_global_patches_{self.patch_size}"

    def feature_names(self):
        return [f"coords_global_patches_{self.patch_size}"]


class LocalDists(FeatureExtractor):
    """
    Feature extractor that computes local distances.

    For each point, computes the distances to its left and right k nearest neighbors (assuming the sequence of atoms is in order).
    """
    def __init__(self, k=5, is_sorted=False):
        super().__init__()
        self.k = k
        self.is_sorted = is_sorted

    import torch

    def get_knn_dists(self, x: torch.Tensor, k : int) -> torch.Tensor:
        """
        x : [B, N, 3]
        Returns:
            knn_dists: [B, N, 2k]
                For each position i, distances to up to k neighbors on the left
                and k on the right. Distances are sorted ascending. If there are
                fewer than 2k neighbors (near sequence ends), the missing
                entries are padded with 0.
        """
        B, N, _ = x.shape
        device = x.device

        # Full pairwise distance matrix on GPU: [B, N, N]
        dists = torch.cdist(x, x)  # torch.cdist stays on device

        left_offsets = torch.arange(-k, 0, device=device)
        right_offsets = torch.arange(1, k + 1, device=device)
        offsets = torch.cat([left_offsets, right_offsets], dim=0)  # [2k]

        center = torch.arange(N, device=device).unsqueeze(-1)      # [N, 1]
        neighbor_idx = center + offsets                            # [N, 2k]

        valid_mask = (neighbor_idx >= 0) & (neighbor_idx < N)      # [N, 2k]
        neighbor_idx_clamped = neighbor_idx.clamp(0, N - 1)        # [N, 2k]
        gather_idx = neighbor_idx_clamped.unsqueeze(0).expand(B, -1, -1)
        knn_dists = dists.gather(dim=2, index=gather_idx)          # [B, N, 2k]

        inf = torch.tensor(float("inf"), device=device, dtype=knn_dists.dtype)
        knn_dists = knn_dists.masked_fill(~valid_mask.unsqueeze(0), inf)

        if self.is_sorted:
            knn_dists, _ = torch.sort(knn_dists, dim=-1)

        # Replace +inf (i.e. padded neighbors) with 0 distance
        knn_dists = knn_dists.masked_fill(torch.isinf(knn_dists), 0.0)

        return knn_dists

    # atom_mask should be padding not resolved mask
    def f_map(self, x: torch.Tensor, atom_mask: torch.Tensor | None = None):
        # x: [B, N, 3] or [B, K, N, 3]
        # output : [B, 1, 2k * 3] or [B, K, 2k * 3]
        if x.ndim == 4:
            B, K, N, _ = x.shape
            x = einops.rearrange(x, "b k n c -> (b k) n c")

            dists = self.get_knn_dists(x, self.k)  # [(B*K), N, 2k]
            dists = einops.rearrange(dists, "(b k) n d -> b k (n d)", b=B, k=K)  # [B, K, N*2k]
            return dists # [B, K, N*2k]
        elif x.ndim == 3:
            B, N, _ = x.shape

            dists = self.get_knn_dists(x, self.k)  # [B, N, 2k]
            dists = einops.rearrange(dists, "b n d -> b (n d)")  # [B, N*2k]
            return dists.unsqueeze(1)  # [B, 1, N*2k]
        else:
            raise ValueError(f"Unexpected x shape {x.shape}")
        
    def name(self):
        return f"coords_local_k{self.k}"

    def feature_names(self):
        return [f"coords_local_k{self.k}"]


class LocalTriangleDists(FeatureExtractor):
    """
    Feature extractor that computes local triangle features.

    For each position i, we consider triangles of the form (i, i - d1, i + d2)
    where d1, d2 in {1, ..., k}, as long as the indices are in-bounds.

    Features per triangle:
        - Area of the triangle.
    """
    def __init__(self, k=4):
        super().__init__()
        self.k = k

    def _triangle_features_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, 3]
        Returns:
            feats: [B, N, T * F]
                where T is number of triangles per center,
                F is number of features per triangle (here 1).
        """
        B, N, _ = x.shape
        device = x.device
        k = self.k

        # Sequence indices
        centers = torch.arange(N, device=device)  # [N]

        # Offsets: left in [-k, ..., -1], right in [1, ..., k]
        left_offsets = torch.arange(1, k + 1, device=device)
        right_offsets = torch.arange(1, k + 1, device=device)

        # We'll build triangle features by looping over (d1, d2).
        # k is small (<= 4 or so), so this is fine.
        per_center_feats = []

        eps = 1e-8

        for d1 in range(1, k + 1):
            for d2 in range(1, k + 1):
                idx_center = centers  # [N]
                idx_left   = centers - d1
                idx_right  = centers + d2

                # Valid triangles: all indices in [0, N)
                valid = (idx_left >= 0) & (idx_right < N)  # [N]

                # Clamp for gathering
                idx_left_clamp  = idx_left.clamp(0, N - 1)
                idx_right_clamp = idx_right.clamp(0, N - 1)

                # Shape [B, N, 3]
                c = x[:, idx_center]       # centers
                l = x[:, idx_left_clamp]   # left neighbors
                r = x[:, idx_right_clamp]  # right neighbors

                # Vectors from center
                v_l = l - c   # [B, N, 3]
                v_r = r - c   # [B, N, 3]

                # area via cross product
                cross = torch.cross(v_l, v_r, dim=-1)  # [B, N, 3]
                area = cross.norm(dim=-1)        # [B, N]

                # Stack features: [B, N, 1]
                feats = torch.stack([area], dim=-1)

                # Zero out invalid ones
                valid_mask = valid.view(1, N, 1)  # [1, N, 1]
                feats = feats * valid_mask.to(feats.dtype)

                per_center_feats.append(feats)  # list of [B, N, 1]

        # Concatenate along "triangle" axis: [B, N, T*1]
        feats_all = torch.cat(per_center_feats, dim=-1)
        return feats_all

    def f_map(self, x: torch.Tensor, atom_mask: torch.Tensor | None = None):
        # x: [B, N, 3] or [B, K, N, 3]
        # Output:
        #   - [B, 1, N*T*F]  if x.ndim == 3
        #   - [B, K, N*T*F]  if x.ndim == 4
        if x.ndim == 4:
            B, K, N, _ = x.shape
            x_flat = einops.rearrange(x, "b k n c -> (b k) n c")  # [(B*K), N, 3]

            feats_flat = self._triangle_features_3d(x_flat)       # [(B*K), N, T*F]
            feats = einops.rearrange(
                feats_flat, "(b k) n d -> b k (n d)", b=B, k=K
            )  # [B, K, N*T*F]
            return feats

        elif x.ndim == 3:
            B, N, _ = x.shape
            feats = self._triangle_features_3d(x)                  # [B, N, T*F]
            feats = einops.rearrange(feats, "b n d -> b 1 (n d)")  # [B, 1, N*T*F]
            return feats

        else:
            raise ValueError(f"Unexpected x shape {x.shape}")

    def name(self):
        return f"coords_local_triangle_k{self.k}"

    def feature_names(self):
        return [f"coords_local_triangle_k{self.k}"]


def build_feature_modules(
    input_shape=None,
    has_global=True,
    has_coord=False,
    has_local=[],
    global_patches=[],
    triangle_local_ks=[],
):
    """
    Coord-only version.

    Args:
        input_shape: unused for coords; kept for API compatibility.
        ...

    Returns:
        nn.ModuleList([CoordGlobal(), ...])
    """
    feature_extractors = []

    # Always include a global flattened coord feature
    if has_global:
        feature_extractors.append(CoordGlobal())
    if has_coord:
        feature_extractors.append(CoordLocal())
    # Local distance features
    for k in has_local:
        feature_extractors.append(LocalDists(k=k))

    for p_size in global_patches:
        feature_extractors.append(GlobalPatches(patch_size=p_size))

    for k in triangle_local_ks:
        feature_extractors.append(LocalTriangleDists(k=k))

    # If later you want more coord-based features, append them here.

    return nn.ModuleList(feature_extractors)



@persistent_class
class FeatureEnergyLoss(nn.Module):
    def __init__(self, contra_dict=None):
        super().__init__()
        if contra_dict is None:
            contra_dict = dict(kernel_type="attn_new", use_base=True, use_scale=False)
        self.contra_dict = contra_dict
        self.feats = build_feature_modules(
            has_global=self.contra_dict.get("has_global", True),
            has_coord=self.contra_dict.get("has_coord", False),
            has_local=self.contra_dict.get("local_ks", []),
            global_patches=self.contra_dict.get("global_patches", []),
            triangle_local_ks=self.contra_dict.get("triangle_local_ks", []),
        )

    def forward(
        self,
        target: torch.Tensor,       # [B, N, 3]
        recon: torch.Tensor,        # [B, R, N, 3]
        fixed_neg: torch.Tensor | None = None,   # [B, K, N, 3] or None
        atom_mask: torch.Tensor | None = None,   # [B, N] (we’ll ignore for negs for now or broadcast)
        scale_override: torch.Tensor | None = None, # single scalar
    ):
        # print(f"[DEBUG] Shapes of everything: {target.shape=}, {recon.shape=}, {fixed_neg.shape if fixed_neg is not None else None}, {atom_mask.shape if atom_mask is not None else None}")
        # weights (optional; left as ones for now)
        B = target.shape[0]

        # prepare masks
        target_mask = atom_mask # [B, N]
        recon_mask  = atom_mask.unsqueeze(1) # [B, 1, N]
        neg_mask    = None  # you can extend this later if you also store mask for negatives

        total_loss = 0.0
        all_info = {}

        # right now we only have one feature module (coords_global)
        for i, feat in enumerate(self.feats):
            # features
            with torch.no_grad():
                target_f = feat.f_map(target, atom_mask=target_mask)          # [B, 1, S]
                if fixed_neg is not None:
                    fixed_neg_f = feat.f_map(fixed_neg, atom_mask=neg_mask)   # [B, K, S]
                else:
                    fixed_neg_f = None

            recon_f = feat.f_map(recon, atom_mask=recon_mask) # [B, R, S]

            target_w = torch.ones_like(target_f[:, :, 0])  # [B, 1]
            recon_w  = torch.ones_like(recon_f[:, :, 0])   # [B, R]
            fixed_neg_w = None

            loss, info = group_contra_loss(
                pos=target_f,     # [B, 1, S]
                gen=recon_f,      # [B, R, S]
                neg=fixed_neg_f,  # [B, K, S] or None
                pos_w=target_w,
                gen_w=recon_w,
                neg_w=fixed_neg_w,
                scale_override=scale_override,
                return_info=True,
                realignment=feat.realignment,
                atom_mask=atom_mask,
                **self.contra_dict,
            )

            if loss.shape[0] != B:
                assert loss.shape[0] % B == 0, "Batch size mismatch in loss computation."
                loss = einops.rearrange(loss, "(b r) -> b r", b=B).mean(dim=1)  # [B]

            total_loss = total_loss + loss  # [B]

            prefix = feat.name() if hasattr(feat, "name") else f"feat_{i}"
            all_info[f"{prefix}/loss"] = loss.mean()
            for k, v in info.items():
                all_info[f"{prefix}/{k}"] = v

        total_loss = total_loss / len(self.feats)
        return total_loss, all_info
