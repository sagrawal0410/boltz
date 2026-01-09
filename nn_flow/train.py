# %%
import torch
import torch.nn as nn
import contextlib
import os, time, contextlib
from copy import deepcopy
import torch
from torch.profiler import profile as torch_profile, record_function, ProfilerActivity, schedule as prof_schedule, tensorboard_trace_handler

from train_both import vae_from_combined
from utils.misc import EasyDict, set_seed, easydict_to_dict
from utils.distributed_utils import (
    init_distributed_mode,
    get_rank,
    get_world_size,
    get_local_rank,
    aggregate_info_across_gpus,
)
from utils.global_path import wandb_project, wandb_entity
from utils.ema import EMA
from utils.ckpt_utils import load_last_ckpt, load_ckpt_epoch, ckpt_epoch_numbers
from dataset import postprocess
from tqdm import tqdm
import random
from copy import deepcopy
from utils.fid import visualize_imagenet_samples
import time
import numpy as np
import datetime

from model.vit_ae import Encoder, Decoder
from model.mlp import CondEmbed
from utils.misc import EasyDict
from features import build_feature_modules
from einops import repeat, rearrange

from utils.augment import getEDMAugment
from utils.misc import LinearWarmupCosineDecayLR

from utils.fid import eval_fid
from utils.ckpt_utils import save_ckpt
from utils.profile import print_module_summary
from utils.misc import InfiniteSampler
from utils.ema import EDM2Schedule
import datetime

import os
from torch.nn.parallel import DistributedDataParallel as DDP
from config import (
    load_config,
    build_model_dict,
)
from model.vae import VAE
from memory_bank import MemoryBank
from utils.ckpt_utils import load_last_ckpt, ckpt_epoch_numbers, load_ckpt_epoch

# torch._dynamo.config.verbose = True  # Enables verbose compile logs
# torch._dynamo.config.optimize_ddp = False
torch.set_float32_matmul_precision("high")  # Enable TF32 tensor cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True   # 如果网络里有卷积


# %%
def train_vae(
    model,
    dataset_name,
    optimizer,
    n_steps,
    total_batch_size=128,
    recon_per_step=1000,
    save_per_step=2000,
    eval_gen_per_step=1000,
    eval_recon_per_step=1000,
    logger=None,
    forward_dict=dict(),
    eval_dataset=None,
    train_dataset=None,
    eval_bsz_per_gpu=768,
    lr_schedule=EasyDict(
        lr=2e-4,
        warmup_kimg=10000,
        total_kimg=200000,
        clip_grad=10.0,
    ),
    augmentation_kwargs=EasyDict(enable=False, augmentation_dim=9, kwargs=EasyDict()),
    drop_class=0.1,
    load_dict=EasyDict(
        run_id="",
        epoch="latest",
        continue_training=False,
    ),
    use_bf16=False,
    preprocess_fn=lambda x, label_list: x,
    postprocess_fn=postprocess,
    cond_mem_bank_size=0,  # capacity of conditional memory bank
    uncond_mem_bank_size=0,  # capacity of unconditional memory bank
    gen_mem_bank_size=0,
    n_cond_samples=0,  # number of conditional samples per datapoint
    n_uncond_samples=0,
    n_gen_bank_samples=0,
    self_guidance=False, # when enabled, use uncondtional self samples for guidance
    min_cfg_scale=1.0,
    max_cfg_scale=1.0,
    neg_cfg_sampler_pow=0.0, # probability of sampling x is proportional to x^-neg_cfg_sampler_pow;
    cfg_scale=None,  # deprecated; if provided, treated as min=max=cfg_scale
    uncond=False,
    compile_model=False, #deprecated
    n_classes=1000,
    n_gen_per_label=1,
    push_warmup_steps=0, # number of steps to warmup the push
    warmup_push_size=0, # number of pushed samples during warmup
    online_gen=False,
    use_old_noise=True,  # if has gen_bank: rather to sample from the old noise
    use_online_samples=False,
    persist_latest_ckpts=1, # persist latest n ckpts
    persist_ckpts_every=10**9, # persist ckpts every n steps
    job_name="",  # the unique pointer for saving & loading ckpts.
    eval_base_cfg=1.5, 
    eval_cfg_list=[1.0, 1.5, 2.0],
    eval_on_init=False,
    memory_bank_cpu=True,
    num_workers=16,
    prefetch_factor=4,
    profile_train: bool = False,                      # master switch
    profile_to_tb: bool = True,                       # write TensorBoard traces
    profile_dir: str = "profiles",                    # output folder
    profile_wait_warmup_active_repeat=(1, 1, 3, 1),   # (wait, warmup, active, repeat)
    use_scaler=False, 
    ema_decays = [0.999, 0.9995, 0.9998, 0.9999]

    
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_amp = use_bf16 and torch.cuda.is_available()
    dtype = (
        torch.bfloat16
        if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    print(f"Using {dtype} precision")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if use_amp:
        print(f"Using AMP training with {dtype} precision")
    
    init_profile_train = profile_train
    def _sync_if_needed():
        if init_profile_train and torch.cuda.is_available():
            torch.cuda.synchronize()

    class Timer:
        def __init__(self):
            self.t0 = None
        def start(self):
            _sync_if_needed()
            self.t0 = time.perf_counter()
        def stop(self):
            _sync_if_needed()
            return time.perf_counter() - self.t0

    def empty_aug_labels(bsz):
        return torch.zeros(
            bsz,
            augmentation_kwargs.augmentation_dim,
            device=device,
            dtype=torch.float32,
        )

    # if augmentation_kwargs.enable
    # augmenter = getEDMAugment(**augmentation_kwargs.kwargs)
    # else:
    def augmenter(x):
        return x, empty_aug_labels(x.shape[0])

    distributed = False
    if torch.distributed.is_initialized():
        print("Distributed training initialized")
        distributed = True
        local_rank = get_rank()
        world_size = get_world_size()
    else:
        print("Distributed training not initialized")
        local_rank = 0
        world_size = 1

    set_seed(local_rank)

    base_model = model
    ema_models = {decay: EMA(base_model, decay) for decay in ema_decays}
    is_main_process = local_rank == 0

    pbar = tqdm(range(n_steps)) if is_main_process else range(n_steps)
    start_step_idx = 0
    load_step = 0
    step = 0
    loaded_ckpt = None

    def load(run_id, epoch, continue_training, mark_used=True, eval_on_init=False):
        nonlocal step, pbar, load_step, loaded_ckpt, start_step_idx
        if run_id == "":
            return False
        all_epochs = ckpt_epoch_numbers(run_id)
        if len(all_epochs) == 0:
            print(f"No checkpoint found for job {run_id}")
            return False
        print("Loading from ckpt: ", run_id, epoch)
        if epoch == "latest":
            loaded = load_last_ckpt(run_id, mark_used=mark_used)  # run_id : should be the job name.
            step = int(all_epochs[-1])
        else:
            loaded = load_ckpt_epoch(run_id, epoch, mark_used=mark_used)
            step = int(epoch)
        print("Loading from ckpt: ", step)
        info_x = base_model.load_state_dict(
            (
                loaded["model"]
                if isinstance(loaded["model"], dict)
                else loaded["model"].state_dict()
            ),
            strict=False,
        )
        print(
            "Missing keys:",
            info_x.missing_keys,
            "Unexpected keys:",
            info_x.unexpected_keys,
        )
        for decay, ema_model in ema_models.items():
            key = f"ema_model_{decay}"
            if key in loaded:
                info = ema_model.model.load_state_dict(
                    (
                        loaded[key]
                        if isinstance(loaded[key], dict)
                        else loaded[key].state_dict()
                    ),
                    strict=False,
                )
                print(
                    f"EMA {decay} - Missing keys:",
                    info.missing_keys,
                    "Unexpected keys:",
                    info.unexpected_keys,
                )
            else:
                ema_model.model.load_state_dict(base_model.state_dict(), strict=False)

        loaded_ckpt = loaded

        # Restore optimizer if present
        if "optimizer" in loaded:
            try:
                optimizer.load_state_dict(loaded["optimizer"])  # type: ignore[arg-type]
                print("Optimizer state loaded.")
            except Exception as e:
                print("Failed to load optimizer state:", e)
        if not eval_on_init:
            step += 1
        if continue_training:
            pbar = tqdm(range(step, n_steps))
            start_step_idx = step
            load_step = step
        return True

    load_dict = EasyDict(load_dict)
    resumed_from_initial_ckpt = load(load_dict.run_id, load_dict.epoch, load_dict.continue_training, eval_on_init=eval_on_init)
    # When resuming from the job's latest, avoid marking as used to not pin ephemeral latest unnecessarily
    resumed_from_job = load(job_name, "latest", True, mark_used=False, eval_on_init=False) 
    
    model_copy = deepcopy(base_model)

    def copy_params(src_model, dst_model):
        model_state_dict = src_model.state_dict()
        updated_list = {
            k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()
        }
        info = dst_model.load_state_dict(updated_list, strict=False)
        print(
            "Missing keys:", info.missing_keys, "Unexpected keys:", info.unexpected_keys
        )

    if distributed and world_size > 1:
        model = DDP(
            model, device_ids=[get_local_rank()],
        )  # don't use get_local_rank() here! won't work for slurm.
        model_without_ddp = model.module
    else:
        model_without_ddp = model.cuda()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    is_main_process = local_rank == 0

    autocast_context = (
        torch.cuda.amp.autocast(dtype=dtype) if use_amp else contextlib.nullcontext()
    )

    if n_cond_samples > 0:
        memory_bank = MemoryBank(
            num_classes=(n_classes + 1),
            max_size=max(cond_mem_bank_size, n_cond_samples),
            device=device if not memory_bank_cpu else "cpu",
        )
        if loaded_ckpt is not None and "memory_bank" in loaded_ckpt:
            try:
                memory_bank.load_state_dict(loaded_ckpt["memory_bank"])  # type: ignore[arg-type]
                print("Restored memory_bank from checkpoint.")
            except Exception as e:
                print("Failed to restore memory_bank:", e)

    if n_gen_bank_samples > 0:
        gen_memory_bank = MemoryBank(
            num_classes=(n_classes + 1),
            max_size=max(n_gen_bank_samples, gen_mem_bank_size),
            device=device if not memory_bank_cpu else "cpu",
        )
        if loaded_ckpt is not None and "gen_memory_bank" in loaded_ckpt:
            try:
                gen_memory_bank.load_state_dict(loaded_ckpt["gen_memory_bank"])  # type: ignore[arg-type]
                print("Restored gen_memory_bank from checkpoint.")
            except Exception as e:
                print("Failed to restore gen_memory_bank:", e)

    def push_moco(labels):
        if n_gen_bank_samples == 0:
            return
        with torch.inference_mode():
            labels = repeat(labels, "b -> (n b)", n=n_gen_per_label)
            if online_gen:
                gen_dict = base_model.generate(labels)
            else:
                gen_dict = ema_models[ema_decays[0]].model.generate(labels) # Assuming the first EMA is the primary one for generation
            samples, noise = gen_dict.pop("samples"), gen_dict.pop("noise")
            gen_memory_bank.add(
                {
                    "samples": samples,
                    "idx": torch.randint(0, 10**9, (samples.shape[0],), device=device),
                    "noise": noise,
                },
                labels,
            )

    push_moco(
        torch.zeros(1, dtype=torch.int64, device=device)
    )  # initialize the moco bank

    all_samples_bank = MemoryBank(
        num_classes=1,
        max_size=max(uncond_mem_bank_size, n_uncond_samples),
        device=device if not memory_bank_cpu else "cpu",
    )
    if loaded_ckpt is not None and "all_samples_bank" in loaded_ckpt:
        try:
            all_samples_bank.load_state_dict(loaded_ckpt["all_samples_bank"])  # type: ignore[arg-type]
            print("Restored all_samples_bank from checkpoint.")
        except Exception as e:
            print("Failed to restore all_samples_bank:", e)

    def empty_class_labels(bsz):
        return torch.ones(bsz, device=device, dtype=torch.int64) * n_classes

    def push_samples(imgs, labels):
        batch_size = imgs.shape[0]
        original_sample_indices = torch.randint(
            0, 10**9, (batch_size,)
        )  # Large range for unique indices

        if n_cond_samples > 0:
            memory_bank.add({"data": imgs, "idx": original_sample_indices}, labels)
        if n_uncond_samples > 0:
            all_samples_bank.add(
                {"data": imgs, "idx": original_sample_indices}, torch.zeros_like(labels)
            )
    # Resolve cfg scale controls for training and eval
    # Backward-compat: if deprecated cfg_scale is provided, map to fixed min=max
    if cfg_scale is not None:
        min_cfg_scale = float(cfg_scale)
        max_cfg_scale = float(cfg_scale)

    def sample_cfg_scales(bsz):
        frac = torch.rand(bsz, device=device)
        pw = 1 - neg_cfg_sampler_pow
        if abs(pw) < 1e-6:
            return torch.exp(torch.log(min_cfg_scale) + frac * (torch.log(max_cfg_scale) - torch.log(min_cfg_scale)))
        return (min_cfg_scale ** pw + frac * (max_cfg_scale ** pw - min_cfg_scale ** pw)) ** (1/pw)

    def samples_to_inputs(samples):
        """
        Args:
            samples: tuple of (imgs, labels)
        Returns:
            args: list of tensors
            kwargs: dictionary of keyword arguments
            Guarantee: kwargs['c'] will be a tensor of shape (batch_size, )
        """
        imgs, labels = samples

        if uncond:
            labels = torch.zeros_like(labels)

        # Generate random indices for this batch
        push_samples(imgs, labels)

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        push_moco(labels)

        samples, aug_labels = augmenter(imgs)  # be careful

        if n_cond_samples > 0:
            samples_dict = memory_bank.sample(labels, n_samples=n_cond_samples)
            samples = samples_dict["data"]

        label_list = labels
        if self_guidance and drop_class > 0:
            label_list = torch.where(torch.rand(labels.shape[0], device=device) < drop_class, empty_class_labels(labels.shape[0]), labels)
        samples = preprocess_fn(samples, label_list)
        samples = samples.to(device, non_blocking=True)
        args = [samples]

        batch_cfg_scale = sample_cfg_scales(labels.shape[0])

        neg_samples = None
        neg_w = 0.0
        neg_w_list = []
        noise_dict = None
        old_samples = None

        if n_gen_bank_samples > 0:
            if use_online_samples:
                with torch.inference_mode():
                    gen_labels = repeat(labels, "b -> (n b)", n=n_gen_bank_samples)
                    gen_dict = base_model.generate(gen_labels)
                    neg_samples = gen_dict.pop("samples").to(device)
                    neg_samples = rearrange(
                        neg_samples, "(n b) ... -> b n ...", n=n_gen_bank_samples
                    )
            else:
                print("Using gen_memory_bank", n_gen_bank_samples)
                neg_samples = gen_memory_bank.sample(
                    labels, n_samples=n_gen_bank_samples
                )["samples"].to(device)

            neg_w_list.append(
                torch.ones(neg_samples.shape[0], neg_samples.shape[1], device=device)
            )
            if use_old_noise:
                gen_old_dict = gen_memory_bank.sample(
                    labels, n_samples=forward_dict.recon
                )
                old_samples = gen_old_dict["samples"].to(device)
                noise_dict = gen_old_dict["noise"].to(device)

        if n_uncond_samples > 0:
            # Correctly format the avoid_idx_list and use the original indices
            if self_guidance:
                with torch.inference_mode():
                    uncond_samples = base_model.generate(empty_class_labels(labels.shape[0] * n_uncond_samples))["samples"].to(device)
                    uncond_samples = rearrange(
                        uncond_samples, "(n b) ... -> b n ...", n=n_uncond_samples
                    )
            else:
                uncond_samples_dict = all_samples_bank.sample(
                    torch.zeros_like(labels),
                    n_samples=n_uncond_samples,
                )
                uncond_samples = uncond_samples_dict["data"].to(device, non_blocking=True)
            num_current_negs = 0 if neg_samples is None else neg_samples.shape[1]
            uncond_w_vec = (
                (batch_cfg_scale - 1)
                * (forward_dict.recon - 1 + num_current_negs)
            ) / uncond_samples.shape[1]
            w = uncond_w_vec.unsqueeze(1).expand(uncond_samples.shape[0], uncond_samples.shape[1])
            neg_w_list.append(w)
            neg_samples = (
                uncond_samples
                if neg_samples is None
                else torch.cat([neg_samples, uncond_samples], dim=1)
            )

        if neg_samples is not None and neg_w_list:
            neg_w = torch.cat(neg_w_list, dim=1)

        kwargs = dict(
            c=label_list,
            **easydict_to_dict(forward_dict),
            neg_samples=neg_samples,
            neg_w=neg_w,
            old_samples=old_samples,
            noise_dict=noise_dict,
            cfg_scale=batch_cfg_scale,
        )
        return args, kwargs

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    model_without_ddp.train()

    dataset_sampler = InfiniteSampler(
        dataset=train_dataset, rank=local_rank, num_replicas=world_size, seed=0
    )
    local_batch_size = total_batch_size // max(1, world_size)
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=dataset_sampler,
            batch_size=local_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,                 # 让 worker 常驻
            prefetch_factor=prefetch_factor,                       # 默认2，适当加大
        )
    )

    # profiler 
    profile_train = profile_train and (local_rank == 0)
    prof = None
    if profile_train:
        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)

        # 调度窗口：建议抓短窗口
        w, wu, a, r = profile_wait_warmup_active_repeat  # e.g. (0,1,8,1)
        sched = prof_schedule(wait=w, warmup=wu, active=a, repeat=r)

        # 目录：在 run_dir/profile_dir 下分别放 TB 与 Chrome
        base_dir   = os.path.join(getattr(logger, "run_dir", "."), profile_dir)
        tb_dir     = os.path.join(base_dir, "tb")
        chrome_dir = os.path.join(base_dir, "chrome")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(chrome_dir, exist_ok=True)
        tb_handler = tensorboard_trace_handler(tb_dir, worker_name=f"rank{local_rank}") if profile_to_tb else None

        PROF_RECORD_SHAPES  = True
        PROF_PROFILE_MEMORY = True
        PROF_WITH_STACK     = True
        PROF_WITH_MODULES   = True

        def _dump_key_averages_tables(p, logger, topk=50):
            try:
                ka_cuda = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=topk)
            except Exception:
                ka_cuda = None
            try:
                ka_cpu  = p.key_averages().table(sort_by="self_cpu_time_total",  row_limit=topk)
            except Exception:
                ka_cpu = None

            out_dir = os.path.join(base_dir, "keyavg")
            os.makedirs(out_dir, exist_ok=True)
            ts = int(time.time())
            if ka_cuda:
                with open(os.path.join(out_dir, f"top{topk}_cuda_{ts}.txt"), "w") as f:
                    f.write(ka_cuda)
            if ka_cpu:
                with open(os.path.join(out_dir, f"top{topk}_cpu_{ts}.txt"), "w") as f:
                    f.write(ka_cpu)

        def _on_trace_ready(p):
            if tb_handler is not None:
                tb_handler(p)
            try:
                out_json = os.path.join(chrome_dir, f"rank{local_rank}_{int(time.time())}.json")
                p.export_chrome_trace(out_json)
            except RuntimeError as e:
                if "already saved" not in str(e):
                    raise
            try:
                _dump_key_averages_tables(p, logger, topk=50)
            except Exception:
                pass

        prof = torch_profile(
            activities=acts,
            schedule=sched,
            on_trace_ready=_on_trace_ready,   # <--- 只用我们自定义的 handler；不要再额外传一次
            record_shapes=PROF_RECORD_SHAPES,
            profile_memory=PROF_PROFILE_MEMORY,
            with_stack=PROF_WITH_STACK,
            with_modules=PROF_WITH_MODULES,
            with_flops=False,
        )
        prof.__enter__()  # 注意：只有 rank0 会进入；其它 rank 上 prof 仍是 None

    start_time = time.time()

    for steps in pbar:
        total_timer = Timer()
        total_timer.start()
        print("steps", steps)
        # if steps == start_step_idx and debug_speed:
        #     for x in range(2):
        #         with torch.no_grad():
        #             if is_main_process:
        #                 old_time = time.time()
        #                 imgs, labels = next(dataset_iterator)
        #                 args, kwargs = samples_to_inputs((imgs[:2], labels[:2]))
        #                 copy_params(base_model, model_copy)
        #                 table = print_module_summary(model_copy, inputs=args, kwargs=kwargs, max_nesting=4)
        #                 logger.log_dict({"model_summary": table})
        #                 logger.log_dict({"profile_time_per_kimg": (time.time() - old_time) / (1000 * total_batch_size / 1000)})
        lr = LinearWarmupCosineDecayLR(
            (steps + 1 - (load_step if not resumed_from_job else 0)) * total_batch_size / 1000, lr_schedule
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logger.set_step(steps)

        # FID gen

        do_eval = (steps % eval_gen_per_step == 0 or steps == n_steps - 1) or (eval_on_init and steps == start_step_idx)

        if eval_dataset is not None and do_eval:  # and steps != 0:
            with torch.inference_mode():

                def map_label(labels):
                    res = labels.to(device)
                    if uncond:
                        res = torch.zeros_like(res)
                    return res

                old_time = time.time()
                base_model.eval()
                for ema in ema_models.values():
                    ema.model.eval()

                def model_to_gen_func(model, temp, cfg_val):
                    return lambda x: postprocess_fn(
                        model.generate(map_label(x[1]), temp=temp, cfg_scale=cfg_val)["samples"],
                        label_list=map_label(x[1]),
                    )

                def class_only_gen_func(model, temp, cfg_val):
                    return lambda x: postprocess_fn(
                        model.generate(x, temp=temp, cfg_scale=cfg_val)["samples"],
                        label_list=map_label(x),
                    )

                def eval_fid_func(model, cfg_val, samples=5000, log_prefix="gen_model", to_visualize=True, log_folder_name="fid", **kwargs):
                    results = eval_fid(
                        model_to_gen_func(model, 1.0, cfg_val),
                        eval_dataset,
                        logger,
                        log_prefix=f"{log_prefix}",
                        dataset=dataset_name,
                        gpu_batch_size=eval_bsz_per_gpu,
                        total_samples=samples,
                        log_folder_name=log_folder_name,
                        **kwargs
                    )
                    if to_visualize and n_classes >= 1000:
                        visualize_imagenet_samples(class_only_gen_func(model, 1.0, cfg_val), logger, log_prefix=f"{log_prefix}_cfg{cfg_val}")

                    return results

                if steps != 0:
                    best_fid_base, best_cfg_base = 10**9, eval_base_cfg
                    best_fid_ema, best_cfg_ema = 10**9, eval_base_cfg

                    fid_dict = eval_fid_func(base_model, eval_base_cfg, log_prefix="model", to_visualize=True, log_folder_name=f"CFG{eval_base_cfg}")
                    best_fid_base = fid_dict["fid"]

                    best_ema, best_isc = 0, float('-inf')

                    for decay, ema_model in ema_models.items():
                        fid_dict = eval_fid_func(ema_model.model, eval_base_cfg, log_prefix=f"ema_{decay}", to_visualize=True, log_folder_name=f"CFG{eval_base_cfg}")
                        if fid_dict["isc_mean"] > best_isc:
                            best_isc = fid_dict["isc_mean"]
                            best_ema = decay
                            best_fid_ema = fid_dict["fid"]
                        
                    if steps % (5 * eval_gen_per_step) == 0 or (eval_on_init and steps == start_step_idx):
                        first_ema = ema_models[best_ema].model
                        for cfg in eval_cfg_list:
                            if cfg == eval_base_cfg:
                                continue
                            base_fid_dict = eval_fid_func(base_model, cfg, log_prefix=f"CFG{cfg}", to_visualize=True, log_folder_name=f"Model")
                            ema_fid_dict = eval_fid_func(first_ema, cfg, log_prefix=f"CFG{cfg}", to_visualize=True, log_folder_name=f"EMA_{best_ema}")
                            if base_fid_dict["fid"] < best_fid_base:
                                best_fid_base = base_fid_dict["fid"]
                                best_cfg_base = cfg
                            if ema_fid_dict["fid"] < best_fid_ema:
                                best_fid_ema = ema_fid_dict["fid"]
                                best_cfg_ema = cfg
                        
                        logger.log_dict({
                            "fid_best/best_fid_base": best_fid_base,
                            "fid_best/best_cfg_base": best_cfg_base,
                            "fid_best/best_fid_ema": best_fid_ema,
                            "fid_best/best_cfg_ema": best_cfg_ema,
                            "fid_best/best_ema": best_ema,
                            "fid_best/best_isc": best_isc,
                        })

                        eval_fid_func(first_ema, best_cfg_ema, log_prefix=f"EMA", samples=50000, to_visualize=True, log_folder_name=f"fid_best", eval_clip=True, eval_prc_recall=True)

                base_model.train()
                logger.log_dict(
                    {
                        "eval_time_per_kimg": (time.time() - old_time)
                        / (eval_gen_per_step * total_batch_size / 1000)
                    }
                )

        if is_main_process and (
            steps % save_per_step == 0 or steps == n_steps - 1
        ):  # and steps != 0:
            old_time = time.time()
            # To prevent CUDA OOM, we get state dicts and move them to CPU before saving.
            # This avoids creating model copies on GPU.
            ckpt_data = {
                "model": {
                    k.replace("_orig_mod.", ""): v.cpu()
                    for k, v in base_model.state_dict().items()
                }
            }
            for decay, ema_model in ema_models.items():
                ckpt_data[f"ema_model_{decay}"] = {
                    k.replace("_orig_mod.", ""): v.cpu()
                    for k, v in ema_model.model.state_dict().items()
                }
            # Save optimizer and memory banks
            try:
                # Create a new state dict on CPU to prevent modifying the live optimizer state and to avoid a temporary copy on GPU.
                original_optimizer_state = optimizer.state_dict()
                cpu_optimizer_state = {
                    "state": {
                        pid: {
                            k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in st.items()
                        }
                        for pid, st in original_optimizer_state["state"].items()
                    },
                    "param_groups": original_optimizer_state["param_groups"],
                }
                ckpt_data["optimizer"] = cpu_optimizer_state
            except Exception as e:
                print("Failed to save optimizer state:", e)
            if n_cond_samples > 0:
                ckpt_data["memory_bank"] = memory_bank.state_dict()
            if n_gen_bank_samples > 0:
                ckpt_data["gen_memory_bank"] = gen_memory_bank.state_dict()
            if all_samples_bank is not None:
                ckpt_data["all_samples_bank"] = all_samples_bank.state_dict()
            save_ckpt(
                job_name,
                steps,
                ckpt_data,
                max_ckpts=persist_latest_ckpts,
                persist_every=persist_ckpts_every,
            )
            logger.log_dict(
                {
                    "save_time_per_kimg": (time.time() - old_time)
                    / (save_per_step * (steps + 1 - start_step_idx) * total_batch_size / 1000)
                }
            )

        # push samples
        if steps < push_warmup_steps:
            push_timer = Timer()
            with record_function("push_warmup") if profile_train else contextlib.nullcontext():
                push_timer.start()
                need_push = warmup_push_size // local_batch_size
                for i in range(need_push):
                    samp = next(dataset_iterator)
                    push_samples(samp[0], samp[1])
                push_dt = push_timer.stop()
            if profile_train:
                logger.log_dict({"prof/push_warmup_s_perkimg": push_dt / (total_batch_size / 1000)})


        # data loading
        data_timer = Timer()
        with record_function("data_loading") if profile_train else contextlib.nullcontext():
            data_timer.start()
            samples = next(dataset_iterator)

            data_dt = data_timer.stop()
        if profile_train:
            logger.log_dict({"prof/data_loading_s_perkimg": data_dt / (total_batch_size / 1000)})

        # prepare inputs
        prep_timer = Timer()
        with record_function("prepare_inputs") if profile_train else contextlib.nullcontext():
            prep_timer.start()
            with torch.cuda.amp.autocast(dtype=dtype) if use_amp else contextlib.nullcontext():
                args, kwargs = samples_to_inputs(samples)
            prep_dt = prep_timer.stop()

        # forward
        fwd_timer = Timer()
        with record_function("forward") if profile_train else contextlib.nullcontext():
            fwd_timer.start()
            with torch.cuda.amp.autocast(dtype=dtype) if use_amp else contextlib.nullcontext():
                loss, info = model(*args, **kwargs)
            fwd_dt = fwd_timer.stop()


        # backward
        bwd_timer = Timer()
        with record_function("backward") if profile_train else contextlib.nullcontext():
            bwd_timer.start()
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), lr_schedule.clip_grad)
            bwd_dt = bwd_timer.stop()

        # Optimizer step
        opt_timer = Timer()
        with record_function("optimizer_step") if profile_train else contextlib.nullcontext():
            opt_timer.start()
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            opt_dt = opt_timer.stop()

        # EMA
        ema_timer = Timer()
        with record_function("ema_update") if profile_train else contextlib.nullcontext():
            ema_timer.start()
            for ema_model in ema_models.values():
                ema_model.update(base_model)
            ema_dt = ema_timer.stop()

        # Logging
        log_timer = Timer()
        with record_function("log_metrics") if profile_train else contextlib.nullcontext():
            log_timer.start()
            logger.log_dict(info)
            logger.log_dict({"grad_norm": grad_norm.mean(), "lr": lr, "kimg": (steps + 1) * total_batch_size / 1000})
            logger.log_dict(
                {
                    "time_per_kimg": (time.time() - start_time)
                    / ((steps + 1 - start_step_idx) * total_batch_size / 1000)
                })
            total_dt = total_timer.stop()
            log_dt = log_timer.stop()

        if profile_train:
            denom = (total_batch_size / 1000)
            logger.log_dict({
                "prof/prepare_s_perkimg":    prep_dt / denom,
                "prof/forward_s_perkimg":    fwd_dt  / denom,
                "prof/backward_s_perkimg":   bwd_dt  / denom,
                "prof/optim_s_perkimg":      opt_dt  / denom,
                "prof/ema_s_perkimg":        ema_dt  / denom,
                "prof/log_s_perkimg":        log_dt  / denom,
                "prof/step_total_s_perkimg": total_dt / denom,
                "prof/step_total_s":         total_dt,
            })
        if steps <= 2:
            no_grad_params = [
                (n, p) for n, p in model.named_parameters() if p.grad is None
            ]
            print("Params without grads:", ", ".join([n for n, p in no_grad_params]))


        if prof is not None:
            prof.step()
    
    if prof is not None:
        prof.__exit__(None, None, None)

    return {"model": base_model, "ema_model": ema_models}


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--job_name", type=str, default=None, help="Job name for wandb."
    )
    return parser


def main(args):
    """
    This function is the main function for training the VAE.
    It initializes the distributed mode and the model.
    It then trains the model.
    Args:
        args: the arguments for the training;
        requires:
            args.dist_url: the url of the distributed training
            args.job_name: the name of the job
    Returns:
        None
    """

    args = deepcopy(args)
    
    if not getattr(args, "both", False):
        config = load_config(args.config)
        config['config_path'] = args.config
        if args.job_name:
            config.wandb.name = args.job_name
    else:
        config = load_config(args.config)
        config = vae_from_combined(config, args.job_name)
        config['config_path'] = args.config
        args.job_name = config.wandb.name

    if (config.train.n_steps - 1) in ckpt_epoch_numbers(args.job_name):
        print("Found latest ckpt; should early exit")
        return

    # Setup distributed training
    if "dist_url" in config:
        args.dist_url = config.dist_url
    init_distributed_mode(args)
    os.environ["LOCAL_RANK"] = str(args.gpu)

    model_dict = build_model_dict(config, VAE)

    # Pre/post processing are configured in build_model_dict based on dataset

    train_vae(
        model_dict.model,
        model_dict.dataset_name,
        model_dict.optimizer,
        **model_dict.train,
        logger=model_dict.logger,
        eval_dataset=model_dict.eval_dataset,
        train_dataset=model_dict.train_dataset,
        job_name=args.job_name,
    )


# %%
if __name__ == "__main__":
    args = get_args_parser().parse_args()

    args.dist_url = "env://"
    main(args)

"""
export ts=$(date +%Y%m%d_%H%M%S)
torchrun --nproc_per_node=8 -m train --config configs/debug.yaml --job_name test_debug_${ts}
"""
# bash run_local.sh --config configs_new/mnist_h100/mnist_256baseline.yaml -n mnist_256baseline
# bash run_local.sh --config configs_new/mnist_h100/mnist_256baseline_attnnew.yaml -n mnist_256baseline_attnnew
# bash run_local.sh --config configs_new/mnist_h100/mnist_256baseline_moco_attnnew.yaml -n mnist_256baseline_moco_attnnew
# bash run_local.sh --config configs_new/mnist_h100/mnist_256baseline_moco_small_steps.yaml -n mnist_256baseline_moco_small_steps
# bash run_local.sh --config configs_new/mnist_h100/mnist_256baseline_moco_step1.yaml -n mnist_256baseline_moco_step1
# bash run_local.sh --config configs_new/mnist_h100/mnist_256baseline_moco_small_step_moco.yaml -n mnist_256baseline_moco_small_step_moco
# bash run_local.sh --config configs_new/mnist_h100/continue/mnist_continue_no_moco.yaml -n mnist_continue_no_moco
# bash run_local.sh --config configs_new/mnist_h100/continue/mnist_continue_online.yaml -n mnist_continue_online
# bash run_local.sh --config configs_new/mnist_h100/continue/mnist_continue_gen_online.yaml -n mnist_continue_gen_online
# bash run_local.sh --config configs_new/mnist_h100/continue/mnist_continue_online_bank.yaml -n mnist_continue_online_bank
# torchrun --nproc_per_node=8 -m train  --config configs/debug.yaml --job_name test_debug_4
# bash run_local.sh --config configs_new/mnist_h100/continue/mnist_continue_online_bank_L.yaml -n mnist_continue_online_bank_L
# bash run_local.sh --config configs_new/mnist_h100/continue/mnist_continue_online_bank_moco.yaml -n mnist_continue_online_bank_moco
# bash run_local.sh --config configs_new/mnist_h100/continue/mnist_continue_online_bank_moco_L.yaml -n mnist_continue_online_bank_moco_L
# bash run_local.sh --config configs_new/mnist_h100/continue/mnist_pretrain_moco_L.yaml -n mnist_pretrain_moco_L
# bash run_local.sh --config configs_new/mnist_h100/mnist_256baseline_attnnew_morebsz.yaml -n mnist_256baseline_attnnew_morebsz
# bash run_local.sh --config configs_new/mnist_h100/loss/mnist_256baseline_R_0.1.yaml -n mnist_256baseline_R_0.1
# bash run_local.sh --config configs_new/mnist_h100/loss/mnist_256baseline_R_multi.yaml -n mnist_256baseline_R_multi
# bash run_local.sh --config configs_new/mnist_h100/loss/mnist_256baseline_R_multi.yaml -n mnist_256baseline_R_multi
# bash run_local.sh --config configs_new/mnist_h100/loss/mnist_256baseline_R_0.1_notrans.yaml -n mnist_256baseline_R_0.1_notrans
'''
export ts=$(date +%Y%m%d_%H%M%S)
torchrun --nproc_per_node=8 -m train  --config configs/debug.yaml --job_name test_debug_${ts}
micromamba activate new
srun --gpus-per-node=8 \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --mem=1760G \
     -t 3-0 \
     --qos=h100_core_shared \
     --account=flows \
     --pty bash -i

TODO:
    see if compile speed faster
    change logging to global 

'''