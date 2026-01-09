# %%
import torch
import torch.nn as nn
import contextlib

from utils.misc import EasyDict, set_seed, easydict_to_dict
from utils.distributed_utils import init_distributed_mode, get_rank, get_world_size, get_local_rank, aggregate_info_across_gpus, sum_info_across_gpus
from utils.ema import EMA
from utils.ckpt_utils import load_last_ckpt, load_ckpt_epoch, ckpt_epoch_numbers
from dataset import postprocess
from tqdm import tqdm
import random
from copy import deepcopy
import time
from train_both import mae_from_combined
import numpy as np  
import datetime

from model.vit_ae import Encoder, Decoder
from model.mlp import CondEmbed
from utils.misc import EasyDict
from einops import repeat, rearrange

from utils.augment import getEDMAugment
from utils.misc import LinearWarmupCosineDecayLR

from utils.fid import eval_fid
from utils.ckpt_utils import save_ckpt
from utils.profile import print_module_summary
from utils.misc import InfiniteSampler
from utils.ema import EDM2Schedule
import datetime
from utils.augment import getEDMAugment

import os
from torch.nn.parallel import DistributedDataParallel as DDP
from config import load_config, build_model_dict, null_prepost, compose_pre_post
from memory_bank import MemoryBank
from utils.ckpt_utils import load_last_ckpt, ckpt_epoch_numbers, load_ckpt_epoch
from torch.utils.data import DataLoader
from utils.misc import TemporalSeed
torch._dynamo.config.verbose = True  # Enables verbose compile logs
torch.set_float32_matmul_precision('high')  # Enable TF32 tensor cores

from model.mae_resnet import MAEResNet
from model.mae_resnet_gn import MAEResNetGN

from model.mae_convnext import MAEConvNeXt
from model.mae_vit import MAEViT
# %%

def eval_loss(
    model,
    cond_dataset,
    logger,
    total_samples=5000,
    gpu_batch_size=768,
    preprocess_fn=lambda x, label_list: x,
    postprocess_fn=postprocess,
    log_prefix="",
    forward_kwargs=None,
):
    """
    Generate and evaluate FID for a given generator and dataset.
    Args:
        generator: a function, takes in a batch in dataset, returns loss & info
        cond_dataset: a dataset, each index returns a batch of cond.
    """

    distributed = torch.distributed.is_initialized()
    if distributed:
        torch.distributed.barrier()
    start_time = time.time()
    rank = get_rank() if distributed else 0
    world_size = get_world_size() if distributed else 1
    context = TemporalSeed(rank)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            cond_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        loader = DataLoader(cond_dataset, batch_size=gpu_batch_size, sampler=sampler)
    else:
        loader = DataLoader(cond_dataset, batch_size=gpu_batch_size, shuffle=True)

    # generate samples
    current_samples = 0
    samples_per_gpu = total_samples // world_size

    info_dict = dict()
    with torch.inference_mode():
        for batch in loader:
            sample_size = min(gpu_batch_size, samples_per_gpu - current_samples)
            if sample_size <= 0:
                break

            if sample_size < gpu_batch_size:
                if isinstance(batch, tuple) or isinstance(batch, list):
                    batch = [x[:sample_size] for x in batch]
                else:
                    assert isinstance(batch, torch.Tensor)
                    batch = batch[:sample_size]
            
            fk = forward_kwargs or {}
            loss, info = model(preprocess_fn(batch[0], batch[1]).cuda(), batch[1].cuda(), **fk)
            info = {k: v * sample_size for k, v in info.items()}
            info['n_elts'] = sample_size
            for k in info:
                if k not in info_dict:
                    info_dict[k] = info[k]
                else:
                    info_dict[k] += info[k]
            current_samples += sample_size
    # Derive device from model parameters to avoid attribute errors
    model_device = next(model.parameters()).device
    info_sum = sum_info_across_gpus(info_dict, model_device)
    n_elts = info_sum['n_elts']
    info_sum = {f'{log_prefix}{k}': v / n_elts for k, v in info_sum.items() if k != 'n_elts'}
    logger.log_dict(info_sum)

    if distributed:
        torch.distributed.barrier()
    context.resume()
    if distributed:
        torch.distributed.barrier()



def train_vae(
    config,
    model,
    dataset_name,
    optimizer,
    n_steps,
    total_batch_size=128,
    ema_dict=dict(halflife_kimg=500),
    eval_per_step=1000,
    save_per_step=2000,
    logger=None,
    eval_dataset=None,
    train_dataset=None,
    job_name="", # the unique pointer for saving & loading ckpts. 
    # Added defaults for optional training controls
    use_bf16=False,
    compile_model=False,
    eval_gen_batch_size=64,
    lr_schedule=EasyDict(lr=2e-4, warmup_kimg=10000, total_kimg=200000, clip_grad=1.0),
    load_dict=EasyDict(run_id="", epoch="latest", continue_training=False),
    edm_augment=False,
    lambda_cls=0.5,
    mask_ratio_min=0.75,
    mask_ratio_max=0.95,
    preprocess_fn=lambda x, label_list: x,
    postprocess_fn=postprocess,
    extra_train_kwargs=dict(),
    extra_eval_kwargs=dict(),
    finetune_cls=0.2, # during finetune: will use this
    finetune_last_steps=10000, # last steps will be finetuning, with the finetune_cls
    finetune_save_per_step=2000, # save the finetuned model every finetune_save_per_step steps
    eval_only=False, # only load ckpt from load_id, and do the evals
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    use_amp = use_bf16 and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    if use_amp:
        print(f"Using AMP training with {dtype} precision")
    
    distributed = False
    if torch.distributed.is_initialized():
        print("Distributed training initialized")
        distributed = True
        rank = get_rank()
        world_size = get_world_size()
    else:
        print("Distributed training not initialized")
        rank = 0
        world_size = 1

    set_seed(rank)

    base_model = model
    if 'halflife_kimg' in ema_dict:
        decay_schedule = lambda t: 0.5 ** (total_batch_size / (ema_dict['halflife_kimg'] * 1000))
    else:
        edm_schedule = EDM2Schedule(ema_dict['edm_scale'])
        decay_schedule = lambda t: edm_schedule.decay_scale(t)

    ema_model = EMA(base_model, decay_schedule(1))
    is_main_process = rank == 0
    pbar = tqdm(range(n_steps)) if is_main_process else range(n_steps)
    load_step = 0
    step = 0
    def load(run_id, epoch, continue_training):
        nonlocal step, pbar, load_step
        if run_id == "":
            return
        all_epochs = ckpt_epoch_numbers(run_id)
        if len(all_epochs) == 0:
            print(f"No checkpoint found for job {run_id}")
            return
        print("Loading from ckpt: ", run_id, epoch)
        if epoch == "latest":
            loaded = load_last_ckpt(run_id) # run_id : should be the job name. 
            step = int(all_epochs[-1])
        else:
            loaded = load_ckpt_epoch(run_id, epoch)
            step = int(epoch)
        print("Loading from ckpt: ", step)
        info_x = base_model.load_state_dict(loaded["model"] if isinstance(loaded["model"], dict) else loaded["model"].state_dict(), strict=False)
        print("Missing keys:", info_x.missing_keys, "Unexpected keys:", info_x.unexpected_keys)
        info_y = ema_model.model.load_state_dict(loaded["ema_model"] if isinstance(loaded["ema_model"], 
                                                                                   dict) else loaded["ema_model"].state_dict(), strict=False)
        print("Missing keys:", info_y.missing_keys, "Unexpected keys:", info_y.unexpected_keys)
        # step += 1 # avoid evaluation; remove if want eval.
        if continue_training:
            pbar = tqdm(range(step, n_steps))
            load_step = step
        return
    load(load_dict.run_id, load_dict.epoch, load_dict.continue_training)
    load(job_name, "latest", True) # load the latest checkpoint of the job. 
        
    model_copy = deepcopy(base_model)

    def copy_params(src_model, dst_model):
        model_state_dict = src_model.state_dict()
        updated_list = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}
        info = dst_model.load_state_dict(updated_list, strict=False)
        print("Missing keys:", info.missing_keys, "Unexpected keys:", info.unexpected_keys)

    if compile_model:
        model.compile_model()

    if distributed and world_size > 1:
        model = DDP(model, device_ids=[get_local_rank()]) # don't use get_local_rank() here! won't work for slurm. 
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    is_main_process = rank == 0

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    model_without_ddp.train()

    autocast_context = torch.cuda.amp.autocast(dtype=dtype) if use_amp else contextlib.nullcontext()

    dataset_sampler = InfiniteSampler(dataset=train_dataset, rank=rank, num_replicas=world_size, seed=0)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    sampler=dataset_sampler, 
                                                    batch_size=total_batch_size // max(1, world_size), 
                                                    num_workers=8, 
                                                    pin_memory=True, 
                                                    prefetch_factor=8))

    if edm_augment:
        augmenter = getEDMAugment()
        def aug(x):
            return augmenter(x)[0]
    else:
        def aug(x):
            return x
    
    extra_train_kwargs = extra_train_kwargs or {}
    extra_eval_kwargs = extra_eval_kwargs or {}
    start_time = time.time()

    ckpt_id = {'cls': 0, 'nocls': 0}

    def samples_to_inputs(samples, lambda_cls):
        imgs, labels = samples
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = aug(imgs)
        imgs = preprocess_fn(imgs, labels)
        imgs = imgs.to(device)
        kwargs = dict(lambda_cls=float(lambda_cls), mask_ratio_min=float(mask_ratio_min), mask_ratio_max=float(mask_ratio_max))
        kwargs.update(extra_train_kwargs)
        return (imgs, labels), kwargs
    for steps in pbar:
        print(f"Steps: {steps}")


        do_profile = (not eval_only) and steps % 1000 == 1
        if do_profile:
            with torch.no_grad():
                if is_main_process:
                    old_time = time.time()
                    imgs, labels = next(dataset_iterator)
                    args, kwargs = samples_to_inputs((imgs[:32], labels[:32]), lambda_cls=lambda_cls    )
                    copy_params(base_model, model_copy)
                    table = print_module_summary(model_copy, inputs=args, kwargs=kwargs, max_nesting=4)
                    logger.log_table({"model_summary": table})
                    logger.log_dict({"profile_time_per_kimg": (time.time() - old_time) / (1000 * total_batch_size / 1000)})

        current_step = (steps + 1 - load_step)

        finetune_start = n_steps - finetune_last_steps
        if steps > finetune_start:
            current_step = steps - finetune_start
        lr = LinearWarmupCosineDecayLR(current_step * total_batch_size / 1000, lr_schedule)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        logger.set_step(steps)

        do_eval = eval_dataset is not None and (eval_per_step > 0) and (steps % eval_per_step == 0 or steps == n_steps - 1)

        if eval_only:
            has_ckpts = ckpt_epoch_numbers(load_dict.run_id)
            do_eval = False
            if steps in has_ckpts:
                print(f"Loading ckpt from {load_dict.run_id} at step {steps}")
                do_eval = True
                load(load_dict.run_id, steps, False)
                if distributed:
                    torch.distributed.barrier()
        
        # FID gen
        if do_eval: # and steps != 0:
            with torch.inference_mode():
                old_time = time.time()
                base_model.eval()
                ema_model.model.eval()
                fk = dict(lambda_cls=float(lambda_cls), mask_ratio_min=float(mask_ratio_min), mask_ratio_max=float(mask_ratio_max))
                fk.update(extra_eval_kwargs)
                eval_loss(base_model, train_dataset, logger, total_samples=5000, gpu_batch_size=eval_gen_batch_size, log_prefix="eval_train/", forward_kwargs=fk, preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                
                eval_loss(base_model, eval_dataset, logger, total_samples=5000, gpu_batch_size=eval_gen_batch_size, log_prefix="eval/", forward_kwargs=fk, preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                eval_loss(ema_model.model, eval_dataset, logger, total_samples=5000, gpu_batch_size=eval_gen_batch_size, log_prefix="eval_ema/", forward_kwargs=fk, preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                no_mask = dict(lambda_cls=float(lambda_cls), mask_ratio_min=0.0, mask_ratio_max=0.0)
                no_mask.update(extra_eval_kwargs)
                eval_loss(base_model, eval_dataset, logger, total_samples=5000, gpu_batch_size=eval_gen_batch_size, log_prefix="eval_no_mask/", forward_kwargs=no_mask, preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                eval_loss(ema_model.model, eval_dataset, logger, total_samples=5000, gpu_batch_size=eval_gen_batch_size, log_prefix="eval_ema_no_mask/", forward_kwargs=no_mask, preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                
                base_model.train()
                logger.log_dict({"eval_time_per_kimg": (time.time() - old_time) / (eval_per_step * total_batch_size / 1000)})

        should_save = False
        if steps < finetune_start:
            should_save = (steps % save_per_step == 0 or steps == n_steps - 1)
            if should_save:
                ckpt_id['nocls'] = steps
        else:
            should_save = (steps % finetune_save_per_step == 0 or steps == n_steps - 1)
            if should_save:
                ckpt_id['cls'] = steps
        if eval_only:
            should_save = False
        if is_main_process and should_save: # and steps != 0:
            old_time = time.time()
            model_2 = deepcopy(model_copy)
            copy_params(ema_model.model, model_2)
            copy_params(base_model, model_copy)
            save_ckpt(
                job_name,
                steps,
                {
                    "model": model_copy,
                    "ema_model": model_2,
                },
            )
            logger.log_dict({"save_time_per_kimg": (time.time() - old_time) / (save_per_step * (steps + 1) * total_batch_size / 1000)})

        # Training step
        do_train = not eval_only
        if do_train:
            samples = next(dataset_iterator)
            with autocast_context:
                lambda_now = lambda_cls if steps < finetune_start else finetune_cls
                args, kwargs = samples_to_inputs(samples, lambda_cls=lambda_now)
                # print(args[0].shape, args[1].shape)
                loss, info = model(*args, **kwargs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            max_grad_norm = lr_schedule.clip_grad
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            if steps <= 2:
                no_grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is None]
                print("Params without grads:", ', '.join([n for n, p in no_grad_params]))
            
            logger.log_dict({"grad_norm": float(grad_norm)})
            logger.log_dict({"lr": lr})
            logger.log_dict({"kimg": (steps + 1) * total_batch_size / 1000})
            logger.log_dict({"time_per_kimg": (time.time() - start_time) / ((steps + 1) * total_batch_size / 1000)})
            logger.log_dict(info)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            decay = decay_schedule(steps)
            ema_model.update(base_model, decay)
    # barrier after training step
    if distributed:
        torch.distributed.barrier()
    return ckpt_id

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the config file.')
    parser.add_argument('--job_name', type=str, default=None, help='Job name for wandb.')
    return parser

def main(args):
    ''' 
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
    '''
    args = deepcopy(args)
    

    if not getattr(args, "both", False):
        config = load_config(args.config)
        config['config_path'] = args.config
        if args.job_name:
            config.wandb.name = args.job_name
    else:
        config = load_config(args.config)
        config = mae_from_combined(config, args.job_name)
        config['config_path'] = args.config
        args.job_name = config.wandb.name
    
    if (config.train.n_steps - 1) in ckpt_epoch_numbers(args.job_name):
        print("Found latest ckpt; should early exit")
        return
    
    # check if ckpt n_steps - 1 exists for job_name

    # Setup distributed training
    if 'dist_url' in config:
        args.dist_url = config.dist_url
    init_distributed_mode(args)
    os.environ["LOCAL_RANK"] = str(args.gpu)
    model_type = config.get("model_type", "mae_resnet")
    if model_type == "mae_resnet":
        model_class = MAEResNet
    elif model_type == "mae_resnet_gn":
        model_class = MAEResNetGN
    elif model_type == "mae_convnext":
        model_class = MAEConvNeXt
    elif model_type == "mae_vit":
        model_class = MAEViT
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model_dict = build_model_dict(config, model_class)
    
    train_vae(
        config,
        model_dict.model,
        model_dict.dataset_name,
        model_dict.optimizer,
        **model_dict.train,
        logger=model_dict.logger,
        eval_dataset=model_dict.eval_dataset,
        train_dataset=model_dict.train_dataset,
        job_name=args.job_name
    )

# %%
if __name__ == "__main__":
    args = get_args_parser().parse_args()

    args.dist_url = "env://"
    main(args)

'''
export ts=$(date +%Y%m%d_%H%M%S)
torchrun --nproc_per_node=8 -m train_mae --config config_mae/resnet34_debug.yaml --job_name resnet34_debug_${ts}
'''
# bash run_v100.sh config_cls/dropout --cls
