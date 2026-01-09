# %%
import torch
import torch.nn as nn
import wandb
import contextlib

from utils.misc import EasyDict, set_seed, easydict_to_dict
from utils.distributed_utils import init_distributed_mode, get_rank, get_world_size, get_local_rank
from utils.ema import EMA
from utils.ckpt_utils import load_last_ckpt, load_ckpt_epoch, ckpt_epoch_numbers
from dataset import postprocess
from tqdm import tqdm
import random
from copy import deepcopy
import time
import numpy as np  
import datetime

from model.sit import FMLoss
from model.mlp import CondEmbed
from utils.misc import EasyDict
from energy_loss import build_feature_modules
from einops import repeat, rearrange

from utils.augment import getEDMAugment
from utils.misc import LinearWarmupCosineDecayLR

from utils.ckpt_utils import save_ckpt
from utils.profile import print_module_summary
from utils.misc import InfiniteSampler
import datetime

import os
from torch.nn.parallel import DistributedDataParallel as DDP
from config import load_config, build_model_dict, null_prepost, compose_pre_post
from model.resnet import LatentResNet

torch._dynamo.config.verbose = True  # Enables verbose compile logs
torch.set_float32_matmul_precision('high')  # Enable TF32 tensor cores

from dataset import get_dataset
from utils.misc import add_weight_decay


def eval_acc()
# %%
def train_classifier(
    config,
    model,
    optimizer,
    n_steps,
    total_batch_size=128,
    ema_dict=dict(halflife_kimg=500),
    eval_per_step=1000,
    logger=None,
    forward_dict=dict(),
    eval_dataset=None,
    train_dataset=None,
    lr_schedule=EasyDict(
        lr=2e-4,
        warmup_kimg=10000,
        total_kimg=200000,
        clip_grad=10.0,
    ),
    load_dict=EasyDict(
        run_id="",
        epoch="latest",
        continue_training=False,
    ),
    job_name="", # the unique pointer for saving & loading ckpts. 
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
        local_rank = get_rank()
        world_size = get_world_size()
    else:
        print("Distributed training not initialized")
        local_rank = 0
        world_size = 1

    set_seed(local_rank)

    base_model = model
    if 'halflife_kimg' in ema_dict:
        decay_schedule = lambda t: 0.5 ** (total_batch_size / (ema_dict['halflife_kimg'] * 1000))
    else:
        edm_schedule = EDM2Schedule(ema_dict['edm_scale'])
        decay_schedule = lambda t: edm_schedule.decay_scale(t)

    ema_model = EMA(base_model, decay_schedule(1))
    is_main_process = local_rank == 0

    pbar = tqdm(range(n_steps)) if is_main_process else range(n_steps)

    load_step = 0
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
        step += 1 # avoid evaluation; remove if want eval.
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
        model = torch.compile(model)

    if distributed and world_size > 1:
        model = DDP(model, device_ids=[get_local_rank()]) # don't use get_local_rank() here! won't work for slurm. 
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    torch.cuda.synchronize()
    is_main_process = local_rank == 0
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    model_without_ddp.train()

    autocast_context = torch.cuda.amp.autocast(dtype=dtype) if use_amp else contextlib.nullcontext()

    dataset_sampler = InfiniteSampler(dataset=train_dataset, rank=local_rank, num_replicas=world_size, seed=0)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    sampler=dataset_sampler, 
                                                    batch_size=total_batch_size // max(1, world_size), 
                                                    num_workers=16, 
                                                    pin_memory=True))


    start_time = time.time()
    for steps in pbar:
        lr = LinearWarmupCosineDecayLR((steps + 1 - load_step) * total_batch_size / 1000, lr_schedule)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        logger.set_step(steps)
            
        # FID gen
        if eval_dataset is not None and (steps % eval_per_step == 0 or steps == n_steps - 1): # and steps != 0:
            with torch.inference_mode():
                old_time = time.time()
                base_model.eval()
                ema_model.model.eval()

                gen_ema = lambda x: postprocess_fn(ema_model.model.sample(x[0].shape[0], 50, x[1].to(device), sampler="heun"), [x[1]])
                gen_model = lambda x: postprocess_fn(base_model.sample(x[0].shape[0], 50, x[1].to(device), sampler="heun"), [x[1]])
                # gen_slow = lambda x: postprocess_fn(ema_slow.model.sample(x[0].shape[0], 50, x[1].to(device), sampler="heun"), [x[1]])
                
                if steps != 0:
                    eval_fid(config, gen_ema, eval_dataset, logger, log_prefix="gen_ema", dataset=dataset_name)
                    eval_fid(config, gen_model, eval_dataset, logger, log_prefix="gen_model", dataset=dataset_name)
                    # eval_fid(config, gen_slow, eval_dataset, logger, log_prefix="gen_slow", dataset=dataset_name)
                    
                    if steps % (10 * eval_gen_per_step) == 0:
                        eval_fid(config, gen_ema, eval_dataset, logger, log_prefix="gen_ema", dataset=dataset_name, total_samples=50000, gpu_batch_size=64)

                base_model.train()
                logger.log_dict({"eval_time_per_kimg": (time.time() - old_time) / (eval_gen_per_step * total_batch_size / 1000)})

        
        if is_main_process and (steps % save_per_step == 0 or steps == n_steps - 1) and steps != 0:
            old_time = time.time()
            model_2 = deepcopy(model_copy)
            copy_params(ema_model.model, model_2)
            copy_params(base_model, model_copy)
            save_ckpt(
                config,
                wandb.run.id,
                steps,
                {
                    "model": model_copy,
                    "ema_model": model_2,
                },
            )
            logger.log_dict({"save_time_per_kimg": (time.time() - old_time) / (save_per_step * steps * total_batch_size / 1000)})

        # Training step
        samples = next(dataset_iterator)
        with autocast_context:
            args, kwargs = samples_to_inputs(samples)
            loss, info = model(*args, **kwargs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        max_grad_norm = lr_schedule.clip_grad
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if steps <= 2:
            no_grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is None]
            print("Params without grads:", ', '.join([n for n, p in no_grad_params]))
        
        logger.log_dict({"grad_norm": grad_norm.mean().item()})
        logger.log_dict({"lr": lr})
        logger.log_dict({"kimg": (steps + 1) * total_batch_size / 1000})
        logger.log_dict({"time_per_kimg": (time.time() - start_time) / ((steps + 1) * total_batch_size / 1000)})

        logger.log_dict(info)
        logger.log_dict({"loss": loss.item()})
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        decay = decay_schedule(steps)
        ema_model.update(base_model, decay)
        ema_slow.update(base_model, decay ** 0.3)
        
    return {"model": base_model, "ema_model": ema_model.model}

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

    config = load_config(args.config)

    # Setup job name
    if args.job_name:
        config.wandb.name = args.job_name
    elif config.wandb.name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.wandb.name = f"clean_sanity_check_{timestamp}"

    # Setup distributed training
    if 'dist_url' in config:
        args.dist_url = config.dist_url
    init_distributed_mode(args)
    os.environ["LOCAL_RANK"] = str(args.gpu)

    model_dict = build_model_dict(config, FMLoss)
    
    # Setup pre/post processing
    null_prep = null_prepost()
    pre, post = compose_pre_post([null_prep])
    model_dict.train.preprocess_fn = pre
    model_dict.train.postprocess_fn = post

    train_sit(
        config,
        model_dict.model,
        model_dict.dataset_name,
        model_dict.optimizer,
        **model_dict.train,
        logger=model_dict.logger,
        eval_dataset=model_dict.eval_dataset,
        train_dataset=model_dict.train_dataset,
    )

# %%
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)# python -m train_sit --config configs/sit_configs/sit_B_1_192.yaml --job_name sit_B_1_192
