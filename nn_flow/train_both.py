
from copy import deepcopy

import torch

from utils.misc import EasyDict
def mae_from_combined(combined_config, job_name):
    mae_config = EasyDict(deepcopy(combined_config.mae))
    mae_config.wandb = deepcopy(combined_config.wandb)
    mae_job_name = job_name + "_mae"
    mae_config.wandb.name = mae_job_name
    return mae_config

def mae_epoch_from_config(mae_config):
    steps = mae_config.train.n_steps
    last_cls = 0
    last_non_cls = 0
    finetune_steps = mae_config.train.finetune_last_steps
    finetune_start = steps - finetune_steps
    save_every_steps = mae_config.train.save_per_step
    save_after_finetune = mae_config.train.finetune_save_per_step

    for i in range(steps):
        to_save = False
        use_cls = False
        if i >= finetune_start:
            use_cls = True
            if i % save_after_finetune == 0:
                to_save = True
        else:
            if i % save_every_steps == 0:
                to_save = True
        if i == steps - 1:
            to_save = True
        if to_save:
            if use_cls:
                last_cls = i
            else:
                last_non_cls = i
    return last_cls, last_non_cls

def vae_from_combined(combined_config, job_name):
    vae_config = EasyDict(deepcopy(combined_config.vae))
    vae_config.wandb = deepcopy(combined_config.wandb)
    vae_job_name = job_name + "_vae"
    vae_config.wandb.name = vae_job_name

    mae_config = mae_from_combined(combined_config, job_name)

    if "clip_dict" not in vae_config.model:
        vae_config.model.clip_dict = EasyDict()
    vae_config.model.clip_dict.update(deepcopy(mae_config.model))
    vae_config.model.clip_dict.update({'model_type': mae_config.model_type})
    mae_epoch = mae_epoch_from_config(mae_config)
    if combined_config.get("use_cls", False):
        mae_epoch = mae_epoch[0]
    else:
        mae_epoch = mae_epoch[1]
    print("Guessing MAE epoch:", mae_epoch)
    vae_config.model.clip_dict.load_dict = EasyDict(run_id = mae_config.wandb.name, epoch = str(mae_epoch), load_entry = "ema_model")
    return vae_config
