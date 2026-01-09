import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

from dataset.toys import build_dataset, eval
from utils.misc import EasyDict, set_seed
from utils.distributed_utils import init_distributed_mode, get_rank, get_world_size, get_local_rank, aggregate_info_across_gpus
from utils.logging_utils import WandbLogger
from config import load_config
import argparse
from copy import deepcopy

class Model(nn.Module):
    def __init__(self, noise_dim=16, output_dim=3, acti="gelu"):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        if acti == "gelu":
            acti_fn = nn.GELU
        elif acti == "relu":
            acti_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation function: {acti}")
        self.net = nn.Sequential(
            nn.Linear(self.noise_dim, 128),
            acti_fn(),
            nn.Linear(128, 128),
            acti_fn(),
            nn.Linear(128, 128),
            acti_fn(),
            nn.Linear(128, 128),
            acti_fn(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def generate(self, bsz):
        device = next(self.parameters()).device
        return self.forward(torch.randn(bsz, self.noise_dim, device=device))


def build_model(gen_config):
    if gen_config.type == "mlp":
        return Model(**gen_config.kwargs)
    else:
        raise ValueError(f"Unknown generator type: {gen_config.type}")


def cdist(x, y, eps=1e-8):
    """
    Args:
        x: [B, C1, D]
        y: [B, C2, D]
    Returns: [B, C1, C2]

    Same effect as torch.cdist, but faster.
    """
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    return (xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot).clamp(min=eps).sqrt()


from energy_loss import attn_contra_loss, attn_loss_new

def train_toy(
    model,
    dataset_cfg,
    n_steps,
    neg_cfg = EasyDict(
        dataset=None, 
        n_neg_samples=0,
        cfg_scale=1.0,
    ),
    logger=None,
    lr_schedule=EasyDict(
        lr=3e-3,
        clip_grad=1.0,
    ),
    loss_config=EasyDict(
        sample_norm=True,
        scale_dist=False,
        no_R_norm=True,
        no_global_norm=False,
        new_R_norm=False,
        scale_dist_normed=True,
        R_list=[0.2],
    ),
    data_bsz=256,
    gen_per_data=256,
    pos_per_data=1,
):
    # Setup distributed training=
    
    if torch.distributed.is_initialized():
        distributed = True
        world_size = get_world_size()
        rank = get_rank()
    else:
        distributed = False
        world_size = 1 
        rank = 0
    device = torch.device(f"cuda:{get_local_rank()}")
    set_seed(rank)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_schedule.lr)
    if distributed:
        model = DDP(model, device_ids=[get_local_rank()])
    model_without_ddp = model.module

    # Build dataset
    dataset = build_dataset(**dataset_cfg)
    if neg_cfg.dataset is not None:
        neg_dataset = build_dataset(**neg_cfg.dataset)

    # Training loop
    pbar = tqdm(range(n_steps)) if get_rank() == 0 else range(n_steps)
    for step in pbar:
        logger.set_step(step)
        optimizer.zero_grad()
        bsz_per_rank = data_bsz // world_size
        data = dataset.sample(bsz_per_rank * pos_per_data).to(device)
        data = data.reshape(bsz_per_rank, pos_per_data, -1)

        noise = torch.randn(gen_per_data * bsz_per_rank, model_without_ddp.noise_dim, device=device)
        gen = model(noise)
        gen = gen.reshape(bsz_per_rank, gen_per_data, -1)
        w = torch.ones_like(gen[:, :, 0])
        
        if neg_cfg.dataset is not None:
            neg_samples = neg_dataset.sample(bsz_per_rank * neg_cfg.n_neg_samples).to(device)
            neg_samples = neg_samples.reshape(bsz_per_rank, neg_cfg.n_neg_samples, -1)
            neg_w = (neg_cfg.cfg_scale - 1) * (gen_per_data - 1) / neg_samples.shape[1]
            gen = torch.cat([gen, neg_samples], dim=1)
            w = torch.cat([w, torch.ones_like(neg_samples[:, :, 0]) * neg_w], dim=1)

        cur_config = deepcopy(loss_config)
        loss_type = cur_config.pop("type", "attn")
        if loss_type == "attn":
            loss, info = attn_contra_loss(
                target=data, recon=gen, weight_r=w, return_info=True, **cur_config
            )
        elif loss_type == "attn_new":
            loss, info = attn_loss_new(
                gen=gen, fixed_pos=data, **cur_config
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        loss = loss.mean()

        loss.backward()

        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), 2)
                for p in model.parameters()
                if p.grad is not None
            ]), 2
        )
        aggr_info = aggregate_info_across_gpus(info, device)
        if get_rank() == 0:
            logger.log_dict({"grad_norm": total_norm.item()})

        torch.nn.utils.clip_grad_norm_(model.parameters(), lr_schedule.clip_grad)
        optimizer.step()

        if get_rank() == 0:
            logger.log_dict({"loss": loss.item(), **aggr_info})

            if step % 500 == 0:
                eval_res = eval(dataset, lambda bsz: model_without_ddp.generate(bsz)[:, :2])
                logger.log_dict({
                    "eval/nn_data": eval_res["nn_data"],
                    "eval/precision": eval_res["precision"],
                    "eval/recall": eval_res["recall"],
                })
                logger.log_image("eval/fig", eval_res["fig"])
                logger.log_image("eval/hist_fig", eval_res["hist_fig"])
                logger.log_image("eval/quadrant_fig", eval_res["quadrant_fig"])


def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the config file.')
    parser.add_argument('--job_name', type=str, default=None, help='Job name for wandb.')
    return parser


def main(args):
    ''' 
    This function is the main function for training the toy model.
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
        config.wandb.name = f"toy_experiment_{timestamp}"

    # Setup distributed training
    if 'dist_url' in config:
        args.dist_url = config.dist_url
    init_distributed_mode(args)
    model = build_model(config.gen)
    logger = WandbLogger()
    logger.setup_wandb(config)

    train_toy(
        model,
        config.dataset,
        logger=logger,
        **config.train,
    )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.dist_url = "env://"
    main(args)

# torchrun --nproc_per_node=2 --master_port=12345 -m toy --config toy_configs/new_loss/edm_toy.yaml --job_name edm_toy_new
# torchrun --nproc_per_node=2 --master_port=12345 -m toy --config toy_configs/cfg/cfg1.5_gen8_neg8.yaml --job_name cfg1.5_gen8_neg8
# torchrun --nproc_per_node=2 --master_port=12346 -m toy --config toy_configs/cfg/cfg1.5_gen8_neg16.yaml --job_name cfg1.5_gen8_neg16
# torchrun --nproc_per_node=2 --master_port=12347 -m toy --config toy_configs/cfg/cfg1.5_gen8_neg32.yaml --job_name cfg1.5_gen8_neg32
# torchrun --nproc_per_node=2 --master_port=12348 -m toy --config toy_configs/cfg/cfg1.5_gen8_neg32_nosoft.yaml --job_name cfg1.5_gen8_neg32_nosoft
# torchrun --nproc_per_node=2 --master_port=12349 -m toy --config toy_configs/cfg/cfg1.5_gen8_neg32_ood.yaml --job_name cfg1.5_gen8_neg32_ood