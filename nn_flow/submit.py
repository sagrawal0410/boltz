# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import submitit

import toy as toy_trainer
import train as trainer
import train_sit as sit_trainer
import train_cls as cls_trainer
import train_mae as mae_trainer
import train_both as both_trainer
import tempfile
from copy import deepcopy
import shutil
import atexit



def parse_args():
    trainer_parser = trainer.get_args_parser()
    # toy_parser = toy_trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser], add_help=False)
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=10080, type=int, help="Duration of the job in minutes")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--partition", default="learn", type=str, help="Partition where to submit")
    parser.add_argument("--qos", default="flows_high", type=str, help="SLURM QoS")
    parser.add_argument("--use_volta32", action='store_true', help="Use volta32gb")
    parser.add_argument("--account", default="flows", type=str, help="SLURM account")
    parser.add_argument("--requeue", action='store_true', help="SLURM requeue option")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    parser.add_argument('--toy', action='store_true', help="Use toy trainer")
    parser.add_argument('--sit', action='store_true', help="Use sit trainer")        

    parser.add_argument('--both', action='store_true', help="Use both trainer")        
    parser.add_argument('--dist_timeout', default=3000, type=int, help='Timeout for distributed process group initialization in seconds')
    parser.add_argument("--num_retries", default=2, type=int, help="Number of times to resubmit the job on failure")

    parser.add_argument('--cls', action='store_true', help="Use cls trainer")
    parser.add_argument('--mae', action='store_true', help="Use mae trainer")
    return parser.parse_args()


def get_shared_folder() -> Path:
    return Path(".").resolve()


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

import re
def extract_nodes_from_filenames(folder):
    excluded_nodes = set()
    files = os.listdir(folder)
    for fname in files:
        parts = fname.split(',')
        for part in parts:
            node_match = re.search(r'(learnfair\d{4,})', part)
            if node_match:
                excluded_nodes.add(node_match.group(1))
    return sorted(excluded_nodes)

def bad_nodes_from_all_runs(base_dir):
    nodes_set = set()
    for run in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, run)
        if os.path.isdir(run_path):
            nodes = extract_nodes_from_filenames(run_path)
            for node in nodes:
                nodes_set.add(node)
    return nodes_set

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        # import main_pretrain as trainer
        import train as trainer

        self._setup_gpu_args()
        try:
            if self.args.toy:
                toy_trainer.main(self.args)
            elif self.args.sit:
                sit_trainer.main(self.args)
            elif self.args.cls:
                cls_trainer.main(self.args)
            elif self.args.mae:
                mae_trainer.main(self.args)
            else:
                trainer.main(self.args)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Exception in job: {e}\n{tb}")

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

        base_temp_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, base_temp_dir, ignore_errors=True)
        triton_home = os.path.join(base_temp_dir, ".triton")
        triton_cache = os.path.join(base_temp_dir, ".triton_cache") 
        inductor_cache = os.path.join(base_temp_dir, ".torch_inductor")
        print(f"TRITON_HOME: {triton_home}")

        os.makedirs(triton_home, exist_ok=True)
        os.makedirs(triton_cache, exist_ok=True)
        os.makedirs(inductor_cache, exist_ok=True)

        # 5. 设置环境变量
        os.environ["TRITON_HOME"] = triton_home
        os.environ["TRITON_CACHE_DIR"] = triton_cache
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache


import re
def extract_nodes_from_filenames(folder):
    excluded_nodes = set()
    files = os.listdir(folder)
    for fname in files:
        parts = fname.split(',')
        for part in parts:
            node_match = re.search(r'(learnfair\d{4,})', part)
            if node_match:
                excluded_nodes.add(node_match.group(1))
    return sorted(excluded_nodes)

def bad_nodes_from_all_runs(base_dir):
    nodes_set = set()
    for run in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, run)
        if os.path.isdir(run_path):
            nodes = extract_nodes_from_filenames(run_path)
            for node in nodes:
                nodes_set.add(node)
    return nodes_set

def main():
    args = parse_args()
    print("Running job with name:", args.job_name)
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    suspect_nodes = [
        "learnfair2085", "learnfair2086", "learnfair2087", "learnfair2088",
        "learnfair2097", "learnfair2098", "learnfair2099", "learnfair2100",
        "learnfair2153", "learnfair2154", "learnfair2155", "learnfair2156",
        "learnfair2165", "learnfair2166", "learnfair2167", "learnfair2168",
        "learnfair2245", "learnfair2246", "learnfair2247", "learnfair2248",
        "learnfair2249", "learnfair2250", "learnfair2251", "learnfair2252",
        "learnfair2253", "learnfair2254", "learnfair2255", "learnfair2256",
        "learnfair2257", "learnfair2258", "learnfair2259", "learnfair2260", 
        "learnfair2425", "learnfair2426", "learnfair2427", "learnfair2428",
        # Adding nodes from first error log
        "learnfair1105", "learnfair1891", "learnfair1421", "learnfair1464",
        "learnfair1462", "learnfair1458", "learnfair1545", "learnfair1460",
        "learnfair1547", "learnfair1888", "learnfair1892", "learnfair1436",
        "learnfair1435", "learnfair1548", "learnfair1886", "learnfair1422",
        # Adding nodes from second error log (clip_config)
        "learnfair2133", "learnfair2134", "learnfair2135", "learnfair2136",
        "learnfair2201", "learnfair2202", "learnfair2203", "learnfair2204",
        "learnfair2229", "learnfair2230", "learnfair2231", "learnfair2232",
        "learnfair2289", "learnfair2290", "learnfair2291", "learnfair2292",
        # Adding nodes from third error log (dino_config)
        "learnfair0367", "learnfair0683", "learnfair0693", "learnfair0702",
        "learnfair0716", "learnfair0722", "learnfair0747", "learnfair0782",
        "learnfair0785", "learnfair0791", "learnfair0795", "learnfair0810",
        "learnfair0857", "learnfair0886", "learnfair0893", "learnfair0923",
        # [2026,2053-2055,2221-2224,2261-2264,2465-2468]
        "learnfair2026", "learnfair2053", "learnfair2054", "learnfair2055",
        "learnfair2221", "learnfair2222", "learnfair2223", "learnfair2224",
        "learnfair2261", "learnfair2262", "learnfair2263", "learnfair2264",
        "learnfair2465", "learnfair2466", "learnfair2467", "learnfair2468",
        # Adding nodes from latest error log
        "learnfair2029", "learnfair2067", "learnfair2068", "learnfair2205",
        "learnfair2206", "learnfair2207", "learnfair2208", "learnfair2217",
        "learnfair2218", "learnfair2219", "learnfair2220", "learnfair2309",
        "learnfair2310", "learnfair2311", "learnfair2312", "learnfair2065",
        # Adding nodes from job 50308264 error log
        "learnfair2233", "learnfair7472", "learnfair7462", "learnfair2328",
        "learnfair2360", "learnfair7734", "learnfair7735", "learnfair7485",
        "learnfair7459", "learnfair7523", "learnfair2341", "learnfair2174",
        "learnfair2392", "learnfair2180", "learnfair2157", "learnfair2151",
        # Adding nodes from job 50308255 error log
        "learnfair2419", "learnfair2420", "learnfair2417", "learnfair2124",
        "learnfair2239", "learnfair2078", "learnfair2077", "learnfair2121",
        "learnfair2238", "learnfair2122", "learnfair2237", "learnfair2044",
        "learnfair2079", "learnfair2240", "learnfair2080", "learnfair2042",
        # Adding nodes from job 50324959 error log
        "learnfair0355", "learnfair0340", "learnfair0351", "learnfair0337",
        "learnfair5101", "learnfair5102", "learnfair5103", "learnfair5130",
        "learnfair5131", "learnfair5133", "learnfair5134", "learnfair5135",
        "learnfair5151", "learnfair5154", "learnfair5156", "learnfair5157",
        "learnfair5159", "learnfair5233", "learnfair5234", "learnfair5241",
        "learnfair0356", "learnfair0346", "learnfair0353", "learnfair0347",
        # Adding nodes from latest error log
        "learnfair1262", "learnfair1264", "learnfair1406", "learnfair1407",
        "learnfair1699", "learnfair1700", "learnfair1701", "learnfair1702",
        "learnfair1729", "learnfair1730", "learnfair1731", "learnfair1732",
        "learnfair1733", "learnfair1734", "learnfair1735", "learnfair1736",
        "learnfair5105", "learnfair5106", "learnfair5109", "learnfair5111",
        "learnfair5197", "learnfair5198", "learnfair5199", "learnfair5200",
        "learnfair5201", "learnfair5202", "learnfair5203", "learnfair5204",
        "learnfair5205", "learnfair5206", "learnfair5207", "learnfair5208",
        "learnfair5053", "learnfair5054", "learnfair5055", "learnfair5056",
        "learnfair5057", "learnfair5059", "learnfair5060", "learnfair5061",
        "learnfair5062", "learnfair5063", "learnfair5064", "learnfair5125",
        "learnfair5126", "learnfair5128", "learnfair5132", "learnfair5136",
        # Adding nodes from latest error log
        "learnfair5149", "learnfair5150", "learnfair5152", "learnfair5153",
        "learnfair5155", "learnfair5158", "learnfair5246", "learnfair5247",
        "learnfair5248", "learnfair5249", "learnfair5250", "learnfair5252",
        "learnfair5253", "learnfair5254", "learnfair5255", "learnfair5256",
        # Adding nodes from job 51554249
        "learnfair2028", "learnfair2032", "learnfair2041", "learnfair2060",
        "learnfair2074", "learnfair2082", "learnfair2083", "learnfair2085",
        "learnfair2269", "learnfair2270", "learnfair2296", "learnfair2389",
        "learnfair2391", "learnfair2401", "learnfair2402", "learnfair2403",
        # Adding nodes from job 51553780
        "learnfair5041", "learnfair5042", "learnfair5044", "learnfair5045",
        "learnfair5047", "learnfair5048", "learnfair5185", "learnfair5186",
        "learnfair5187", "learnfair5188", "learnfair5189", "learnfair5190",
        "learnfair5191", "learnfair5194", "learnfair5195", "learnfair5196",
        # Adding nodes learnfair[5078-5079,5081-5082,5087-5088,5127,5178-5179,5181-5182,5210,5218-5220,5244]
        "learnfair5078", "learnfair5079", "learnfair5081", "learnfair5082",
        "learnfair5087", "learnfair5088", "learnfair5127", "learnfair5178",
        "learnfair5179", "learnfair5181", "learnfair5182", "learnfair5210",
        "learnfair5218", "learnfair5219", "learnfair5220", "learnfair5244",
        # Adding nodes from latest error log
        "learnfair1104", "learnfair1123", "learnfair1197", "learnfair1214",
        "learnfair1221", "learnfair1230", "learnfair1254", "learnfair1582",
        "learnfair1584", "learnfair1687", "learnfair1690", "learnfair1761",
        "learnfair1820", "learnfair1963", "learnfair2008", "learnfair2016",
        # Adding nodes learnfair[1165-1166,1168,1193,1198-1200,1213,1215,1521-1523,1543-1544,1610-1611]
        "learnfair1165", "learnfair1166", "learnfair1168", "learnfair1193",
        "learnfair1198", "learnfair1199", "learnfair1200", "learnfair1213",
        "learnfair1215", "learnfair1521", "learnfair1522", "learnfair1523",
        "learnfair1543", "learnfair1544", "learnfair1610", "learnfair1611",
        # Adding nodes learnfair[5067,5069-5071,5074,5161-5162,5165-5166,5169-5170]
        "learnfair5067", "learnfair5069", "learnfair5070", "learnfair5071",
        "learnfair5074", "learnfair5161", "learnfair5162", "learnfair5165",
        "learnfair5166", "learnfair5169", "learnfair5170",
        # Adding nodes from error log 53254451
        "learnfair5107", "learnfair5108", "learnfair5110", "learnfair5112",
        "learnfair5235", "learnfair5236", "learnfair5237", "learnfair5238",
        "learnfair5239", "learnfair5243", "learnfair5257", "learnfair5259",
        "learnfair5262", "learnfair5264", "learnfair5265", "learnfair5266",
    ]

    is_h100 = "h100" in args.qos or "h200" in args.qos
    if not is_h100:
        bad_nodes = bad_nodes_from_all_runs("/private/home/mingyangd/dmy/nn_flow/runs")
        for node in bad_nodes:
            if node not in suspect_nodes:
                suspect_nodes.append(str(node))
        print(suspect_nodes)
    kwargs['slurm_exclude'] = 'learnfair7516,learnfair7518,learnfair7519,learnfair7576,learnfair7578,learnfair7625,learnfair7627,' \
                              'learnfair7552,learnfair7553,learnfair7554,learnfair7555,learnfair7596,learnfair7597,learnfair7620,' \
                              'learnfair7621,learnfair7622,learnfair7623,learnfair7573,learnfair7564,learnfair7565,learnfair7566,' \
                              'learnfair7567,learnfair7664,learnfair7665,learnfair7666,learnfair7667,learnfair7556,learnfair7557,' \
                              'learnfair7558,learnfair7559,learnfair7560,learnfair7561,learnfair7562,learnfair7563,learnfair7636,' \
                              'learnfair7637,learnfair7638,learnfair7677,learnfair7678,learnfair7679,learnfair7685,learnfair7686,' \
                              'learnfair7687,learnfair7545,learnfair7546,learnfair7547,learnfair7483,learnfair7633,learnfair7635,' \
                              'learnfair7650,learnfair7651,learnfair7672,learnfair7675,learnfair7688,learnfair7690,learnfair7702,' \
                              'learnfair7703,learnfair7528,learnfair7530,learnfair7531,learnfair7540,learnfair7541,learnfair7542,' \
                              'learnfair7543,learnfair7585,learnfair7586,learnfair7587,learnfair7616,learnfair7619,learnfair7536,' \
                              'learnfair7537,learnfair7538,learnfair7539,learnfair7648,learnfair7663,learnfair7704,learnfair7705,' \
                              'learnfair7706,learnfair7707,learnfair7590,learnfair7591,learnfair7626,learnfair7649,learnfair7662,' \
                              'learnfair7548,learnfair7549,learnfair7550,learnfair7551,learnfair7470,learnfair7488,learnfair7490,' \
                              'learnfair7491,learnfair7657,learnfair7708,learnfair7465,learnfair7609,learnfair7610,learnfair7611,' \
                              'learnfair7716,learnfair7718,learnfair7719,learnfair7630,learnfair7631,learnfair7641,learnfair7642,' \
                              'learnfair7643,learnfair7692,learnfair7694,learnfair7695,learnfair7612,learnfair7613,learnfair7614,' \
                              'learnfair7532,learnfair7533,learnfair7534,learnfair7724,learnfair7725,learnfair7726,learnfair7727,' \
                              'learnfair7535,learnfair7724,learnfair7725,learnfair7726,learnfair7727,learnfair7569,learnfair7570,' \
                              'learnfair7571,learnfair7644,learnfair7676,learnfair7711,learnfair7696,learnfair7697,learnfair7660,' \
                              'learnfair7661,learnfair7713,learnfair7714,learnfair7673,learnfair7693,learnfair7698,learnfair7709,' \
                              'learnfair7645,learnfair7646,learnfair7484,learnfair7489,learnfair7497,learnfair7499,learnfair7515,' \
                              'learnfair7487,learnfair7467,learnfair7510,learnfair7525,learnfair7604,learnfair7680,learnfair7682,' \
                              'learnfair7720,learnfair7723,learnfair7529,learnfair7583,learnfair7618,learnfair7632,learnfair7639,' \
                              'learnfair7659,learnfair7652,learnfair7653,learnfair7654,learnfair7655,learnfair7668,learnfair7669,' \
                              'learnfair7670,learnfair7671,learnfair7468,learnfair7628,learnfair7629,learnfair7568,learnfair7640,' \
                              'learnfair7624,learnfair7482,learnfair7588,learnfair7589,learnfair7605,learnfair7607,learnfair7506,' \
                              'learnfair7512,learnfair7732,learnfair7733,learnfair7600,learnfair7602,learnfair7647,learnfair7656,' \
                              'learnfair7700,learnfair7520,learnfair7521,learnfair7728,learnfair7729,learnfair7731,learnfair7502,' \
                              'learnfair7509,learnfair7584,learnfair7592,learnfair7599,learnfair7608,learnfair7594,learnfair7473,' \
                              'learnfair7474,learnfair7478,learnfair7505,learnfair7577,learnfair7691,learnfair7699,learnfair7581,' \
                              'learnfair1613,learnfair1614,learnfair1615,learnfair1616,learnfair1805,learnfair1806,learnfair1807,' \
                              'learnfair1808,learnfair1809,learnfair1810,learnfair1811,learnfair1812,learnfair1441,learnfair1442,' \
                              'learnfair1443,learnfair1444,learnfair1445,learnfair1446,learnfair1447,learnfair1448,learnfair1449,' \
                              'learnfair7457,learnfair7458,learnfair7508,learnfair7498,learnfair7460,learnfair7461,learnfair7463,' \
                              'learnfair7500,learnfair7479,learnfair7494,learnfair7495,learnfair7501,learnfair7048,learnfair7049,' \
                              'learnfair7052,learnfair7053,learnfair7056,learnfair7058,learnfair7060,learnfair7062,learnfair7064,' \
                              'learnfair7065,learnfair7066,learnfair7067,learnfair7068,learnfair7069,learnfair7070,learnfair7071,' \
                              'learnfair7721,learnfair7722,learnfair7476,learnfair7477,learnfair7511,learnfair7513,learnfair7514,' \
                              'learnfair7579,learnfair7593,learnfair7595,learnfair7601,learnfair7603' + ','.join(suspect_nodes)
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'

    if args.comment:
        kwargs['slurm_comment'] = args.comment
    if args.qos:
        kwargs['slurm_qos'] = args.qos
    if args.account:
        kwargs['slurm_account'] = args.account
    if args.job_name:
        kwargs['slurm_job_name'] = args.job_name
    if args.requeue:
        kwargs['slurm_requeue'] = args.requeue
    if is_h100:
        cpus_per_task = 10
        mem_gb = 80 * num_gpus_per_node
        kwargs.pop('slurm_exclude')
    else:
        cpus_per_task = 6
        mem_gb = 40 * num_gpus_per_node
    executor.update_parameters(
        mem_gb=mem_gb,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        setup=[
            "export FI_EFA_SET_CUDA_SYNC_MEMOPS=0",
            # "export NCCL_P2P_NET_CHUNKSIZE=524288",
            "export NCCL_RAS_ENABLE=0",
            # "export NCCL_IB_DISABLE=1",
            "export NCCL_SOCKET_IFNAME=eth0",
            # "export NCCL_P2P_DISABLE=1",
            "export NCCL_DEBUG=INFO",
            "export XDG_CACHE_HOME=/checkpoint/flows/mingyangd/.cache",
            "export NCCL_NSOCKS_PERTHREAD=2",
            "export NCCL_SOCKET_NTHREADS=4"
        ],
        **kwargs
    )

    executor.update_parameters(name=args.job_name)

    args.output_dir = args.job_dir

    def multi_jobs(args, n_retries, init_id=None):
        last_id = None
        for i in range(args.num_retries + 1):
            args.dist_url = get_init_file().as_uri()
            trainer = Trainer(args)
            if i > 0:
                executor.update_parameters(slurm_additional_parameters={'dependency': f'afterany:{last_id}'})
            else:
                extra = dict()
                if init_id is not None:
                    extra['dependency'] = f'afterany:{init_id}'
                executor.update_parameters(slurm_additional_parameters=extra)
            job = executor.submit(trainer)
            print(f"Submitted job {job.job_id}")
            last_id = job.job_id
        return last_id
    
    if not args.both:
        multi_jobs(args, args.num_retries)
    else:
        mae_args = deepcopy(args)
        mae_args.mae = True
        last_mae_id = multi_jobs(mae_args, args.num_retries)
        multi_jobs(args, args.num_retries, init_id=last_mae_id)

if __name__ == "__main__":
    main()
