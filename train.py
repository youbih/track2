import os
import time
import random
import argparse
import numpy as np

import torch
from datetime import datetime
import torch.backends.cudnn as cudnn

import my_affectgpt.tasks as tasks
from my_affectgpt.common.config import Config
from my_affectgpt.common.dist_utils import get_rank, init_distributed_mode
from my_affectgpt.common.logger import setup_logger
from my_affectgpt.common.registry import registry
from my_affectgpt.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *

def setup_seeds(config): 
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help=(
            "overwrite params in yaml. "
            "Example: --options datasets.mercaptionplus.ratio=0.1 run.max_epoch=5 run.iters_per_epoch=1000"
        ),
    )
    args = parser.parse_args()
    return args

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base")) # 'affectgpt.runners.runner_base.RunnerBase'
    return runner_cls

def main():

    args = parse_args()
    cfg = Config(args)

    # 分布式训练：异步错误处理 + 超时，避免 NCCL 死锁时无限挂起
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "1800"
    job_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
    job_id = f"{job_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    print(job_id)
    print(f"cfg_path: {args.cfg_path}")

    # print logging files
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger() 
    cfg.pretty_print()

    # load task and start training
    task = tasks.setup_task(cfg) # video_text_pretrain
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg,
        job_id=job_id, 
        task=task, 
        model=model, 
        datasets=datasets
    )
    runner.train()

    if cfg.run_cfg.distributed:
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    main()
