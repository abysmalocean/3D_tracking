import argparse
import json
import os
import sys

from numba.core.errors import (NumbaDeprecationWarning, 
                              NumbaPendingDeprecationWarning, 
                              NumbaWarning)
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import yaml

from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument(
        "config",
        help="train config file path")
    
    parser.add_argument(
        "--work_dir", 
        help="The dir to save logs and models",
        required=True)
    
    parser.add_argument(
        "--resume_from",
        default=None,
        help="the checkpoint file to resume from")
    
    parser.add_argument(
        "--gups",
        type=int, 
        default=1,
        help="number of gpus to use (only applicable to non-distributed training)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="whether to evaluate the checkpoint during training")
    
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    print("|-***** The config file is : *****")
    for arg in vars(args):
        print("|--->  ", arg, getattr(args, arg))
    print("|_***** Finis the config file *****")
    return args


    
    
    

def main(): 
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    cfg.local_rank = args.local_rank
    cfg.work_dir = args.work_dir
    cfg.resume_from = args.resume_from
    
    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    # copy important files to backup
    backup_dir = os.path.join(cfg.work_dir, "det3d")
    os.makedirs(backup_dir, exist_ok=True)
    
    # TODO: dig into the build_detector
    model = build_detector(cfg.model, 
                           train_cfg=cfg.train_cfg, 
                           test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    print(cfg.workflow)
    #if len(cfg.workflow) == 2:
    #    datasets.append(build_dataset(cfg.data.val))
    
    
    #if not os.path.exists(cfg.checkpoint_config):
    #    # save det3d version, config file content and class names in
    #    # checkpoints as meta data
    #    cfg.checkpoint_config.meta = dict(
    #        config=cfg.text, CLASSES=datasets[0].CLASSES
    #    )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    distributed = False
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
    )
    
    print("program finished")
    


if __name__ == "__main__":
    main()

