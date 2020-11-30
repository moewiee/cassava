import gc
import os
import sys
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler

from sklearn.metrics import accuracy_score

from cvcore.config import get_cfg_defaults
from cvcore.data import make_image_label_dataloader
from cvcore.model import build_model
from cvcore.solver import make_optimizer, build_scheduler
from cvcore.utils import setup_determinism, setup_logger, load_checkpoint
from cvcore.tools import parse_args, train_loop, valid_model, copy_model, test_model
from cvcore.losses import LabelSmoothingCrossEntropy

scaler = GradScaler()
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args, cfg):
    # Set logger
    logger = setup_logger(
        args.mode,
        cfg.DIRS.LOGS,
        0,
        filename=f"{cfg.NAME}.txt")

    # Avoid possibly unbound
    scheduler = None
    train_loader = None
    valid_loader = None
    test_loader = None
    train_criterion = None
    valid_criterion = None
    make_dataloader = make_image_label_dataloader


    # Define model
    model = build_model(cfg)
    if cfg.SOLVER.SWA.ENABLED:
        model_swa = build_model(cfg)
    else:
        model_swa = None
    optimizer = make_optimizer(cfg, model)

    # Define loss
    if cfg.LOSS.NAME == "ce":
        valid_criterion = nn.CrossEntropyLoss().cuda()
        train_criterion = nn.CrossEntropyLoss().cuda()
    if cfg.LOSS.NAME == "smooth-ce":
        valid_criterion = nn.CrossEntropyLoss().cuda()
        train_criterion = LabelSmoothingCrossEntropy().cuda()

    model = model.cuda()
    model = nn.DataParallel(model)
    if cfg.SOLVER.SWA.ENABLED:
        model_swa = model_swa.cuda()
        model_swa = nn.DataParallel(model_swa)

    # Load checkpoint
    model, start_epoch, best_metric = load_checkpoint(args, logger.info, model)

    # Load and split data
    if args.mode in ("train", "valid"):
        df = pd.read_csv(cfg.DATA.CSV)
        valid_df = df[df["fold"].isin([args.fold])]
        valid_loader = make_dataloader(
            cfg, "valid", valid_df["image_id"].values, valid_df["label"].values)
        if args.mode == "train":
            train_df = df[~df["fold"].isin([args.fold])]
            if cfg.DATA.EXTERNAL:
                external_df = pd.read_csv(cfg.DATA.EXTERNAL)
                train_df = pd.concat([train_df, external_df])
            train_loader = make_dataloader(
                cfg, "train", train_df["image_id"].values, train_df["label"].values)
    elif args.mode == "test":
        test_df = pd.read_csv(cfg.DATA.TEST_CSV)
        if "fold" in test_df.columns:
            test_df = test_df[test_df["fold"] == args.fold]
        test_loader = make_dataloader(
                cfg, "test", test_df["image_id"].values, None)


    # Build training scheduler
    if args.mode in ("train", "trainseries"):
        scheduler = build_scheduler(cfg, len(train_loader))

    # Run script
    if args.mode == "train":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHES[-1]):
            if cfg.SOLVER.SWA.ENABLED and epoch == cfg.SOLVER.SWA.START_EPOCH:
                copy_model(model_swa, model)
            train_loop(logger.info, cfg, model,
                       model_swa if epoch >= cfg.SOLVER.SWA.START_EPOCH else None,
                       train_loader, train_criterion, optimizer,
                       scheduler, epoch, scaler)
            _, best_metric = valid_model(logger.info, cfg,
                                      model_swa if cfg.SOLVER.SWA.ENABLED and \
                                        epoch >= cfg.SOLVER.SWA.START_EPOCH else model,
                                      valid_loader, valid_criterion,
                                      accuracy_score, epoch, best_metric, True)
    elif args.mode == "valid":
        valid_model(logger.info, cfg, model,
                    valid_loader, valid_criterion,
                    accuracy_score, start_epoch)
    else:
        test_model(cfg, model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.opts != "":
        cfg.merge_from_list(args.opts)

    assert args.config
    print(cfg)

    # make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])
    # seed, run
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)
