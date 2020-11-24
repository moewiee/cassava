import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from cvcore.utils import save_checkpoint
import pandas as pd
import torchvision.transforms.functional as TF


def valid_model(_print, cfg, model, valid_loader,
                loss_function, score_function, epoch,
                best_metric=None, checkpoint=False):
    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, lb) in enumerate(tbar):
            image = image.cuda()
            lb = lb.cuda()
            output = model(image)
            if cfg.INFER.TTA:
                output += model(TF.vflip(image))
                output += model(TF.hflip(image))

            preds.append(output.cpu())
            targets.append(lb.cpu())

    preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)

    # record
    val_loss = loss_function(preds.float(), targets)
    score = score_function(targets, torch.argmax(preds, 1))

    _print(f"VAL LOSS: {val_loss:.5f}, SCORE: {score:.5f}")
    # checkpoint
    if checkpoint:
        is_best = score > best_metric
        best_metric = max(score, best_metric)
        save_dict = {"epoch": epoch + 1,
                     "arch": cfg.NAME,
                     "state_dict": model.state_dict(),
                     "best_metric": best_metric}
        save_filename = f"{cfg.NAME}.pth"
        if is_best: # only save best checkpoint, no need resume
            save_checkpoint(save_dict, is_best,
                            root=cfg.DIRS.WEIGHTS, filename=save_filename)
            print("score improved, saving new checkpoint...")
        return score, best_metric


def test_model(cfg, model, test_loader):
    model.eval()

    preds = []
    tbar = tqdm(test_loader)

    with torch.no_grad():
        for i, (image, name) in enumerate(tbar):
            image = image.cuda()
            output = model(image)
            if cfg.INFER.TTA:
                output += model(TF.vflip(image))
                output += model(TF.hflip(image))
            output = torch.argmax(output.cpu(), 1).numpy()
            for n, o in zip(name, output):
                preds.append([n, o])
    pd.DataFrame(data=preds, columns=["image_id", "label"]).to_csv("submission.csv", index=False)