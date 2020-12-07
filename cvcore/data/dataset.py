from collections import OrderedDict
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import gc
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import pytorch
from .randaug import RandAugment, to_tensor_randaug
from .albu import AlbuAugment, to_tensor_albu
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
import glob
import random
import PIL

class ImageLabelDataset(Dataset):
    def __init__(self, images, label, soft_label=None, mode='train', cfg=None):
        self.cfg = cfg
        self.images = images
        self.mode = mode
        assert self.cfg.DATA.TYPE in ("multiclass", "multilabel")
        assert self.mode in ("train", "valid", "test")
        if mode == "train":
            self.dir = self.cfg.DIRS.TRAIN_IMAGES
        elif mode == "valid":
            self.dir = self.cfg.DIRS.VALIDATION_IMAGES
        else:
            self.dir = self.cfg.DIRS.TEST_IMAGES

        if self.mode in ("train", "valid"):
            self.label = label
            self.soft_label = soft_label

        assert self.cfg.DATA.AUGMENT in ("randaug", "albumentations")
        if self.cfg.DATA.AUGMENT == "randaug":
            self.transform = RandAugment(n=self.cfg.DATA.RANDAUG.N,
                m=self.cfg.DATA.RANDAUG.M, random_magnitude=self.cfg.DATA.RANDAUG.RANDOM_MAGNITUDE)
            self.to_tensor = to_tensor_randaug()
        elif self.cfg.DATA.AUGMENT == "albumentations":
            self.transform = AlbuAugment(self.cfg)
            self.to_tensor = to_tensor_albu(self.cfg)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lb = None
        if self.mode in ("train", "valid"):
            lb = self.label[idx]
            soft_lb = torch.Tensor(self.soft_label[idx])
            if self.cfg.DATA.TYPE == "multilabel":
                lb = lb.astype(np.float32)
                if not isinstance(lb, list):
                    lb = [lb]
                lb = torch.Tensor(lb)
        sop_study = self.images[idx]
        image_name = f"{self.dir}/{sop_study}"
        image = Image.open(image_name)
        if self.cfg.DATA.INP_CHANNEL == 3:
            image = image.convert("RGB")
        elif self.cfg.DATA.INP_CHANNEL == 1:
            image = image.convert("L")
        if self.mode == "train":
            if isinstance(self.transform, AlbuAugment):
                image = np.asarray(image)
            elif isinstance(self.transform, RandAugment):
                image = image.resize(self.cfg.DATA.IMG_SIZE, resample=PIL.Image.BILINEAR)
            image = self.transform(image)
            image = self.to_tensor(image)
            return image, lb, soft_lb
        elif self.mode == "valid":
            if isinstance(self.transform, AlbuAugment):
                image = np.asarray(image)
            elif isinstance(self.transform, RandAugment):
                if self.cfg.DATA.VALID_DEF_SIZE:
                    image = image.resize((600, 800), resample=PIL.Image.BILINEAR)
                else:
                    image = image.resize(self.cfg.DATA.IMG_SIZE, resample=PIL.Image.BILINEAR)
            image = self.to_tensor(image)
            return image, lb, soft_lb
        else:
            if isinstance(self.transform, AlbuAugment):
                image = np.asarray(image)
            elif isinstance(self.transform, RandAugment):                
                if self.cfg.DATA.VALID_DEF_SIZE:
                    image = image.resize((600, 800), resample=PIL.Image.BILINEAR)
                else:
                    image = image.resize(self.cfg.DATA.IMG_SIZE, resample=PIL.Image.BILINEAR)
            image = self.to_tensor(image)
            return image, sop_study

def make_image_label_dataloader(cfg, mode, images, labels, soft_label):
    dataset = ImageLabelDataset(images, labels, soft_label, mode=mode, cfg=cfg)
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset,
                         np.random.choice(np.arange(len(dataset)), 500))
    shuffle = True if mode == "train" else False
    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=shuffle,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader
