from torchvision import transforms
import torch
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import random
import numpy as np
import torchvision.transforms.functional as TF
import random

from albumentations import Compose, HorizontalFlip, VerticalFlip, Normalize, RandomResizedCrop, OneOf, \
            Blur, CoarseDropout, GaussNoise, ShiftScaleRotate, ElasticTransform, IAAAdditiveGaussianNoise, \
            Equalize, RandomBrightnessContrast, GridDistortion, RandomResizedCrop, OpticalDistortion, \
            Posterize, MedianBlur, Solarize, CenterCrop, Resize         
import albumentations.augmentations.functional as AF
from albumentations.pytorch import ToTensor
from albumentations.core.transforms_interface import ImageOnlyTransform


class AlbuAugment:
    def __init__(self, cfg):
        transformation = []
        transformation += [
            RandomResizedCrop(cfg.DATA.IMG_SIZE[0], cfg.DATA.IMG_SIZE[1], scale=(0.2, 1.0)),
            ShiftScaleRotate(),
            RandomBrightnessContrast(),
            HorizontalFlip(),
            VerticalFlip(),
            OneOf([
                GridDistortion(),
                OpticalDistortion(),
                ElasticTransform(approximate=True)], p=0.8),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
                MedianBlur(blur_limit=3),
                Blur(blur_limit=3)], p=0.8),
            CoarseDropout(max_holes=2, max_height=36, max_width=36)
        ]

        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']

class to_tensor_albu:
    def __init__(self, cfg):
        if cfg.DATA.VALID_DEF_SIZE:
            transformation = [Resize(600, 800)]
        else:
            transformation = [Resize(cfg.DATA.IMG_SIZE[0], cfg.DATA.IMG_SIZE[1])]
        transformation += [Normalize(),
                           ToTensor()]
        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']
