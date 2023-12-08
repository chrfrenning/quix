import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T
import torchvision.transforms.v2 as v2
import warnings
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from typing import (
    Optional, Tuple, Callable, Dict, Any, List,
    Generic
)
from ..cfg import TDat, DataConfig

class GaussianBlurPIL:

    def __init__(self, p=1.0, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img:Image.Image):
        if random.random() <= self.p:
            filter = ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
            img = img.filter(filter)

        return img    


class SolarizePIL:

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img:Image.Image):
        if random.random() < self.p:
            img = ImageOps.solarize(img)
        return img


class GrayscalePIL:

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img:Image.Image):
        if random.random() < self.p:
            img = img.convert('L').convert('RGB')
        return img
    

class SaturationPIL:
    def __init__(self, p=1.0, par=0.3, alpha=0.5):
        self.p = p
        self.par = par
        self.alpha = alpha

    def __call__(self, img:Image.Image):
        if random.random() < self.p:
            fac = random.betavariate(self.alpha, self.alpha) * (1 + self.par)
            img = ImageEnhance.Color(img).enhance(fac)
        return img
    

class ContrastBrightnessPIL:

    def __init__(self, par=0.3):
        self.low = max([1-par, 0])
        self.high = 1+par
    
    def __call__(self, img:Image.Image):
        bfac = random.uniform(self.low, self.high)
        cfac = random.uniform(self.low, self.high)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(bfac)
            img = ImageEnhance.Contrast(img).enhance(cfac)
        else:
            img = ImageEnhance.Contrast(img).enhance(cfac)
            img = ImageEnhance.Brightness(img).enhance(bfac)
        return img


class NoiseTensor:

    def __init__(self, eps_sd_max=3e-2, alpha=0.5):
        self.eps_sd_max = eps_sd_max
        self.alpha = alpha

    def __call__(self, img:torch.Tensor):
        sd = random.betavariate(self.alpha, self.alpha) * self.eps_sd_max
        img.add_(torch.randn_like(img)*sd).clip_(0,1)
        return img
    
    
class ToRGBTensor:
    
    def __init__(self):
        self.totensor = T.ToTensor()
    
    def __call__(self, img:Image.Image):
        tensor = self.totensor(img)
        _, h, w = tensor.shape
        return tensor.expand(3, h, w)
    

class Identity:

    def __call__(self, *args):
        return args


INTERPOLATION_MODES = {
    'nearest':v2.InterpolationMode.NEAREST,
    'bilinear':v2.InterpolationMode.BILINEAR,
    'bicubic':v2.InterpolationMode.BICUBIC,
    'all':[
        v2.InterpolationMode.NEAREST,
        v2.InterpolationMode.BILINEAR,
        v2.InterpolationMode.BICUBIC
    ]
}

RANDAUG_DICT = {
    'none': (0, 0),
    'light': (2, 10),
    'medium': (2, 15),
    'strong': (2, 20),
}

def parse_train_augs(cfg:DataConfig, num_classes:Optional[int]=None) -> Tuple[Callable,Callable]:
    # Although some of the parameters are guaranteed to be in correct 
    # form in config, we use fallbacks in interest of good practice.
    intp_modes = INTERPOLATION_MODES.get(cfg.intp_modes, 'all')
    identity = Identity()

    # RandAug
    randaug = v2.RandAugment(*RANDAUG_DICT.get(cfg.randaug, 'none'))

    # Aug3 (DEiT paper)
    blurkernel = int(cfg.img_size * .1) | 1 # Use standard from SimCLR paper.
    aug3 = v2.RandomChoice([
        v2.RandomSolarize(0.5, 1.0),
        v2.ColorJitter(0, 0, (0., 1.5), 0), # Saturation instead of grayscale
        v2.GaussianBlur(blurkernel)
    ])

    # Set preaugmentations
    if cfg.randaug != 'none' and cfg.aug3:
        mainaug = v2.RandomChoice([aug3, randaug])
    elif cfg.aug3:
        mainaug = aug3
    elif cfg.randaug != 'none':
        mainaug = randaug
    else:
        mainaug = identity

    # Random resize crop (randomize interpolations)
    resizecrop = v2.RandomChoice([
        v2.RandomResizedCrop(
            cfg.img_size, cfg.rrc_scale, cfg.rrc_ratio, 
            interpolation=itm, antialias=True
        )
        for itm in intp_modes
    ])

    # Check number of classes for cutmix / mixup
    use_cutmix = cfg.cutmix_alpha > 0 and num_classes is not None
    use_mixup = cfg.mixup_alpha > 0 and num_classes is not None
    if use_cutmix and num_classes is None:
        warnings.warn('CutMix specified without number of classes. Dropping CutMix augmentation.')
    if use_mixup and num_classes is None:
        warnings.warn('MixUp specified without number of classes. Dropping MixUp augmentation.')
    num_classes = num_classes if num_classes is not None else 0

    # Final augmentations
    addaug = [
        v2.ColorJitter(0.3, 0.3) if cfg.jitter else identity,   # Only brightness and contrast
        v2.RandomHorizontalFlip() if cfg.hflip else identity,   # On by default
        v2.RandomVerticalFlip() if cfg.vflip else identity,     # Off by default
    ]

    # Mixup / Cutmix (50 - 50 chance)
    batch_augs = v2.RandomChoice([
        v2.MixUp(num_classes=num_classes, alpha=cfg.mixup_alpha) if use_mixup else identity,
        v2.CutMix(num_classes=num_classes, alpha=cfg.cutmix_alpha) if use_cutmix else identity,
    ])

    # Compose Augmentations
    sample_augs = v2.Compose([resizecrop, mainaug, *addaug])

    return sample_augs, batch_augs
    

def parse_val_augs(cfg:DataConfig, num_classes:Optional[int]=None) -> Callable:
    val_img_size = cfg.val_size if cfg.val_size is not None else cfg.img_size
    sample_augs = v2.RandomResizedCrop(val_img_size, (1.0,1.0), antialias=True)
    return sample_augs
