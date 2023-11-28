import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrsched
import torchvision as tv
import torchvision.transforms as T

import argparse
import random
import numpy as np

from torch.optim import Optimizer
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from typing import Optional, Tuple, Callable, Dict, Any, List


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
    
