import random
from pytools import Norm
import torch
import numpy as np

from torchvision.transforms import RandomRotation as torchRotation
from torchvision.transforms import ToPILImage
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import Normalize, ToTensor, RandomAutocontrast
from torchvision.transforms import RandomResizedCrop, CenterCrop, GaussianBlur
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


def get_transforms(cropsize=(400,400), prob=0.5, train=True):
    transforms = list()
    if train:
        transforms.extend(
            [ToTensor(),
             RandomResizedCrop(cropsize, scale=(0.9, 1.0),
                               ratio=(.9, 1.1)),
             RandomHorizontalFlip(),
             RandomVerticalFlip(),
             GaussianBlur(5),
             RandomAutocontrast()
            #  ToTensor(),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # transforms.extend([ToTensor(),
        #                    RandomCrop(cropsize),
        #                    RandomHFlip(prob),
        #                    RandomVFlip(prob),
        #                    #ToTensor()
        # ])
    else:
        transforms.extend([ToTensor(),
                           CenterCrop(cropsize)])
        
    return Compose(transforms)
    