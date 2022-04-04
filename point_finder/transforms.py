import random
import torch
import numpy as np

from torchvision.transforms import RandomRotation as torchRotation
from torchvision.transforms import RandomCrop as rc
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F

class ToTensor(object):
    def __call__(self, image, bbs):
        _xt = F.to_tensor(image).type(torch.FloatTensor)
        _bt = F.to_tensor(bbs).type(torch.FloatTensor)
        return _xt, _bt 

class RandomCrop(object):
    
    def __init__(self, size):
        self.size = size 

    def __call__(self, image, _bbs):
        
        go = 0
        while go == 0:
            y, x, h, w = rc.get_params(image, self.size)

            bbs = np.moveaxis(_bbs.numpy(), 0, -1)
            bbs = bbs - np.array([[y, x]])

            bbslist = list()
            for bb in list(bbs):
                if bb.min() >= 0:
                    if bb.max() < max(self.size):
                        bbslist.append(bb)
                
            print(y, x,h, w, len(bbslist))

            if len(bbslist) > 0:
                go = 1    
            else:
                continue
            
        image = F.crop(image, y,x, h, w)  
        bbs = F.to_tensor(np.array(bbslist))
        print(image.shape)
        return image, bbs
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbs):
        for t in self.transforms:
            image, bbs = t(image, bbs)        
        return image, bbs

def get_transforms(cropsize=(800, 800), train=True):
    if train:
        transforms = [ToTensor(),
                      RandomCrop(cropsize),
                      ]
    else:
        transforms = [ToTensor()]
    
    return Compose(transforms)