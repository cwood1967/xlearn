import random
import torch

from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class RandomHFlip(object):
    """
    Class to flip an image horizontally
    """
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, mask):
        """ used to perfrom the transform
        
        Parameters
        ----------
        image: a torch (tensor channel, y, x)
        mask: a torch (y, x)

        Returns
        -------
        Transformed image and target
        """

        if random.random() > self.prob:
            image = image.flip(-1)
            mask = mask.flip(-1)

        return image, mask

class RandomCrop(object):
    """
    Class to randomly crop to a size 
    """
    def __init__(self, size):
        '''
        Parameters
        ----------
        size: tuple(y, x) - size of the cropped image
        '''
        self.size = size

    def __call__(self, image, mask):
        """ Performs the  crop transform

        Parameters
        ----------
        image: torch tensor (channel, y, x)
        mask: torch tensor (channel, y, x)

        Returns
        -------
        Transformed image and mask
        """
        h, w = image.shape[-2:]
        hc = self.size[1]
        wc = self.size[0]
        ymax = h - hc
        xmax = w - wc

        ry = random.randint(0, ymax)
        rx = random.randint(0, xmax)

        image = image[:, ry:ry + hc, rx:rx + wc]
        mask = mask[:, ry:ry + hc, rx:rx + wc]
        return image, mask


class RandomVFlip(object):
    """
    Class to flip an image vertically
    """
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, mask):
        """ used to perfrom the transform
        
        Parameters
        ----------
        image: a torch (tensor channel, y, x)
        mask: a torch (tensor channel, y, x)

            Returns
            -------
            Transformed image and mask
            """

        if random.random() < self.prob:
            image = image.flip(-2)
            mask = mask.flip(-2)

        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        """
        Convert numpy array to tensor. Will move channel dimension to in front
        of y,x. For example, [512, 512, 3] -> [3, 512, 512]
        Two dimensional images will be converted: [512, 512] -> [1, 512, 512]
        """
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        return image, mask

def get_transforms(cropsize=(400,400), prob=0.5, train=True):
    transforms = list()
    if train:
        transforms.extend([ToTensor(), RandomCrop(cropsize),
                        RandomHFlip(prob), RandomVFlip(prob)])
    else:
        transforms = ([ToTensor()])
        
    return Compose(transforms)
    