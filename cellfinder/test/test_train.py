# %%
import autoreload
%load_ext autoreload
%autoreload 2
# %%
import os


from matplotlib import pyplot as plt
import numpy as np

import matplotlib.patches as patches
import torch
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import tifffile

from . import train
from  cellfinder import dataset
from cellfinder import transforms
from cellfinder.utils import image_to_patches
import cellfinder.utils

# %%

def run_test():
    xforms = transforms.get_transforms(cropsize=(400,400)) #, mean=(0.5,), std=(0.25,))
    print(xforms)
    data = dataset.PombeDataset('Data', 'Images', 'Masks', xforms)
    #print(data.images)
    vr, mr = data.__getitem__(3)
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].imshow(vr[0])
    xboxes = mr['boxes']
    for b in xboxes:
        print(b)
        rx, ry, rxf, ryf = b
        w = rxf - rx
        h = ryf - ry
        rp = patches.Rectangle((rx, ry), w, h, edgecolor='r', facecolor='none')
        _ = ax[0].add_patch(rp)
        
    ax[1].imshow(mr['masks'].sum(axis=0))

# %%
tm = train.main('Data', 'Images', 'Masks', epochs=20)
# %%
torch.save(tm.state_dict(), "Data/trained_model_20200930.pt")

# %%
#tm = train.get_model(2)
#tm.load_state_dict(torch.load("Data/trained_model.pt"))