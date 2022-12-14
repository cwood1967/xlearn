'''
Using the information from
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''

from re import I
import time
import os
import copy

import torch
from torch.utils.data import DataLoader

import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.backends import cudnn
import numpy as np
import torchvision

from torchvision import datasets, models, transforms
from torchvision.models import resnet101, resnet
from matplotlib import pyplot as plt
import tifffile

from . import xforms


def get_model(num_classes, device, pretrained=True, lr=0.001):
    '''
    Get the pretrained model with the desired number of classes
    
    Parmeters
    ---------
    num_classes : int
        the number of classes in the model
        
    device : str
        "cuda" is training on a gpu, "cpu" otherwise
        
    pretrained : bool
        True is using a pretrained model, False otherwise
        
    Returns
    -------
    model : torch.nn.Module 
        The resnet101 model
    '''
    if pretrained:
        model = resnet101(weights=resnet.ResNet101_Weights.IMAGENET1K_V2)
    else:
        model = resnet101()
        
    # model = models.inception_v3(pretrained=pretrained)
      
    in_features = model.fc.in_features
    
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    exp_lr_sched = lr_scheduler.StepLR(optimizer=optimizer, step_size=20,
                                       gamma=0.5)
 
    
    return model, loss, optimizer, exp_lr_sched

def get_dataloader(datadir, xforms, batchsize=8):
    """ Create a dataloader for training classifier 

    Parameters
    ----------
    datadir : str 
        The root directory for Images. Different classes are in separate
        directories.
    xforms : dict 
        Augmentation transforms
        'train' : key for training transforms 
        'val' : key for validation transforms
    batchsize : int, optional
        Size of the dataset, by default 8

    Returns
    -------
    ds : dict
        Training and Validation dataset folders
    dataloaders : dict
        DatalLoaders for "train" and "val"
    ds_sizes : dict
        The length of the training and validation datasets.
    """
    ds= {x: datasets.ImageFolder(os.path.join(datadir, x),
                                 xforms[x])
         for x in ['train', 'val']}
    
    ds['train'].loader = tifffile.imread
    ds['val'].loader = tifffile.imread
    
    dataloaders = {x: DataLoader(ds[x], batch_size=batchsize,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    ds_sizes = {x: len(ds[x]) for x in ['train', 'val']}
    
    return ds, dataloaders, ds_sizes


def main(root='Data', epochs=100, cropsize=(400,400),
         batch_size=8, pretrained=True, num_classes=4, lr=.001):
    """ Run the training

    Parameters
    ----------
    root : str, optional
        The location of the training data
    epochs : int, optional
        The number of epochs, default 100
    cropsize : tuple, optional
        The size to crop the images in augmentation, by default (400,400)
    batch_size : int, optional
        The size of each batch, by default 8
    pretrained : bool, optional
        True to use pretrained weights, False to train from
        scratch, by default True
    num_classes : int, optional
        The number of class to train on, by default 4

    Returns
    -------
    resnet101(torch.nn.Module)
        The trained model at the iteration with the best accuracy
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # num_classes = 4

    xf = {'train':xforms.get_transforms(cropsize=cropsize),
          'val':xforms.get_transforms(cropsize=cropsize, train=False)}
    
        
    ds, dataloaders, ds_sizes = get_dataloader(root, xf, batchsize=batch_size)

    model, func_loss, optimizer, lr_sched = get_model(num_classes, device, lr=lr)
    
    bwt = copy.deepcopy(model.state_dict())
    best_acc = 0.0 

    for e in range(epochs):
        print(f'Epoch {e}/{epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = func_loss(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #if phase == 'val':
                    #print(preds)
                    #print(labels.data)
                    #print(running_corrects)
                
            if phase == 'train':
                lr_sched.step()
                print(lr_sched.get_last_lr())
                 
            epoch_loss = running_loss / ds_sizes[phase]
            epoch_acc = running_corrects.double() / ds_sizes[phase]
            
            print(f"{phase} Loss {epoch_loss:4f} Acc: {epoch_acc:4f}, {running_corrects}")
            
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
            
             
    model.load_state_dict(best_model_wts)
    return model