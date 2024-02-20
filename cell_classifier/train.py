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
from torch.utils.data import Dataset, random_split

from torchvision import datasets, models, transforms
from torchvision.models import resnet101, resnet, resnet50
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
        # model = resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V2)
        model = resnet101(weights=resnet.ResNet101_Weights.IMAGENET1K_V2)
    else:
        model = resnet101()
        
    # model = models.inception_v3(pretrained=pretrained)
      
    in_features = model.fc.in_features
    
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                    momentum=0.0,
                    nesterov=False,
                    weight_decay=.0,
                    )

    #optimizer = optim.NAdam(model.parameters(), lr=lr,
    #                       weight_decay=.00001, eps=5e-8)
    
    exp_lr_sched = lr_scheduler.StepLR(optimizer=optimizer, step_size=20,
                                       gamma=0.9)
 
    
    return model, loss, optimizer, exp_lr_sched


class cdataset(Dataset):
    
    def __init__(self, dataset, transform=None, classmap=None):
        self.dataset = dataset
        self.transform = transform
        self.classmap = classmap
        
    def __getitem__(self, index):
        
        data = self.dataset[index][0]
        
        if self.transform:
            x = self.transform(data)
        else:
            x = data
            
        _y = self.dataset[index][1]
        if self.classmap is None:
            y = _y
        else:
            y = self.classmap[_y]
            
            
        return x, y 
    
    def __len__(self):
        return len(self.dataset)


def get_dataloader(datadir, xforms, batchsize=8, classmap=None):
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
    # ds= {x: datasets.ImageFolder(os.path.join(datadir, x),
    #                              xforms[x])
    #      for x in ['train', 'val']}
   

    _ds = datasets.ImageFolder(datadir)
    _ds.loader = tifffile.imread
    
    nt = int(.8*(len(_ds)))
    nv = int(.5*(len(_ds) - nt))
    ntest = len(_ds) - nv - nt
    _train, _val, _test = random_split(_ds, (nt, nv, ntest))

    train_ds = cdataset(_train, xforms['train'], classmap=classmap)
    val_ds = cdataset(_val, xforms['val'], classmap=classmap)
    # test_ds = cdataset(_test, xforms['val'], classmap=classmap)
    
    test_images = [_ds.samples[i] for i in _test.indices]
    ds = {'train':train_ds, 'val':val_ds}
    
    dataloaders = {x: DataLoader(ds[x], batch_size=batchsize,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    ds_sizes = {x: len(ds[x]) for x in ['train', 'val']}
    
    return ds, dataloaders, ds_sizes, test_images


def main(root='Data', epochs=100, cropsize=(400,400),
         model=None, classmap=None,
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

    print(num_classes)

    xf = {'train':xforms.get_transforms(cropsize=cropsize),
          'val':xforms.get_transforms(cropsize=cropsize, train=False)}
    
        
    ds, dataloaders, ds_sizes, test_images = get_dataloader(root, xf,
                                    batchsize=batch_size, classmap=classmap)

    if model is None:
        model, func_loss, optimizer, lr_sched \
            = get_model(num_classes, device, lr=lr)
    else:
        func_loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.0,
                        nesterov=False,
                        weight_decay=.0,
                        )

    
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
    return model, test_images