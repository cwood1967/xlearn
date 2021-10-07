import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import dataset
from . import transforms


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    print("IFM", in_features_mask)
    hidden_layer = 256
    
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
    return model

    
def collate(batch):
    return tuple(zip(*batch))

def main(root='Data', image_dir='Images', mask_dir='Masks',
         epochs=50, cropsize=(400, 400), batch_size=8):
    '''
    Train the model and save network snapshots.
    
    root : str
       The path to the data directory. Default is 'Data'
    
    image_dir : str
        Directory inside to root where traing images are located. Default 'Images'
    
    mask_dir : str
        Directory for training masks. Default is 'Masks'
    
    epochs : int
        Number of epochs to train. Default 50
    
    cropsize : tuple (int, int)
        Size of patches to use in training. Default (400, 400)
    
    batch_size : int
        Size of image batches. Default 8
    
    Returns : MaskRCNN
        The trained pytorch Mask RCNN model
    '''

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    num_classes = 2
    
    xforms = transforms.get_transforms(cropsize=cropsize)
    data = dataset.PombeDataset(root, image_dir, mask_dir, xforms)
    data_test = dataset.PombeDataset(root, image_dir, mask_dir, xforms)
    
    indices = torch.randperm(len(data)).tolist()
    # data = torch.utils.data.Subset(data, indices[:8])

    data_test = torch.utils.data.Subset(data_test, indices[-1])
    
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate)
    
    data_test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate)
    
    
    model = get_model(2)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
                                weight_decay=0.0005)
    
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=10,
                                               gamma=0.1)
    
    num_epochs = epochs
    model.train()
    print(f'Training on {device}...')
    min_mask_loss = 10.0
    timefmt = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    modeldir = f"model_{timefmt}"
    os.mkdir(modeldir)
    for i in range(num_epochs):
        elosses = 0 
        for images, targets in data_loader:
            #print('LEN', len(images))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
           
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            elosses += losses 
            #if i % (num_epochs//100) == 0: 
            #print("**", i, losses)#['loss_mask'])
            # print("***", i, loss_dict['loss_box_reg'].cpu().detach().numpy(),
            #       loss_dict['loss_mask'].cpu().detach().numpy())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(i, elosses)
        if elosses < min_mask_loss:
            min_mask_loss = elosses        
        # if loss_dict['loss_mask'].cpu().detach().numpy() < min_mask_loss:
        #     min_mask_loss = loss_dict['loss_mask'].cpu().detach().numpy() 
            torch.save(model.state_dict(), f"{modeldir}/trained_min_mask_model_{i:04d}.pt")
            print("i, saved min mask loss")
            #lr_sched.step()
    print(loss_dict)
    torch.save(model.state_dict(), f"{modeldir}/trained_last_{i:04d}.pt")
    return model
