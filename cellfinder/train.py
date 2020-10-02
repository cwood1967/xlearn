import os

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
    
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer,
                                                      num_classes)
    
    return model

    
def collate(batch):
    return tuple(zip(*batch))

def main(root='Data', image_dir='Images', mask_dir='Masks',
         epochs=50, cropsize=(400, 400)):
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
        data, batch_size=8, shuffle=True, num_workers=4,
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
    print('Training...')
    for i in range(num_epochs):
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
           
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            #if i % (num_epochs//100) == 0: 
            print("**", i, losses)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            #lr_sched.step()
    print(loss_dict)
    return model


