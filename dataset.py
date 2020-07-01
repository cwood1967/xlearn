import os

import torch
import torchvision
import numpy as np
#from PIL import Image
from skimage.io import imread
from scipy.ndimage import measurements

class PombeDataset(object):

    def __init__(self, rootdir, image_dir, mask_dir,transforms):
        self.rootdir = rootdir
        self.transforms = transforms
        image_path = os.path.join(rootdir, image_dir)
        mask_path = os.path.join(rootdir, mask_dir)
        images = list(sorted(os.listdir(image_path)))
        self.remove_nonimage(images)
        self.images = [os.path.join(rootdir, image_dir, j) for j in images]

        masks = list(sorted(os.listdir(mask_path)))
        self.remove_nonimage(masks)
        self.masks = [os.path.join(rootdir, mask_dir, j) for j in masks]

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = imread(image_path)
        mask = imread(mask_path)
        
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        _mask = mask.numpy()
        mask_labels, nobjects = self.label_mask(_mask)
        oid = np.unique(mask_labels)[1:]
        sep_masks = mask_labels == oid[:, None, None]
        sep_masks = torch.as_tensor(sep_masks, dtype=torch.uint8)
        boxes = self.get_boxes(mask_labels, nobjects)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((nobjects,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:,1])*(boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((nobjects,), dtype=torch.int64)

        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = sep_masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)
        return image, target

    def label_mask(self, mask):
        labels, nlabels = measurements.label(mask)
        return labels, nlabels
    
    def remove_nonimage(self, pathlist):
        ew = ('png', 'tif', 'tiff', 'jpg', 'jpeg')
        for j in pathlist:
            if not j.endswith(ew):
                pathlist.remove(j)

    def get_boxes(self, mask_labels, nobjects):

        boxes = list()
        for i in range(nobjects):
            pos = np.where(mask_labels == i + 1)
            xmin = pos[1].min()
            xmax = pos[1].max()
            ymin = pos[0].min()
            ymax = pos[0].max()
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes

    def __len__(self):
        return len(self.images)
    
    def napari_boxes(self, b):
        nbox = np.array(
                        [[b[1], b[0]],
                        [b[3], b[0]],
                        [b[3], b[2]],
                        [b[1], b[2]]]
        )
        return nbox