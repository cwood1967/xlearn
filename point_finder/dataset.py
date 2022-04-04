
import os
import glob

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import tifffile

class kp_dataset(Dataset):
    
    def __init__(self, rootdir, image_dir, labels_dir, keypoints_dir,
                 transforms):
       
        self.rootdir = rootdir
        self.image_dir = image_dir
        self.labels_file = labels_dir
        self.keypoints_file = keypoints_dir

        self.transforms = transforms

        image_path = os.path.join(rootdir, image_dir)
        kp_path = os.path.join(rootdir, keypoints_dir)
        labels_path = os.path.join(rootdir, labels_dir)

        self.image_files = glob.glob(f"{image_path}/*.tif")
        self.keypoint_files = glob.glob(f"{kp_path}/*_kp.npy")
        self.label_files = glob.glob(f"{labels_path}/*_label.npy")
        print(self.label_files)
    
    def __getitem__(self, idx):
        
        image_path = self.image_files[idx]
        print(image_path)
        kp_path = self.keypoint_files[idx]
        print(kp_path)
        labels_path = self.label_files[idx]
       
        _image = tifffile.imread(image_path)
        if len(_image.shape) == 2:
            a_image = np.expand_dims(_image, -1)

        ''' Get the min and max of each channel. The channel axis is -1 [2]'''
        amin = _image.min(axis=(0, 1), keepdims=True)
        amax = _image.max(axis=(0, 1), keepdims=True)

        ''' Normalize the image from 0 to 1'''
        _image = (_image - amin)/(amax - amin)


        _bbs = np.load(kp_path)
        _labels = np.load(labels_path)

        print(_bbs.shape)
        image, bbs = self.transforms(_image, _bbs)
       
        print(image.shape, bbs.shape) 
        labels = torch.as_tensor(_labels)
        keypoints = self.boxes_to_kp(bbs)
        print(keypoints)
        keypoints = torch.as_tensor(keypoints)
 
        target = {'boxes':bbs,
                  'labels': labels,
                  'keypoints':keypoints}


        return image, target

    def boxes_to_kp(self, bbs):

        keypoints = list()
        for bb in bbs:
            kp = bb.mean(axis=(-2, -1))#.astype(np.int32)
            keypoints.append(kp)
        return keypoints 
        
        

