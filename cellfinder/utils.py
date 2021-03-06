from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.patches as patches

def image_to_patches(image : np.array, size : Tuple =(256, 256), shift=False):
    ndim = len(image.shape)
    sizeX = image.shape[-1]
    sizeY = image.shape[-2]
    
    amin = image.min(axis=(-2, -1), keepdims=True)
    amax = image.max(axis=(-2, -1), keepdims=True)
    image = (image - amin)/(amax - amin)
    
    patch_x = size[1]
    patch_y = size[0]
    ## start of assuming one t and z, possibly multiple channels
    ## and will put everything in a patch into network
    numx = sizeX//patch_x
    numy = sizeY/patch_y

    dx = patch_x//2
    dy = patch_y//2
        
    patch_dict = dict()
    for jy in np.arange(0, sizeY, dy):
        jyf = jy + patch_y
        if jyf > sizeY:
            jyf = sizeY
            jy = sizeY - patch_y
        for jx in np.arange(0, sizeX, dx):
            jxf = jx + patch_x
            if jxf > sizeX:
                jxf = sizeX
                jx = sizeX - patch_x
            patch_dict[(jy, jx)] = (image[..., jy:jyf, jx:jxf])
    
    if shift:
        for jy in np.arange(patch_y//2, sizeY, dy):
            jyf = jy + patch_y
            if jyf > sizeY:
                jyf = sizeY
                jy = sizeY - patch_y
            for jx in np.arange(patch_x//2, sizeX, dx):
                jxf = jx + patch_x
                if jxf > sizeX:
                    jxf = sizeX
                    jx = sizeX - patch_x
                patch_dict[(jy, jx)] = (image[..., jy:jyf, jx:jxf])
            
    return patch_dict

def reconstruct_from_dict(patch_dict, shape, patchsize, shift=False):
    recon = np.zeros(shape, dtype=np.float32)
    #print(patch_dict.keys())
    dx = patchsize[1]
    dy = patchsize[0]
    pad = np.zeros(patchsize, dtype=np.float32)
    pds = max(4, int(.01*dx))
    pad[pds:dy-pds, pds:dx-pds] = 1
    all_boxes = list()
    for k, vs in patch_dict.items():
        v = vs[0]
        boxes = vs[1]
        ky, kx = k
        v = v*pad
        rv = recon[..., ky:ky + dy, kx:kx + dx]
        recon[..., ky:ky + dy, kx:kx + dx] = np.where(v > rv, v, rv)
        
        if boxes:
            for box in boxes:
                box = box.copy()  
                try:
                    box[0] += kx
                    box[2] += kx
                    box[1] += ky
                    box[3] += ky
                except:
                    print(box)
                all_boxes.append(box)
    return recon, all_boxes
    

def getHighProb(predictions, prob=0.9):
    masks = predictions['masks']
    if len(masks) == 0:
        return [0], [0]
    scores = predictions['scores']
    boxes = predictions['boxes']
    res_list= list()
    box_list = list()
    res_list.append(0*masks[0])
    for i, s in enumerate(scores):
        if s >= prob:
            res_list.append(masks[i])
            box_list.append(boxes[i])

    return np.stack(res_list), box_list


def res_to_cpu(predictions):
    pcpu = list()
    for p in predictions:
        masks = p['masks'].cpu().detach().numpy()
        scores = p['scores'].cpu().detach().numpy()
        boxes = p['boxes'].cpu().detach().numpy()
        pcpu.append({'masks':masks, 'scores':scores, 'boxes':boxes})
        
    return pcpu
    
def tensors_to_cpu(tensor_list):
    tcpu = [t.cpu().detach().numpy() for t in tensor_list]
    return tcpu


def res_image(x, shape, prob=.75):
    """Return maximum projection of probabilities

    Parameters
    ----------
    x : dict
        predictions dictionary 
    shape : shape of the patch
        [description]
    prob : float, optional
        [description], by default .75

    Returns
    -------
    numpy array
        max projection of probabilites
    """
    r, boxes = getHighProb(x, prob=prob)
    if len(r) > 1:
        return r.max(axis=0), boxes 
    else:
        return np.zeros(shape, dtype=np.float32), None
    
    
def plot_image(recon, image, boxes, filename):
    
    fig, ax = plt.subplots(2,1, figsize=(8,16))
    ax[0].imshow(recon)
    print(len(boxes))
    try:
        for b in boxes:
            rx, ry, rxf, ryf = b
            w = rxf - rx
            h = ryf - ry
            rp = patches.Rectangle((rx, ry), w, h, edgecolor='r', facecolor='none')
            _ = ax[0].add_patch(rp)
    except:
        print("no boxes to show")
        
    ax[1].imshow(image.squeeze())
    plt.show()
    if filename is not None:
        fname = filename[:-3] + ".pdf"
        plt.savefig('Data/Results/' + fname)
        print('Saved', 'Data/Results/' + fname)