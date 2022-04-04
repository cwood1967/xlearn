import os
import logging

import numpy as np
import tifffile
import torch

from cellfinder import infer, train
from cjwutils.simr_io import nd2read
from cjwutils.misc import simrpathutils

models = {"Pombe":'models/trained_model_20210326_ascis.pt',
          "Spores":'models/trained_model_20210331_spores.pt',
          }

def makeCNN(modelfile, size=400, probability=.9):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 

    model = train.get_model(2)
    model.load_state_dict(torch.load(modelfile))
    cnn = infer.predict(model, device=device,
                        size=(size, size),
                        probability=probability)
    return cnn

def open_image(filename, virtual=False):
    fileformat = filename.split('.')[-1].lower()
    if fileformat in ['tif', 'tiff']:
        _x = tifffile.imread(filename)
        return tifffile.imread(filename), len(_x.shape), _x.shape 
   
    if fileformat.lower() == 'nd2':
        x = nd2read.nikonImage(filename)
        x.nd2.bundle_axes = "yx"
        x.nd2.iter_axes = "tzc"
        channel = 2
        zconf = x.find_z_config(channel=channel)
        print(zconf)
        stack = x.frame_generator(range(*zconf))
        shape = (x.sizes['t'], x.sizes['y'], x.sizes['x'])
        return (stack, 5, shape)
    
def inferimage(filename, savepath, model="Pombe"):
    modelfile = models[model]
    print(model, modelfile)
    cnn = makeCNN(modelfile, probability=.4)
    print(filename)
    #image = tifffile.imread(filename)
    imageformat = filename.split('.')[-1].lower()
    image, ndims, shape = open_image(filename)
    fname = os.path.basename(filename)
    lastdot = fname.rindex('.')
    sname = fname[:lastdot] + "_pred.tif"
    saveto = savepath + "/" + sname
    logging.info(f"Working on {filename} with model {model}")
    if ndims > 2:
        try:
            tifffile.imwrite(saveto, inferstack(cnn, image, shape),  imagej=True)
            logging.info(f"Prediction saved to {saveto}")
        except Exception as e:
            print(e)
            logging.warning(f"Trouble saving to {saveto}")
    else:
        tifffile.imwrite(saveto, inferslice(cnn, image), imagej=True)

def inferstack(cnn, stack, shape):
    res = np.zeros(shape, dtype=np.float32)
    for i, z in enumerate(stack):
        logging.info(f"working on z slice {i}")
        res[i] = inferslice(cnn, z)
    return res
    

def inferslice(cnn, x):
    with torch.no_grad():
        _res, _ = cnn(x)
    res = _res.astype(np.float32)
    del _res
    torch.cuda.empty_cache()
    return res.astype(np.float32)

def easy():
    x = np.random.randn(int(1e5))
    print(x.mean())
