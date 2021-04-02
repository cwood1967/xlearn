
import glob
import numpy as np
import tifffile
import torch
from matplotlib import pyplot as plt

from cellfinder import infer, train

def run_test(model_file, x, prob=.95, rundir=False):
    tm = train.get_model(2)
    tm.load_state_dict(torch.load(model_file))
    # %%

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    import time
    cnn = infer.predict(tm, size=(400, 400), probability=prob)
    r, b = cnn(x)
    return r 
    # for test in files:
    #     x = tifffile.imread(test)
    #     t1 = time.time()
    #     r, b = cnn(x)
    #     print(r.mean(), time.time() - t1)
    
    if rundir:
        infer.inferdir('Data/ND2Images/', 'Data/trained_model.pt', 'Data/SavedRes', '*.tif')

