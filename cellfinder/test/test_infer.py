
import glob
import numpy as np
import tifffile
import torch
from matplotlib import pyplot as plt

from cellfinder import infer, train

def run_test(model_file, rundir=False):
    tm = train.get_model(2)
    tm.load_state_dict(torch.load(model_file))
    # %%

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    gp = '/home/cjw/Code/xlearn/cellfinder/Data/ND2Images/*.tif'
    files = sorted(glob.glob(gp))
                    
    print(len(files))
    import time
    cnn = infer.predict(tm, size=(400, 400), probability=.95)
    rf = np.random.randint(0, len(files))
    x = tifffile.imread(files[rf])
    r, b = cnn(x)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(x) 
    plt.subplot(1,2,2)
    plt.imshow(r)    
    # for test in files:
    #     x = tifffile.imread(test)
    #     t1 = time.time()
    #     r, b = cnn(x)
    #     print(r.mean(), time.time() - t1)
    
    if rundir:
        infer.inferdir('Data/ND2Images/', 'Data/trained_model.pt', 'Data/SavedRes', '*.tif')

