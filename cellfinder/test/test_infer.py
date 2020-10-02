# %%
from math import inf
import autoreload
%load_ext autoreload
%autoreload 2
# %%

import glob
import tifffile
import torch

import infer
import train

# %%
tm = train.get_model(2)
tm.load_state_dict(torch.load("Data/trained_model_20200930.pt"))
# %%

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
files = sorted(glob.glob('Data/ND2Images/*.tif'))
len(files)
                
# %%
import time
cnn = infer.predict(tm, size=(400, 400), probability=.95)
for test in files:
    x = tifffile.imread(test)
    t1 = time.time()
    r, b = cnn(x)
    print(r.mean(), time.time() - t1)
# %%
infer.inferdir('Data/ND2Images/', 'Data/trained_model.pt', 'Data/SavedRes', '*.tif')
# %%
r.shape
# %%
