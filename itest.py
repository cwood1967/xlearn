# %%
import autoreload
%load_ext autoreload
%autoreload 2

# %%
import tifffile
from cellfinder.test import test_infer
# %%

imagefile = ('/n/core/micro/jeg/ac1692/NIkon-3PO/3-18-20/'
         '3-17-20_WT-Agar-ON_bestz_transmitted.tif')

image = tifffile.imread(imagefile)

# %%
modelfile = "/home/cjw/Code/xlearn/cellfinder/Data/trained_model_20200930.pt"
r = test_infer.run_test(modelfile, image[78,2], prob=.5)
# %%
from matplotlib import pyplot as plt
r[1600,:] = 1
r[1644, :] = 1
r[:,1600] = 1
r[:,1644] = 1
plt.figure(figsize=(12,12))
plt.imshow(r) #> .8)
# %%
plt.figure(figsize=(12,12))
plt.imshow(image[78,2])
# %%
r.max()
# %%
