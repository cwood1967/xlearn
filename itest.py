# %%
import autoreload
%load_ext autoreload
%autoreload 2

# %%
from cellfinder.test import test_infer
# %%
r = test_infer.run_test("/home/cjw/Code/xlearn/cellfinder/Data/trained_model_20200930.pt")
# %%
from matplotlib import pyplot as plt
plt.imshow(r > .8)
# %%
