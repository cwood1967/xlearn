
# %%
from dataset import PombeDataset
import transforms
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

root = '/home/cjw/Code/xlearn/Data'
image_dir = 'Ibig'
mask_dir = 'Mbig'

t = transforms.get_transforms()
p = PombeDataset(root, image_dir, mask_dir, t)
v = p.__getitem__(3)

# %%
vn = v[0].numpy()
print(vn.shape)
vn= np.moveaxis(v[0].numpy(), 0, -1)

# %%


fig, ax = plt.subplots(1)
ax.imshow(vn[:,:,0])
for b in v[1]['boxes']:
    print(b)
    rx, ry, rxf, ryf = b
    w = rxf - rx
    h = ryf - ry
    rp = patches.Rectangle((rx, ry), w, h, edgecolor='r', facecolor='none')
    ax.add_patch(rp)

# %%
mask = v[1]['masks'].numpy().max(axis=0)
plt.imshow(mask)

# %%
vn.min()

# %%
