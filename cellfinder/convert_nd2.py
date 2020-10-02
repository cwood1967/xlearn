# %%
import glob
import os

import numpy as np
import tifffile
from nd2reader import ND2Reader
from matplotlib import pyplot as plt

def nd2_reader_slice(path, islice, bundle='tcyx'):
    ndx = ND2Reader(path)
    name = os.path.basename(path)[:-4]
    sizes = ndx.sizes
    
    if 't' not in sizes:
        sizes['t'] = 1
    if 'z' not in sizes:
        sizes['z'] = 1
    if 'c' not in sizes:
        sizes['c'] = 1

    ndx.bundle_axes = 'tyx'
    ndx.iter_axes = 'zc'
    n = len(ndx)

    shape = (sizes['t'], sizes['z'], sizes['c'], sizes['y'], sizes['x'])
    #image  = np.zeros(shape, dtype=np.float32)
    image = ndx.get_frame(islice)

    return image.squeeze(), sizes

def nd2_tiff(filename, tiff_name, xslice):
    z, sizes = nd2_reader_slice(filename, xslice)
    tifffile.imwrite(tiff_name, z.astype(np.float32))

# %%
filenames = glob.glob('/n/core/micro/jeg/ac1692/jjl/20200729_3po_IMARE-99546/20200729_3PO_lexy_starved/*.nd2')
print(filenames)
# %%
for filename in filenames:
    bn = os.path.basename(filename)
    tiff_name = 'Data/ND2Images/' + bn[:-3] + 'tif'
    print(tiff_name)
    nd2_tiff(filename, tiff_name, 32)


    

# %%


# %%
