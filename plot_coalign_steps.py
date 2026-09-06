import os
import numpy as np
import matplotlib.pyplot as plt

from constants import h_ups, w_ups, h_HMI, w_HMI
from utils_load_data import load_L72_sample

L7_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/dataset/L7/saveData/'
L72_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/dataset/L72/saveData/'

time_rec = '20110106_200005'
data_fn = time_rec + '.npz'

L7_path  = os.path.join(L7_dir, data_fn)
L72_path = os.path.join(L72_dir, data_fn)

HMIfield_L7,  SPfield_L7,  Txy_L7  = load_L72_sample(L7_path)
HMIfield_L72, SPfield_L72, Txy_L72 = load_L72_sample(L72_path)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

vmin, vmax = -300, 300

im1 = axes[0].imshow(HMIfield_L7, cmap='bwr', vmin=vmin, vmax=vmax)
axes[0].axvline(x=90, color='k', linewidth=0.5)
axes[0].axhline(y=130, color='k', linewidth=0.5)
axes[0].set_title('HMI field')

axes[1].imshow(SPfield_L7, cmap='bwr', vmin=vmin, vmax=vmax)
axes[1].axvline(x=90 * w_ups, color='k', linewidth=0.5)
axes[1].axhline(y=130 * h_ups, color='k', linewidth=0.5)
axes[1].set_title('Coarse-aligned SP field')

axes[2].imshow(SPfield_L72, cmap='bwr', vmin=vmin, vmax=vmax)
axes[2].axvline(x=90 * w_ups, color='k', linewidth=0.5)
axes[2].axhline(y=130 * h_ups, color='k', linewidth=0.5)
axes[2].set_title('Fine-aligned SP field')

fig.colorbar(im1, ax=axes, orientation='horizontal', shrink=0.8, aspect=60, extend='both')

plt.suptitle(time_rec, fontsize=14)
plt.show()

db