import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import random

def loadCoalignField(dir, fn):
    file_path = os.path.join(dir, fn)
    fields = np.load(file_path)
    HMIfield_crop = fields['HMIfield_crop']
    HMIfield_coalign = fields['HMIfield_coalign']
    SPfield_coalign = fields['SPfield_coalign']
    Txy_crop = fields['Txy_crop']
    
    return HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop

def loadCutField(dir, fn):
    file_path = os.path.join(dir, fn)
    fields_cut = np.load(file_path)
    HMIfield_cut = fields_cut['HMIfield']
    SPfield_cut = fields_cut['SPfield']
    Txy_cut = fields_cut['Txy']
    return HMIfield_cut, SPfield_cut, Txy_cut

def checkFieldCoalign(HMIfield_crop, SPfield_coalign, HMIfield_cut, SPfield_cut, RecTime):
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 8), gridspec_kw={'hspace': 0.2, 'bottom': 0.15})
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    
    def plotField(ax, field, title, xlim=None, ylim=None):
        vabs = np.max(np.abs(field)) * 0.3
        ax.pcolormesh(field, cmap='bwr', vmin=-vabs, vmax=vabs)
        # ax.axis('equal')
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.set_title(title)
    
    plotField(ax1, HMIfield_crop, f'HMIfield (H,W)={HMIfield_crop.shape}')
    plotField(ax4, SPfield_coalign, f'SPfield (H,W)={SPfield_coalign.shape}')
    
    h_HMI_show, w_HMI_show = round(h_end_HMI - h_beg_HMI, 1), round(w_end_HMI - w_beg_HMI, 1)
    plotField(ax2, HMIfield_crop, f'HMIcut (H,W)=({h_HMI_show},{w_HMI_show})', 
              xlim=[w_beg_HMI, w_end_HMI], ylim=[h_beg_HMI, h_end_HMI])
    h_SP_show, w_SP_show = round(h_end_SP - h_beg_SP, 1), round(w_end_SP - w_beg_SP, 1)
    plotField(ax5, SPfield_coalign, f'SPcut (H,W)=({h_SP_show},{w_SP_show})', 
              xlim=[w_beg_SP, w_end_SP], ylim=[h_beg_SP, h_end_SP])
    
    plotField(ax3, HMIfield_cut, f'HMIcut (H,W)=({h_lr_std},{w_lr_std})')
    plotField(ax6, SPfield_cut, f'SPinterp (H,W)=({h_hr_std},{w_hr_std})')
    
    fig.suptitle(RecTime)
        
    plt.show()

data_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/coalignment/fast_mode_valid/'
save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/dataset/saveData/'

## Loading field data
file_list = [fn for fn in os.listdir(save_dir) if fn.endswith('.npz')]
num_total = len(file_list)

fn = file_list[1000]
# fn = random.choice(file_list)
REC_time = fn[:-4]
print(f'Checking the coalignment of file: {fn}')

HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop = loadCoalignField(data_dir, fn)
HMIfield_cut, SPfield_cut, Txy_cut = loadCutField(save_dir, fn)

## Cutting region
h_lr_std = 200
w_lr_std = 200
h_ups = 1.56 # h_hr_std = 312
w_ups = 1.68 # h_hr_std = 336
h_hr_std = int(h_lr_std * h_ups)
w_hr_std = int(w_lr_std * w_ups)

h_HMI, w_HMI = HMIfield_crop.shape
h_SP, w_SP = SPfield_coalign.shape

h_beg_HMI = (h_HMI - h_lr_std) // 2
h_end_HMI = h_beg_HMI + h_lr_std
w_beg_HMI = (w_HMI - w_lr_std) // 2
w_end_HMI = w_beg_HMI + w_lr_std

h_beg_frac = h_beg_HMI / h_HMI
h_end_frac = h_end_HMI / h_HMI
w_beg_frac = w_beg_HMI / w_HMI
w_end_frac = w_end_HMI / w_HMI

h_beg_SP = h_SP * h_beg_frac
h_end_SP = h_SP * h_end_frac
w_beg_SP = w_SP * w_beg_frac
w_end_SP = w_SP * w_end_frac

## Checking coalign steps
checkFieldCoalign(HMIfield_crop, SPfield_coalign, HMIfield_cut, SPfield_cut, REC_time)

db