import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def loadField(target, fn):
    file_path = os.path.join(target, fn)
    fields = np.load(file_path)
    HMIfield_crop = fields['HMIfield_crop']
    HMIfield_coalign = fields['HMIfield_coalign']
    SPfield_coalign = fields['SPfield_coalign']
    
    fields.close()
    
    return HMIfield_crop, HMIfield_coalign, SPfield_coalign

def plotField(target, fn, HMIfield_cut, SPfield_cut, RecTime, vmin=-3000, vmax=3000):
    plt.figure(figsize=(10,5))
    
    plt.subplot(121)
    plt.pcolormesh(HMIfield_cut, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.axis('equal')
    plt.title(f'HMIfield_cut {HMIfield_cut.shape}')
    
    plt.subplot(122)
    plt.pcolormesh(SPfield_cut, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.axis('equal')
    plt.title(f'SPfield_cut {SPfield_cut.shape}')
    
    plt.suptitle(RecTime)
    plt.savefig(os.path.join(target+'SaveFig/', fn))
    plt.close()

def saveField(target, fn, HMIfield_cut, SPfield_cut):
    np.savez(os.path.join(target+'SaveData/', fn), 
            HMIfield_cut = HMIfield_cut, 
            SPfield_cut = SPfield_cut)

data_dir = 'E:/Research/Work/Super_Resolution_Magnetogram/coalignment/SaveData/'
save_dir = 'E:/Research/Work/Super_Resolution_Magnetogram/dataset/'
x_HMI_std = 220
y_HMI_std = 240
x_upscale = 1.7   # x_SP_std = 646
y_upscale = 1.575 # y_SP_std = 378
x_SP_std = int(x_HMI_std * x_upscale)
y_SP_std = int(y_HMI_std * y_upscale)
save_or_not = 1

for fn in os.listdir(data_dir):
    if fn.endswith('.npz'):
        HMIfield_crop, HMIfield_coalign, SPfield_coalign = loadField(data_dir, fn)
        REC_time = fn[:-4]
        
        ####################################################################################################
        # Dataset 1: HMI-crop-field
        # Dataset 2: HMI-coalign-field
        # Dataset 3: SP-field
        ## Above datasets can be directly imported from './coalignment/SaveData/'
        
        ####################################################################################################
        # Dataset 4: HMI-cut-field, cut to match the fixed image shape
        x_HMI, y_HMI = HMIfield_crop.shape
        
        if x_HMI < x_HMI_std or y_HMI < y_HMI_std:
            continue
        
        x_beg_HMI = (x_HMI - x_HMI_std) // 2
        x_end_HMI = x_beg_HMI + x_HMI_std
        y_beg_HMI = (y_HMI - y_HMI_std) // 2
        y_end_HMI = y_beg_HMI + y_HMI_std
        
        HMIfield_cut = HMIfield_crop[x_beg_HMI:x_end_HMI, y_beg_HMI:y_end_HMI]
        
        ####################################################################################################
        # Dataset 5: SP-cut-field, imterpolate to x_SP_std*y_SP_std to match the fixed upscale factor
        x_SP, y_SP = SPfield_coalign.shape
        
        if x_SP < x_SP_std or y_SP < y_SP_std:
            continue
        
        x_beg_frac = x_beg_HMI / x_HMI
        x_end_frac = x_end_HMI / x_HMI
        y_beg_frac = y_beg_HMI / y_HMI
        y_end_frac = y_end_HMI / y_HMI
        
        x_beg_SP = x_SP * x_beg_frac
        x_end_SP = x_SP * x_end_frac
        y_beg_SP = y_SP * y_beg_frac
        y_end_SP = y_SP * y_end_frac
        
        # Constructing grids
        x_SP_arr = np.linspace(0, x_SP, x_SP)
        y_SP_arr = np.linspace(0, y_SP, y_SP)
        xx_SP, yy_SP = np.meshgrid(x_SP_arr, y_SP_arr, indexing='ij')
        
        x_SP_interp_arr = np.linspace(x_beg_SP, x_end_SP, x_SP_std)
        y_SP_interp_arr = np.linspace(y_beg_SP, y_end_SP, y_SP_std)
        xx_SP_interp, yy_SP_interp = np.meshgrid(x_SP_interp_arr, y_SP_interp_arr, indexing='ij')
        
        # Interpolating
        points = np.column_stack((xx_SP.flatten(), yy_SP.flatten()))
        values = SPfield_coalign.flatten()
        SPfield_cut = griddata(points, values, (xx_SP_interp, yy_SP_interp), method='linear')
        
        ####################################################################################################
        # x_ratio_beg = x_beg_HMI / x_HMI
        # x_ratio_end = x_end_HMI / x_HMI
        # y_ratio_beg = y_beg_HMI / y_HMI
        # y_ratio_end = y_end_HMI / y_HMI
        
        # x_beg_SP = int(round(x_ratio_beg * x_SP))
        # x_end_SP = int(round(x_ratio_end * x_SP))
        # y_beg_SP = int(round(y_ratio_beg * y_SP))
        # y_end_SP = int(round(y_ratio_end * y_SP))
        
        # SPfield_cut = SPfield_coalign[x_beg_SP:x_end_SP, y_beg_SP:y_end_SP]
        
        # x_SP_cut, y_SP_cut = SPfield_cut.shape
        
        # x_SP_std_int = int(round(x_SP_std))
        # y_SP_std_int = int(round(y_SP_std))
        
        # x_SP_cut_arr = np.linspace(0, 1, x_SP_cut)
        # y_SP_cut_arr = np.linspace(0, 1, y_SP_cut)
        # xx_SP_cut, yy_SP_cut = np.meshgrid(x_SP_cut_arr, y_SP_cut_arr, indexing='ij')
        
        # x_SP_interp_arr = np.linspace(0, 1, x_SP_std_int)
        # y_SP_interp_arr = np.linspace(0, 1, y_SP_std_int)
        # xx_SP_interp, yy_SP_interp = np.meshgrid(x_SP_interp_arr, y_SP_interp_arr, indexing='ij')
        
        # points = np.column_stack((xx_SP_cut.flatten(), yy_SP_cut.flatten()))
        # values = SPfield_cut.flatten()
        # SPfield_cut = griddata(points, values, (xx_SP_interp, yy_SP_interp), method='linear')
        
        ####################################################################################################
        # Saving datasets
        if save_or_not:
            plotField(save_dir, REC_time + '.png', HMIfield_cut, SPfield_cut, REC_time, vmin=-3000, vmax=3000)
            saveField(save_dir, REC_time + '.npz', HMIfield_cut, SPfield_cut)
        
        # db