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
    Txy_crop = fields['Txy_crop']
    
    return HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop

def plotField(target, fn, HMIfield_crop, SPfield_interp, RecTime, vmin=-3000, vmax=3000):
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'hspace': 0.2, 'bottom': 0.15})
    
    im1 = ax1.pcolormesh(HMIfield_crop, cmap='bwr', vmin=vmin, vmax=vmax)
    ax1.axis('equal')
    ax1.set_title(f'HMIfield (H,W)={HMIfield_crop.shape}')
    
    im2 = ax2.pcolormesh(SPfield_interp, cmap='bwr', vmin=vmin, vmax=vmax)
    ax2.axis('equal')
    ax2.set_title(f'SPfield (H,W)={SPfield_interp.shape}')
    
    fig.suptitle(RecTime)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    
    plt.savefig(os.path.join(target+'SaveFig/', fn), bbox_inches='tight', dpi=150)
    plt.close()

def saveField(target, fn, HMIfield_cut, SPfield_cut, Txy_cut):
    np.savez(os.path.join(target+'SaveData/', fn), 
            HMIfield_cut = HMIfield_cut, 
            SPfield_cut = SPfield_cut, 
            Txy_cut = Txy_cut)
    
def saveField(target, fn, HMIfield_crop, SPfield_interp, Txy_crop):
    np.savez(os.path.join(target+'SaveData/', fn), 
            HMIfield = HMIfield_crop, 
            SPfield = SPfield_interp, 
            Txy = Txy_crop)

data_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/coalignment/fast_mode_valid/'
save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/dataset/'

# h_lr_std = 220
# w_lr_std = 240
# h_ups = 1.7   # h_hr_std = 374
# w_ups = 1.575 # w_hr_std = 378

h_lr_std = 200
w_lr_std = 200
h_ups = 1.56 # h_hr_std = 312
w_ups = 1.68 # h_hr_std = 336

h_hr_std = int(h_lr_std * h_ups)
w_hr_std = int(w_lr_std * w_ups)
save_or_not = 1

error_report = []
file_list = [fn for fn in os.listdir(data_dir) if fn.endswith('.npz')]
num_total = len(file_list)
    
print(f'Begin processing, {num_total} files in total')
print('-' * 50)

for fn in file_list:
    if fn.endswith('.npz'):
        try:
            REC_time = fn[:-4]
            save_fn = os.path.join(save_dir, 'SaveData/', REC_time + '.npz')
            
            if os.path.exists(save_fn):
                print(f'{fn} has been processed, skip it')
                continue
            
            HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop = loadField(data_dir, fn)
            
            ####################################################################################################
            # Dataset 1: HMI-crop-field
            # Dataset 2: HMI-coalign-field
            # Dataset 3: SP-field
            ## Above datasets can be directly imported from './coalignment/'
            
            ####################################################################################################
            # Dataset 4: HMI-cut-field, cut to match the fixed image shape
            h_HMI, w_HMI = HMIfield_crop.shape
            
            if h_HMI < h_lr_std or w_HMI < w_lr_std:
                continue
            
            h_beg_HMI = (h_HMI - h_lr_std) // 2
            h_end_HMI = h_beg_HMI + h_lr_std
            w_beg_HMI = (w_HMI - w_lr_std) // 2
            w_end_HMI = w_beg_HMI + w_lr_std
            
            HMIfield_cut = HMIfield_crop[h_beg_HMI:h_end_HMI, w_beg_HMI:w_end_HMI]
            Txy_cut = Txy_crop[h_beg_HMI:h_end_HMI, w_beg_HMI:w_end_HMI]
            
            ####################################################################################################
            # Dataset 5: SP-cut-field, imterpolate to h_hr_std*w_hr_std to match the fixed upscale factor
            h_SP, w_SP = SPfield_coalign.shape
            
            if h_SP < h_hr_std or w_SP < w_hr_std:
                continue
            
            h_beg_frac = h_beg_HMI / h_HMI
            h_end_frac = h_end_HMI / h_HMI
            w_beg_frac = w_beg_HMI / w_HMI
            w_end_frac = w_end_HMI / w_HMI
            
            h_beg_SP = h_SP * h_beg_frac
            h_end_SP = h_SP * h_end_frac
            w_beg_SP = w_SP * w_beg_frac
            w_end_SP = w_SP * w_end_frac
            
            # Constructing grids
            h_SP_arr = np.linspace(0, h_SP, h_SP)
            w_SP_arr = np.linspace(0, w_SP, w_SP)
            hh_SP, ww_SP = np.meshgrid(h_SP_arr, w_SP_arr, indexing='ij')
            
            h_SP_interp_arr = np.linspace(h_beg_SP, h_end_SP, h_hr_std)
            w_SP_interp_arr = np.linspace(w_beg_SP, w_end_SP, w_hr_std)
            hh_SP_interp, ww_SP_interp = np.meshgrid(h_SP_interp_arr, w_SP_interp_arr, indexing='ij')
            
            # Interpolating
            points = np.column_stack((hh_SP.flatten(), ww_SP.flatten()))
            values = SPfield_coalign.flatten()
            SPfield_cut = griddata(points, values, (hh_SP_interp, ww_SP_interp), method='linear')
            
            ####################################################################################################
            # Saving datasets
            if save_or_not:
                plotField(save_dir, REC_time + '.png', HMIfield_cut, SPfield_cut, REC_time, vmin=-3000, vmax=3000)
                saveField(save_dir, REC_time + '.npz', HMIfield_cut, SPfield_cut, Txy_cut)
                print(f'Successfully processed file: {fn}')
            
        except Exception as e:
            error_report.append({
                'filename': fn,
                'error': str(e)
            })
            print(f'Error occurs when processing {fn}: {e}. skip to the next file')
            continue
        
print('-' * 50)
print(f'total files: {num_total}')
print(f'processing failed: {len(error_report)}')

if error_report:
    print('\nError report:')
    for idx, err in enumerate(error_report, 1):
        print(f"{idx}. file: {err['filename']} | error: {err['error']}")
    
    error_log = os.path.join(save_dir, 'error_report.txt')
    with open(error_log, 'w', encoding='utf-8') as f:
        for err in error_report:
            f.write(f"file: {err['filename']} | error: {err['error']}\n")
    print(f'\nError report saved to: {error_log}')
# db