import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.colors import LogNorm

def loadCutField(dir, fn):
    file_path = os.path.join(dir, fn)
    fields_cut = np.load(file_path)
    HMIfield_cut = fields_cut['HMIfield']
    SPfield_cut = fields_cut['SPfield']
    Txy_cut = fields_cut['Txy']
    return HMIfield_cut, SPfield_cut, Txy_cut

def interp_bicubic(field_old, h_shape_new, w_shape_new):
    h_shape_old, w_shape_old = field_old.shape
    h_arr_old = np.linspace(0, h_shape_old, h_shape_old)
    w_arr_old = np.linspace(0, w_shape_old, w_shape_old)

    h_arr_new = np.linspace(0, h_shape_old, h_shape_new)
    w_arr_new = np.linspace(0, w_shape_old, w_shape_new)

    interp_func = interpolate.interp2d(w_arr_old, h_arr_old, field_old, kind='cubic')
    field_interp = interp_func(w_arr_new, h_arr_new)
    
    return field_interp

def calc_cc(pred, gt):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    epsilon = 1e-8
    numerator = np.sum(pred_flat * gt_flat)
    denominator = np.sqrt(np.sum(pred_flat**2)) * np.sqrt(np.sum(gt_flat**2))
    cc = numerator / (denominator + epsilon)
    
    return cc

save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/dataset/saveData/'

## Loading field data
file_list = [fn for fn in os.listdir(save_dir) if fn.endswith('.npz')]
num_total = len(file_list)

fn = file_list[10]
# fn = random.choice(file_list)
REC_time = fn[:-4]
print(f'Estimating the coalignment error of file: {fn}')

HMIfield_cut, SPfield_cut, Txy_cut = loadCutField(save_dir, fn)

h_ups, w_ups = 1.56, 1.68
h_shape_lr, w_shape_lr = 200, 200
h_shape_hr, w_shape_hr = int(h_shape_lr * h_ups), int(w_shape_lr * w_ups)

SPfield_bicubic = interp_bicubic(SPfield_cut, h_shape_lr, w_shape_lr)
# HMIfield_bicubic = interp_bicubic(HMIfield_cut, h_shape_hr, w_shape_hr)

print('Finding x-shift and y-shift with maximum CC ...')
max_cc = -1
best_shift_w = 0
best_shift_h = 0

for shift_w in range(-10, 11):
    for shift_h in range(-10, 11):
        SP_bicubic_sub = SPfield_bicubic
        HMI_sub = HMIfield_cut
        
        if shift_w > 0:
            SP_bicubic_sub = SP_bicubic_sub[:, shift_w:]
            HMI_sub = HMI_sub[:, :-shift_w]
        elif shift_w < 0:
            SP_bicubic_sub = SP_bicubic_sub[:, :shift_w]
            HMI_sub = HMI_sub[:, -shift_w:]
        
        if shift_h > 0:
            SP_bicubic_sub = SP_bicubic_sub[shift_h:, :]
            HMI_sub = HMI_sub[:-shift_h, :]
        elif shift_h < 0:
            SP_bicubic_sub = SP_bicubic_sub[:shift_h, :]
            HMI_sub = HMI_sub[-shift_h:, :]
        
        sub_cc = calc_cc(SP_bicubic_sub, HMI_sub)
        
        if sub_cc > max_cc:
            max_cc = sub_cc
            best_shift_w = shift_w
            best_shift_h = shift_h

print(f'Best shift: shift_w={best_shift_w}, shift_h={best_shift_h}, with highest cc={max_cc:.6f}')

SPfield_bicubic_shift = SPfield_bicubic
HMIfield_shift = HMIfield_cut

if best_shift_w > 0:
    SPfield_bicubic_shift = SPfield_bicubic_shift[:, best_shift_w:]
    HMIfield_shift = HMIfield_shift[:, :-best_shift_w]
elif best_shift_w < 0:
    SPfield_bicubic_shift = SPfield_bicubic_shift[:, :-best_shift_w]
    HMIfield_shift = HMIfield_shift[:, -best_shift_w:]
    
if best_shift_h > 0:
    SPfield_bicubic_shift = SPfield_bicubic_shift[best_shift_h:, :]
    HMIfield_shift = HMIfield_shift[:-best_shift_h, :]
elif best_shift_h < 0:
    SPfield_bicubic_shift = SPfield_bicubic_shift[:best_shift_h, :]
    HMIfield_shift = HMIfield_shift[-best_shift_h:, :]
            
shift_info = f'x_shift={best_shift_w}\n y_shift={best_shift_h}'

def show_maps(field, title, cmap='bwr'):
    vabs = np.max(np.abs(field))
    plt.imshow(field/vabs, cmap=cmap, vmin=-0.6, vmax=0.6)
    # plt.colorbar()
    plt.title(title)

def show_correlation(pred, gt, xlabel, ylabel, bins=50, vmax=None):
    
    cc = calc_cc(pred, gt)
    
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    h, xedges, yedges, img = plt.hist2d(pred_flat, gt_flat, bins=bins, \
        cmap='jet', cmin=1, norm=LogNorm(clip=True))
    
    plt.colorbar()
    if vmax is not None:
        img.set_clim(vmax=vmax)
    
    min_val = min(np.min(pred_flat), np.min(gt_flat))
    max_val = max(np.max(pred_flat), np.max(gt_flat))
    plt.plot([min_val, max_val], [min_val, max_val], color='purple', label=f'CC={cc:.4f}')
    
    plt.legend()
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{xlabel} .vs. {ylabel}')

plt.figure(figsize=(14,8))

plt.subplot(231)
show_maps(HMIfield_cut, 'LR (HMI)')

plt.subplot(234)
show_maps(SPfield_cut, 'HR (SP)')

plt.subplot(232)
show_maps(SPfield_bicubic + HMIfield_cut, 'Superimpose HR_Bicubic on LR')

plt.subplot(235)
show_correlation(SPfield_bicubic, HMIfield_cut, 'HR_Bicubic', 'LR', bins=50, vmax=None)

# HMI_bicubic_shift = HMIfield_bicubic[:,:-shift]
# SP_shift = SPfield_cut[:,shift:]
# SPfield_bicubic_shift = SPfield_bicubic[:,shift:]
# HMI_shift = HMIfield_cut[:,:-shift]

plt.subplot(233)
show_maps(SPfield_bicubic_shift + HMIfield_shift, 'Superimpose HR_Bicubic_shift on LR')

plt.subplot(236)
show_correlation(SPfield_bicubic_shift, HMIfield_shift, 'HR_Bicubic_shift', 'LR', bins=50, vmax=None)
plt.text(0.95, 0.05, shift_info, transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=10)

plt.suptitle(REC_time)

plt.show()

db