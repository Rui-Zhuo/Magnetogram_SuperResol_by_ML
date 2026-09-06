import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

def loadField(file_path):
    fields = np.load(file_path)
    HMIfield_crop = fields['HMIfield_crop']
    HMIfield_coalign = fields['HMIfield_coalign']
    SPfield_coalign = fields['SPfield_coalign']
    Txy_crop = fields['Txy_crop']
    
    return HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop

def Str2Dtime(REC_time, format_str='%Y%m%d_%H%M%S'):
    dt = datetime.strptime(REC_time, format_str)
    return dt

data_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/coalignment/'
save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/statistics/'
save_or_not = 0

####################################################################################################
# Iterating to show the image shape
h_lr_lst = []
w_lr_lst = []
h_hr_lst = []
w_hr_lst = []
dtime_lst = []

for root, dirs, files in os.walk(data_dir):
    print(f'Begin processing: {root}')
    for fn in files:
        if fn.endswith('.npz'):
            file_path = os.path.join(root, fn)
                
            HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop = loadField(file_path)
            REC_time = fn[:-4]
            
            h_lr, w_lr = HMIfield_crop.shape
            h_hr, w_hr = SPfield_coalign.shape
            dtime = Str2Dtime(REC_time)
            
            h_lr_lst.append(h_lr)
            w_lr_lst.append(w_lr)
            h_hr_lst.append(h_hr)
            w_hr_lst.append(w_hr)
            dtime_lst.append(dtime)

h_lr_arr, w_lr_arr = np.array(h_lr_lst), np.array(w_lr_lst)
h_hr_arr, w_hr_arr = np.array(h_hr_lst), np.array(w_hr_lst)
dtime_arr = np.array(dtime_lst)

h_ups = h_hr_arr / h_lr_arr
w_ups = w_hr_arr / w_lr_arr

print('Begin counting samples of both modes ...')

num_norm = np.sum(h_ups >= 2.5)
num_fast = np.sum(h_ups < 2.5)
ind_norm = np.where(h_ups >= 2.5)
ind_fast = np.where(h_ups < 2.5)

h_lr_norm, h_lr_fast = h_lr_arr[ind_norm], h_lr_arr[ind_fast]
w_lr_norm, w_lr_fast = w_lr_arr[ind_norm], w_lr_arr[ind_fast]
h_hr_norm, h_hr_fast = h_hr_arr[ind_norm], h_hr_arr[ind_fast]
w_hr_norm, w_hr_fast = w_hr_arr[ind_norm], w_hr_arr[ind_fast]
h_ups_norm, h_ups_fast = h_ups[ind_norm], h_ups[ind_fast]
w_ups_norm, w_ups_fast = w_ups[ind_norm], w_ups[ind_fast]

print('Begin counting samples of available files ...')

# h_lr_std = 220
# w_lr_std = 240
# h_ups_std = 1.575
# w_ups_std = 1.7

h_lr_std = 200
w_lr_std = 200
h_ups_std = 1.56
w_ups_std = 1.68

h_hr_std = h_lr_std * h_ups_std
w_hr_std = w_lr_std * w_ups_std

avail_mask = (h_lr_fast >= h_lr_std) & \
             (w_lr_fast >= w_lr_std) & \
             (h_hr_fast >= h_hr_std) & \
             (w_hr_fast >= w_hr_std)

num_avail = np.sum(avail_mask)

####################################################################################################
# Plotting the shape of images
size = 5
linewidth = 2
fontsize = 10

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(w_lr_fast, h_lr_fast, s=size, c='r', label='lr (HMI)')
ax1.scatter(w_hr_fast, h_hr_fast, s=size, c='b', label='hr (SP)')
ax1.axvline(x=w_lr_std, color='r', label=f'w_lr_std={w_lr_std}')
ax1.axvline(x=w_hr_std, color='b', label=f'w_lr_std={w_hr_std}')
ax1.axhline(y=h_lr_std, color='r', linestyle='--', label=f'h_lr_std={h_lr_std}')
ax1.axhline(y=h_hr_std, color='b', linestyle='--', label=f'h_lr_std={h_hr_std}')
ax1.legend()
ax1.set_xlabel('Width')
ax1.set_ylabel('Height')
ax1.set_aspect('equal')
ax1.set_xlim([0, 1100])
ax1.set_ylim([0, 1100])
ax1.grid()

hist = ax2.hist2d(w_ups_fast, h_ups_fast, range=[[1.65,1.75], [1.54,1.64]], bins=[50, 50], cmap='jet')
fig.colorbar(hist[3], ax=ax2)
ax2.axvline(x=w_ups_std, color='w', linestyle='--', label='std_upscale')
ax2.axhline(y=h_ups_std, color='w', linestyle='--', label=f'num_avail={num_avail}/{num_fast}')
ax2.legend()
ax2.set_xlabel('W upscale factor')
ax2.set_ylabel('H upscale factor')
ax2.set_aspect('equal')
ax2.grid()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

ax1.scatter(dtime_lst, w_lr_arr, s=size, c='r', label='lr (HMI)')
ax1.scatter(dtime_lst, w_hr_arr, s=size, c='b', label='hr (SP)')
ax1.axhline(y=w_lr_std, color='r', label=f'w_lr_std={w_lr_std}')
ax1.axhline(y=w_hr_std, color='b', label=f'w_hr_std={w_hr_std}')
ax1.set_ylabel('Width')
ax1.legend()

ax2.scatter(dtime_lst, h_lr_arr, s=size, c='r', label='lr (HMI)')
ax2.scatter(dtime_lst, h_hr_arr, s=size, c='b', label='hr (SP)')
ax2.axhline(y=h_lr_std, color='r', label=f'h_lr_std={h_lr_std}')
ax2.axhline(y=h_hr_std, color='b', label=f'h_hr_std={h_hr_std}')
ax2.set_ylabel('Height')
ax2.legend()

ax3.scatter(dtime_lst, w_ups, s=size, c='r', label='W-upscale')
ax3.scatter(dtime_lst, h_ups, s=size, c='b', label='H-upscale')
ax3.axhline(y=w_ups_std, color='r', label=f'w_ups_std={w_ups_std}')
ax3.axhline(y=h_ups_std, color='b', label=f'h_ups_std={h_ups_std}')
ax3.set_xlabel('Time (mm/dd/yy)')
ax3.set_ylabel('Upscale factor')
ax3.legend()
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%D'))

plt.tight_layout()
plt.show()
        
db
        