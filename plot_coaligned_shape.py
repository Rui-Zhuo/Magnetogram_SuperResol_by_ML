import os
import numpy as np
import matplotlib.pyplot as plt

def loadField(target, fn):
    file_path = os.path.join(target, fn)
    fields = np.load(file_path)
    HMIfield_crop = fields['HMIfield_crop']
    HMIfield_coalign = fields['HMIfield_coalign']
    SPfield_coalign = fields['SPfield_coalign']
    
    fields.close()
    
    return HMIfield_crop, HMIfield_coalign, SPfield_coalign

data_dir = 'E:/Research/Work/Super_Resolution_Magnetogram/coalignment/SaveData/'
save_dir = 'E:/Research/Work/Super_Resolution_Magnetogram/magnetic_flux/'
save_or_not = 0

####################################################################################################
# Iterating to show the image shape
x_crop_lst = []
y_crop_lst = []
x_coal_lst = []
y_coal_lst = []

for fn in os.listdir(data_dir):
    if fn.endswith('.npz'):
        HMIfield_crop, HMIfield_coalign, SPfield_coalign = loadField(data_dir, fn)
        REC_time = fn[:-4]
        
        y_crop, x_crop = HMIfield_crop.shape
        y_coal, x_coal = SPfield_coalign.shape
        
        x_crop_lst.append(x_crop)
        y_crop_lst.append(y_crop)
        x_coal_lst.append(x_coal)
        y_coal_lst.append(y_coal)

####################################################################################################
# Plotting the shape of images
plt.figure(figsize=(14,6))
size = 30
linewidth = 2
fontsize = 10

x_HMI_std = 220
y_HMI_std = 240
x_upscale = 1.7   # x_SP = 646
y_upscale = 1.575 # y_SP = 378
x_SP_std = x_HMI_std * x_upscale
y_SP_std = y_HMI_std * y_upscale
xmin, xmax = 0, 600
ymin, ymax = 0, 900

plt.subplot(121)
plt.scatter(x_crop_lst, x_coal_lst, s=size, c='r', label='X-shape')
plt.scatter(y_crop_lst, y_coal_lst, s=size, c='b', label='Y-shape')
plt.vlines(x=x_HMI_std, colors='r', linestyles='dashed', ymin=ymin, ymax=ymax, label=f'x_HMI={x_HMI_std}')
plt.vlines(x=y_HMI_std, colors='b', linestyles='dashed', ymin=ymin, ymax=ymax, label=f'y_HMI={y_HMI_std}')
plt.hlines(y=x_SP_std, colors='r', xmin=xmin, xmax=xmax, label=f'x_SP={int(x_SP_std)}')
plt.hlines(y=y_SP_std, colors='b', xmin=xmin, xmax=xmax, label=f'y_SP={int(y_SP_std)}')
plt.legend(fontsize=fontsize)
plt.xlabel('HMI_crop image')
plt.ylabel('SP image')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.grid()

plt.subplot(122)
plt.hist2d(np.array(x_coal_lst)/np.array(x_crop_lst), np.array(y_coal_lst)/np.array(y_crop_lst), 
           range=[[1.68,1.74], [1.56,1.62]], bins=[20, 20], cmap='jet')
plt.colorbar()
plt.scatter(np.array(x_coal_lst)/np.array(x_crop_lst), np.array(y_coal_lst)/np.array(y_crop_lst), s=1, c='k')
# plt.legend(fontsize=fontsize)
plt.xlabel('X upscale factor')
plt.ylabel('Y upscale factor')
plt.grid()

plt.show()
        
db
        