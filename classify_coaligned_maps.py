import os
import shutil 
import numpy as np
import matplotlib.pyplot as plt

def loadField(file_path):
    fields = np.load(file_path)
    HMIfield_crop = fields['HMIfield_crop']
    HMIfield_coalign = fields['HMIfield_coalign']
    SPfield_coalign = fields['SPfield_coalign']
    Txy_crop = fields['Txy_crop']
    
    return HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop

data_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/coalignment/'

norm_dir = os.path.join(data_dir, 'normal_mode/')
fast_valid_dir = os.path.join(data_dir, 'fast_mode_valid/')
fast_invalid_dir = os.path.join(data_dir, 'fast_mode_invalid/')
target_dirs = [norm_dir, fast_valid_dir, fast_invalid_dir]

####################################################################################################
x_ups_std = 1.68
y_ups_std = 1.56
x_ups_thres = 2.5
y_ups_thres = 2.5

# Iterating to show the image shape
for root, dirs, files in os.walk(data_dir):
    print(f'Begin processing: {root}')
    for fn in files:
        if fn.endswith('.npz'):
            if root+'/' in target_dirs:
                continue
            
            file_path = os.path.join(root, fn)
                
            HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop = loadField(file_path)
            REC_time = fn[:-4]
            
            y_crop, x_crop = HMIfield_crop.shape
            y_coal, x_coal = SPfield_coalign.shape
            
            x_ups = x_coal / x_crop
            y_ups = y_coal / y_crop
                
            if x_ups >= x_ups_thres or y_ups >= y_ups_thres:
                dest_path = os.path.join(norm_dir, fn)
                shutil.move(file_path, dest_path)
                print(f'Moved {fn} to normal_mode directory')
            
            elif x_ups_std <= x_ups < x_ups_thres and y_ups_std <= y_ups < y_ups_thres:
                dest_path = os.path.join(fast_valid_dir, fn)
                shutil.move(file_path, dest_path)
                print(f'Moved {fn} to fast_mode_valid directory')
            
            elif x_ups < x_ups_std or y_ups < y_ups_std:
                dest_path = os.path.join(fast_invalid_dir, fn)
                shutil.move(file_path, dest_path)
                print(f'Moved {fn} to fast_mode_invalid directory')
            
            else:
                print(f'Cannot classify {fn} into any category')