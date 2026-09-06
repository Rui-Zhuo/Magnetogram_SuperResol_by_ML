import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

from constants import h_HMI, w_HMI, h_HMI_edge, w_HMI_edge, h_ups, w_ups, h_shape_hr, w_shape_hr
from utils_load_data import load_L72_sample, load_output_sample

epoch_beg = time.perf_counter()

solar_phase = 'solar_minimum'  # 'solar_maximum' or 'solar_minimum'
save_or_not = 0

if solar_phase == 'solar_maximum':
    
    REC_time = '20240505120000/'
    run_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/apply_20260416_3/'
    
if solar_phase == 'solar_minimum':
    
    REC_time = '20190125010000/'
    run_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/apply_20260416_5/'
    
    # REC_time = '20190407062400/'
    # run_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/apply_20260416_4/'

patch_dir = os.path.join(run_dir, 'test_pred/')
root_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/application/full-disk/'
save_dir = os.path.join(root_dir, REC_time)

patch_files = [f for f in os.listdir(patch_dir) if f.endswith('.npz')]
save_fn = 'SR_map.npz'
match_fn = 'matched_pairs.csv'
# match_file = None

print(f'Found {len(patch_files)} patches to produce the full-disk magnetogram.')

if match_fn is None:
    file_list = [(f, f) for f in os.listdir(patch_dir) if f.endswith('.npz')]
else:
    print(f'Reading matching-table from file: {match_fn}...')
    with open(os.path.join(run_dir, match_fn), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        file_list = [(row[0].strip(), row[1].strip()) for row in reader if len(row) >= 2]

print(f'Total patches to process: {len(file_list)}')

SRfield = np.full((int((h_HMI-2*h_HMI_edge)*h_ups), int((w_HMI-2*w_HMI_edge)*w_ups)), np.nan, dtype=np.float32)

for input_file, coord_file in file_list:
    x_start = int(re.search(r'xstart\.(\d+)', input_file).group(1))
    y_start = int(re.search(r'ystart\.(\d+)', input_file).group(1))

    patch_path = os.path.join(patch_dir, coord_file)
    
    inp, gt, pred, sample_idx = load_output_sample(patch_path) 

    h_start = int((y_start - h_HMI_edge) * h_ups)
    w_start = int((x_start - w_HMI_edge) * w_ups)
    SRfield[h_start:h_start+h_shape_hr, w_start:w_start+w_shape_hr] = pred.reshape((h_shape_hr, w_shape_hr))

epoch_end = time.perf_counter()
print(f'Time used: {epoch_end - epoch_beg:.2f} seconds')

if save_or_not == 1: 
    save_path = os.path.join(save_dir, save_fn)
    np.savez(save_path, SRfield=SRfield)
    print(f'Saved SR map to: {save_path}')

plt.figure()
plt.imshow(SRfield, cmap='gray', vmin=-1000, vmax=1000)
plt.colorbar()
plt.show()
