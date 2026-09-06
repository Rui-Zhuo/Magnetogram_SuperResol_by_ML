import os
import numpy as np
import pandas as pd

from utils_load_data import load_L72_sample, load_output_sample

# input_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/dataset/L72/saveData/'
# output_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/test_20260416/test_pred/'
# save_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/test_20260416/'

solar_phase = 'solar_minimum' # 'solar_maximum, solar_minimum
root_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/application/full-disk/'

if solar_phase == 'solar_maximum':
    
    REC_time = '20240505120000/'
    save_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/apply_20260416_3/'
    
elif solar_phase == 'solar_minimum':
    
    REC_time = '20190125010000/'
    save_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/apply_20260416_5/'
    
    # REC_time = '20190407062400/'
    # save_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/apply_20260416_4/'

input_dir = os.path.join(root_dir, REC_time, 'HMI_patches/')
output_dir = os.path.join(save_dir, 'test_pred/')

input_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
output_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

print(f'Found {len(input_files)} input files and {len(output_files)} output files.')

matched_pairs = []

for output_file in output_files:
    output_path = os.path.join(output_dir, output_file)
    inp_op, gt_op, pred_op, sample_idx_op = load_output_sample(output_path)

    if not np.any(np.isfinite(pred_op)):
        print(f'Skipped {output_file} with NaN pred_op')
        continue
    
    op_corners = [inp_op[0, 0, 0], inp_op[0, 0, -1], inp_op[0, -1, 0], inp_op[0, -1, -1],
                  inp_op[0, 1, 1], inp_op[0, 1, -2], inp_op[0, -2, 1], inp_op[0, -2, -2],
                  inp_op[0, 2, 2], inp_op[0, 2, -3], inp_op[0, -3, 2], inp_op[0, -3, -3],
                  inp_op[0, 3, 3], inp_op[0, 3, -4], inp_op[0, -4, 3], inp_op[0, -4, -4]]
    
    match_found = False
    
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        HMIfield_ip, SPfield_ip, Txy_ip = load_L72_sample(input_path)
        
        ip_corners = [HMIfield_ip[0, 0], HMIfield_ip[0, -1], HMIfield_ip[-1, 0], HMIfield_ip[-1, -1], 
                      HMIfield_ip[1, 1], HMIfield_ip[1, -2], HMIfield_ip[-2, 1], HMIfield_ip[-2, -2],
                      HMIfield_ip[2, 2], HMIfield_ip[2, -3], HMIfield_ip[-3, 2], HMIfield_ip[-3, -3],
                      HMIfield_ip[3, 3], HMIfield_ip[3, -4], HMIfield_ip[-4, 3], HMIfield_ip[-4, -4]]
        
        valid_pairs = []
        for op_val, ip_val in zip(op_corners, ip_corners):
            if not (np.isnan(op_val) or np.isnan(ip_val)):
                valid_pairs.append((op_val, ip_val))
        
        if len(valid_pairs) == 0:
            continue
        
        is_match = all(np.isclose(op, ip, atol=1e-8) for op, ip in valid_pairs)
        
        if is_match:
            matched_pairs.append([input_file, output_file])
            match_found = True
            print(f'Matching {input_file} to {output_file}')
            break
    
    if not match_found:
        print(f'===!!! WARNING: No matching input file found for output file: {output_file}')
        
df = pd.DataFrame(matched_pairs, columns=['input_filename', 'output_filename'])
save_fn = 'matched_pairs.csv'
save_path = os.path.join(save_dir, save_fn)
df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f'Done! Matched {len(matched_pairs)} pairs of I/O files.')
print(f'Saved matching table to: {save_path}')