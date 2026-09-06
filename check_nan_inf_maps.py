import os
import numpy as np

def loadField(file_path):
    fields = np.load(file_path)
    HMIfield = fields['HMIfield']
    SPfield = fields['SPfield']
    Txy = fields['Txy']
    
    return HMIfield, SPfield, Txy

data_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/dataset/saveData/'
save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/dataset/'
log_path = os.path.join(save_dir, 'check_nan_files.txt')

####################################################################################################
HMI_nan_lst = []
SP_nan_lst = []

for root, dirs, files in os.walk(data_dir):
    print(f'Begin processing: {root}')
    for fn in files:
        if fn.endswith('.npz'):
            file_path = os.path.join(root, fn)
                
            HMIfield, SPfield, Txy = loadField(file_path)
                    
            if np.isnan(HMIfield).any() or np.isinf(HMIfield).any():
                HMI_nan_lst.append(fn)
            elif np.isnan(SPfield).any() or np.isinf(SPfield).any():
                SP_nan_lst.append(fn)

print(f'HMI_nan_lst: {HMI_nan_lst}')
print(f'SP_nan_lst: {SP_nan_lst}')

with open(log_path, 'w', encoding='utf-8') as f:
    f.write('=' * 50 + '\n')
    
    f.write('HMI NAN/INF FILES\n')
    f.write(f'Total number: {len(HMI_nan_lst)}\n')
    if HMI_nan_lst:
        for idx, fn in enumerate(HMI_nan_lst, 1):
            f.write(f'{idx}. {fn}\n')
    
    f.write('\n')  # 空行分隔，提升可读性
    
    f.write('SP NAN/INF FILES\n')
    f.write(f'Total number: {len(SP_nan_lst)}\n')
    if SP_nan_lst:
        for idx, fn in enumerate(SP_nan_lst, 1):
            f.write(f'{idx}. {fn}\n')

print(f'Check results saved to: {log_path}')