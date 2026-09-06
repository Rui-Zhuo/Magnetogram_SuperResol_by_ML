import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def loadField(file_path):
    fields = np.load(file_path)
    HMIfield_crop = fields['HMIfield_crop']
    HMIfield_coalign = fields['HMIfield_coalign']
    SPfield_coalign = fields['SPfield_coalign']
    Txy_crop = fields['Txy_crop']
    
    return HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop

def linear_fit(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    
    k_fit, b_fit, r_fit, p_fit, std_fit = stats.linregress(x, y)
    y_fit = k_fit * x + b_fit
    r2_fit = r_fit ** 2
    res_fit = y - y_fit
    res_std = np.std(res_fit, ddof=2)
    
    return k_fit, b_fit, r2_fit, p_fit, std_fit, y_fit, res_std

data_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/coalignment/'
save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/statistics/'
save_or_not = 0

####################################################################################################
# Iterating to calculate the magnetic flux (signed and unsigned)
HMI_flux_lst = []
SP_flux_lst = []
HMI_flux_uns_lst = []
SP_flux_uns_lst = []


for root, dirs, files in os.walk(data_dir):
    print(f'Begin processing: {root}')
    for fn in files:
        if fn.endswith('.npz'):
            file_path = os.path.join(root, fn)
            try:
                HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop = loadField(file_path)
                REC_time = fn[:-4]
                
                if SPfield_coalign.shape[0] / HMIfield_crop.shape[0] > 2.5: # only fast mode
                    continue
                
                HMI_flux = np.sum(HMIfield_crop)
                SP_flux = np.sum(SPfield_coalign)
                HMI_flux_uns = np.sum(np.abs(HMIfield_crop))
                SP_flux_uns = np.sum(np.abs(SPfield_coalign))
                
                HMI_flux_lst.append(HMI_flux)
                SP_flux_lst.append(SP_flux)
                HMI_flux_uns_lst.append(HMI_flux_uns)
                SP_flux_uns_lst.append(SP_flux_uns)
                
            except Exception as e:
                print(f'RaiseError when processing {file_path}: {e}')
                continue        

####################################################################################################
# Eliminating nan values
SP_flux_lst, HMI_flux_lst = np.array(SP_flux_lst), np.array(HMI_flux_lst)
SP_flux_uns_lst, HMI_flux_uns_lst = np.array(SP_flux_uns_lst), np.array(HMI_flux_uns_lst)

flux_mask = ~np.isnan(SP_flux_lst) & ~np.isnan(HMI_flux_lst)
SP_flux_lst = SP_flux_lst[flux_mask]
HMI_flux_lst = HMI_flux_lst[flux_mask]

flux_uns_mask = ~np.isnan(SP_flux_uns_lst) & ~np.isnan(HMI_flux_uns_lst)
SP_flux_uns_lst = SP_flux_uns_lst[flux_uns_mask]
HMI_flux_uns_lst = HMI_flux_uns_lst[flux_uns_mask]

# Sorting list
sorted_pairs = sorted(zip(SP_flux_lst, HMI_flux_lst))
SP_flux_lst_sorted = np.array([x for x, y in sorted_pairs])
HMI_flux_lst_sorted = np.array([y for x, y in sorted_pairs])

sorted_pairs_uns = sorted(zip(SP_flux_uns_lst, HMI_flux_uns_lst))
SP_flux_uns_lst_sorted = np.array([x for x, y in sorted_pairs_uns])
HMI_flux_uns_lst_sorted = np.array([y for x, y in sorted_pairs_uns])


# Conducting linear fit fot coaligned magnetic flux
k_fit, b_fit, r2_fit, p_fit, std_fit, y_fit, res_std = linear_fit(SP_flux_lst_sorted, HMI_flux_lst_sorted)
k_uns_fit, b_uns_fit, r2_uns_fit, p_uns_fit, std_uns_fit, y_uns_fit, res_uns_std = linear_fit(SP_flux_uns_lst_sorted, HMI_flux_uns_lst_sorted)

####################################################################################################
# Plotting the scatters of coaligned magnetic flux
plt.figure(figsize=(10,4))
size = 30
linewidth = 2
fontsize = 10

plt.subplot(121)
plt.scatter(SP_flux_lst, HMI_flux_lst, s=size, c='b', label='data points')
plt.plot(SP_flux_lst_sorted, y_fit, 'r', linewidth=linewidth, label=f'y={k_fit:.2f}x+{b_fit:.2E}, r2={r2_fit:.4f}')
# plt.fill_between(SP_flux_lst_sorted, y_fit+res_std, y_fit-res_std, color='y', alpha=0.3, label='within 1 std')
plt.fill_between(SP_flux_lst_sorted, y_fit+2*res_std, y_fit-2*res_std, color='y', alpha=0.3, label='within 2 std')
plt.legend(fontsize=fontsize)
plt.xlabel(r'$\Phi_{SP}$')
plt.ylabel(r'$\Phi_{HMI}$')
plt.axis('equal')
plt.grid()

plt.subplot(122)
plt.scatter(SP_flux_uns_lst, HMI_flux_uns_lst, s=size, c='b', label='data points')
plt.plot(SP_flux_uns_lst_sorted, y_uns_fit, 'r', linewidth=linewidth, label=f'y={k_uns_fit:.2f}x+{b_uns_fit:.2E}, r2={r2_uns_fit:.4f}')
# plt.fill_between(SP_flux_uns_lst_sorted, y_uns_fit+res_uns_std, y_uns_fit-res_uns_std, color='y', alpha=0.3, label='within 1 std')
plt.fill_between(SP_flux_uns_lst_sorted, y_uns_fit+2*res_uns_std, y_uns_fit-2*res_uns_std, color='y', alpha=0.3, label='within 2 std')
plt.legend(fontsize=fontsize)
plt.xlabel(r'$\Phi_{SP,uns}$')
plt.ylabel(r'$\Phi_{HMI,uns}$')
plt.axis('equal')
plt.grid()

plt.show()
        
db
        