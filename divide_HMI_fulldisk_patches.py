import os
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
import sunpy.map 
from scipy import interpolate
from astropy.io import fits
import matplotlib.pyplot as plt
import time

epoch_beg = time.perf_counter()

def interp_bicubic(field_lr):
    h_ups, w_ups = 1.56, 1.68
    h_shape_lr, w_shape_lr = 200, 200
    h_shape_hr, w_shape_hr = int(h_shape_lr * h_ups), int(w_shape_lr * w_ups)
    
    h_arr_lr = np.linspace(0, h_shape_lr, h_shape_lr)
    w_arr_lr = np.linspace(0, w_shape_lr, w_shape_lr)

    h_arr_hr = np.linspace(0, h_shape_lr, h_shape_hr)
    w_arr_hr = np.linspace(0, w_shape_lr, w_shape_hr)

    interp_func = interpolate.interp2d(w_arr_lr, h_arr_lr, field_lr, kind='cubic')
    field_interp = interp_func(w_arr_hr, h_arr_hr)
    
    return field_interp

# Import hmi full disk data
solar_phase = 'solar_minimum' # 'solar_maximum','solar_minimum'
root_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/application/full-disk/'
save_or_not = 0

if solar_phase == 'solar_maximum':
    
    REC_time = '20240505120000/'
    HMI_fn = 'hmi.M_720s.20240505_120000_TAI.3.magnetogram.fits'
    
elif solar_phase == 'solar_minimum':
    
    REC_time = '20190125010000/'
    HMI_fn = 'hmi.M_720s.20190125_010000_TAI.3.magnetogram.fits'
    
    # REC_time = '20190407062400/'
    # HMI_fn = 'hmi.M_720s.20190407_062400_TAI.3.magnetogram.fits'

case_dir = os.path.join(root_dir, REC_time)
save_dir = os.path.join(case_dir, 'HMI_patches/')

HMI_path = os.path.join(case_dir, HMI_fn)

# Read HMI field
HMI_map = sunpy.map.Map(HMI_path)
HMI_field = HMI_map.data[:, ::-1]

# Get arcsec per pixel
H, W = HMI_map.data.shape[0], HMI_map.data.shape[1]
HMIX, HMIY = np.meshgrid(np.array(range(W)), np.array(range(H)))
sc = HMI_map.pixel_to_world(HMIX*u.pix, HMIY*u.pix)
HMI_Tx = sc.Tx.arcsec[:, ::-1]
HMI_Ty = sc.Ty.arcsec[:, ::-1]
HMI_Txy = np.sqrt(HMI_Tx**2 + HMI_Ty**2)

# Get the dimensions of the magnetogram
height, width = HMI_field.shape

# Define patch size and stride
patch_size = 200
stride = 200
    
# Calculate the number of patches in each dimension
num_patches_y = ((height // 2 - patch_size) // stride + 1) * 2
num_patches_x = ((width // 2 - patch_size) // stride + 1) * 2

# Show the magnetogram
plt.figure(figsize=(12, 6))
vmin, vmax = -300, 300

plt.subplot(1, 2, 1)
im1 = plt.imshow(HMI_field, cmap='gray', vmin=vmin, vmax=vmax)
plt.colorbar(im1, orientation='horizontal')
plt.title('Original HMI Magnetogram')

# Show all division lines, which begin from the center of the image
plt.subplot(1, 2, 2)
im2 = plt.imshow(HMI_field, cmap='gray', vmin=vmin, vmax=vmax)
plt.colorbar(im2, orientation='horizontal')
for i in range(num_patches_y + 1):
    y = height // 2 - (num_patches_y // 2 - i) * stride
    plt.axhline(y=y, color='red', linestyle='--', linewidth=1)
for i in range(num_patches_x + 1):
    x = width // 2 - (num_patches_x // 2 - i) * stride
    plt.axvline(x=x, color='blue', linestyle='--', linewidth=1)
plt.title('Divided HMI Magnetogram')
if save_or_not == 1:
    plt.savefig(os.path.join(save_dir, f'division_scheme.png'), dpi=300)

plt.tight_layout()

# Divide the magnetogram into patches and save them
HMI_patches = []
Txy_patches = []
bic_patches = []
x_start_lst = []
y_start_lst = []
for i in range(num_patches_y):
    for j in range(num_patches_x):
        y_start = height // 2 - (num_patches_y // 2 - i) * stride
        x_start = width // 2 - (num_patches_x // 2 - j) * stride
        
        HMI_patch = HMI_field[y_start:y_start + patch_size, x_start:x_start + patch_size]
        Txy_patch = HMI_Txy[y_start:y_start + patch_size, x_start:x_start + patch_size]
        bic_patch = interp_bicubic(HMI_patch)
        
        # if not np.isnan(HMI_patch).all():
        HMI_patches.append(HMI_patch)
        Txy_patches.append(Txy_patch)
        bic_patches.append(bic_patch)
        x_start_lst.append(x_start)
        y_start_lst.append(y_start)

# Save the patches as npz files
if save_or_not == 1:
    for i, (HMI_patch, Txy_patch, bic_patch, x_start, y_start) in enumerate(zip(HMI_patches, Txy_patches, bic_patches, x_start_lst, y_start_lst)):
            save_fn = f'patch.xstart.{x_start}.ystart.{y_start}.npz'
            save_path = os.path.join(save_dir, save_fn)
            np.savez(save_path, HMIfield=HMI_patch, SPfield=bic_patch, Txy=Txy_patch)
    print(f'Saved {len(HMI_patches)} patches to {save_dir}')

epoch_end = time.perf_counter()
print(f'Time used: {epoch_end - epoch_beg:.2f} seconds')

# Show randomly selected patches
num_show = 9
np.random.seed(42)
shuffled_indices = np.random.permutation(len(HMI_patches))
plt.figure(figsize=(9, 9))
for i in range(num_show):
    idx = shuffled_indices[i]
    plt.subplot(3, 3, i + 1)
    im = plt.imshow(HMI_patches[idx], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f'Patch {idx} @({x_start_lst[idx]}, {y_start_lst[idx]}) pixels')
    plt.axis('off')

plt.suptitle(f'Patches extracted from {HMI_fn}', fontsize=12)
plt.tight_layout()
if save_or_not == 1:
    plt.savefig(os.path.join(save_dir, f'division_samples.png'), dpi=300)

plt.show()