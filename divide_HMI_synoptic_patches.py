import os
import sunpy.map
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/application/synoptic/HMI_patches/'
save_or_not = 1

# Import HMI synoptic data
syn_dir = 'E:/Research/Data/SDO/HMI/Synoptic/'
syn_fn = 'hmi.Synoptic_Mr.2261.fits'
syn_path = os.path.join(syn_dir, syn_fn)

# Check the shape of the HMI synoptic map
syn_map = sunpy.map.Map(syn_path)
syn_field = syn_map.data
print(syn_field.shape) # 1440, 3600

# syn_map.plot()
# plt.show()

# Import hmi full disk data
fdk_dir = 'E:/Research/Data/SDO/HMI/FullDisk/Magnetogram/4096/20240504-20240507/'
fdk_fn = 'hmi.M_720s.20240507_000000_TAI.3.magnetogram.fits'
fdk_path = os.path.join(fdk_dir, fdk_fn)

# Read HMI field
fdk_map = sunpy.map.Map(fdk_path)
fdk_field = fdk_map.data[:, ::-1]

# Get arcsec per pixel
H, W = fdk_map.data.shape[0], fdk_map.data.shape[1]
HMIX, HMIY = np.meshgrid(np.array(range(W)), np.array(range(H)))
sc = fdk_map.pixel_to_world(HMIX*u.pix, HMIY*u.pix)
HMI_Tx = sc.Tx.arcsec[:, ::-1]
HMI_Ty = sc.Ty.arcsec[:, ::-1]
HMI_Txy = np.sqrt(HMI_Tx**2 + HMI_Ty**2)

# Get the dimensions of the magnetogram
height, width = fdk_field.shape

# Define patch size and stride
patch_size = 200
stride = 200

# Calculate the number of patches in each dimension
num_patches_y = ((height // 2 - patch_size) // stride + 1) * 2

# Show the magnetogram
plt.figure(figsize=(12, 6))
vmin, vmax = -300, 300

plt.subplot(1, 2, 1)
im1 = plt.imshow(fdk_field, cmap='gray', vmin=vmin, vmax=vmax)
plt.colorbar(im1, orientation='horizontal')
plt.title('Original HMI Magnetogram')

# Show all division lines, which begin from the center of the image
plt.subplot(1, 2, 2)
im2 = plt.imshow(fdk_field, cmap='gray', vmin=vmin, vmax=vmax)
plt.colorbar(im2, orientation='horizontal')
for i in range(num_patches_y + 1):
    y = height // 2 - (num_patches_y // 2 - i) * stride
    plt.axhline(y=y, color='red', linestyle='--', linewidth=1)
for i in range(2):
    x = width // 2 - (1/2 - i) * stride
    plt.axvline(x=x, color='blue', linestyle='--', linewidth=1)
plt.title('Divided HMI Magnetogram')
if save_or_not == 1:
    plt.savefig(os.path.join(save_dir, f'{fdk_fn[:-17]}_division.png'), dpi=300)

plt.tight_layout()

# Divide the magnetogram into patches and save them
HMI_patches = []
Txy_patches = []
x_center_lst = []
y_center_lst = []
for i in range(num_patches_y):
        y_start = height // 2 - (num_patches_y // 2 - i) * stride
        x_start = int(width // 2 - 1/2 * stride)
        y_center = y_start + patch_size // 2
        x_center = x_start + patch_size // 2
        
        HMI_patch = fdk_field[y_start:y_start + patch_size, x_start:x_start + patch_size]
        Txy_patch = HMI_Txy[y_start:y_start + patch_size, x_start:x_start + patch_size]
        
        if not np.isnan(HMI_patch).any():
            HMI_patches.append(HMI_patch)
            Txy_patches.append(Txy_patch)
            x_center_lst.append(x_center)
            y_center_lst.append(y_center)

# Save the patches as npz files
if save_or_not == 1:
    for i, (HMI_patch, Txy_patch, x_center, y_center) in enumerate(zip(HMI_patches, Txy_patches, x_center_lst, y_center_lst)):
            save_fn = f'{fdk_fn[:-17]}_patch.{i}.npz'
            save_path = os.path.join(save_dir, save_fn)
            np.savez(save_path, HMI=HMI_patch, Txy=Txy_patch)
    print(f'Saved {len(HMI_patches)} patches to {save_dir}')

# Show randomly selected patches
num_show = 9
np.random.seed(42)
shuffled_indices = np.random.permutation(len(HMI_patches))
plt.figure(figsize=(9, 9))
for i in range(num_show):
    idx = shuffled_indices[i]
    plt.subplot(3, 3, i + 1)
    im = plt.imshow(HMI_patches[idx], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f'Patch {idx} @({x_center_lst[idx]}, {y_center_lst[idx]}) pixels')
    plt.axis('off')

plt.tight_layout()
if save_or_not == 1:
    plt.savefig(os.path.join(save_dir, f'{fdk_fn[:-17]}_samples.png'), dpi=300)

plt.show()