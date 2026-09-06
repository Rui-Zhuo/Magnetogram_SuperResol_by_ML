import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.colors import LogNorm

# --------------------------
# Data Loading Function
# --------------------------
def load_field(target_dir: str, file_name: str) -> tuple:
    """
    Load magnetic field data from .npz file
    Args:
        target_dir: Directory of the .npz file
        file_name: Name of the .npz file
    Returns:
        Tuple of 4 field arrays: HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop
    """
    file_path = os.path.join(target_dir, file_name)
    fields = np.load(file_path)
    hmi_crop = fields['HMIfield_crop']
    hmi_coalign = fields['HMIfield_coalign']
    sp_coalign = fields['SPfield_coalign']
    txy_crop = fields['Txy_crop']
    return hmi_crop, hmi_coalign, sp_coalign, txy_crop

def load_old_field(dir, fn):
    file_path = os.path.join(dir, fn)
    fields_cut = np.load(file_path)
    HMIfield_cut = fields_cut['HMIfield']
    SPfield_cut = fields_cut['SPfield']
    Txy_cut = fields_cut['Txy']
    return HMIfield_cut, SPfield_cut, Txy_cut

# --------------------------
# Universal Bicubic Interpolation
# Merge interp_bicubic/interp_bicubic_with_shift/interp_bicubic_with_cut into one function
# --------------------------
def bicubic_interp_2d(field_old: np.ndarray, 
                      h_new: int = None, w_new: int = None,
                      h_start: float = 0, h_end: float = None,
                      w_start: float = 0, w_end: float = None) -> np.ndarray:
    """
    Universal 2D bicubic interpolation for resizing/shift/crop scenarios
    Args:
        field_old: Original 2D field array to be interpolated
        h_new/w_new: New height/width for resized output (used for resize/shift/crop)
        h_start/w_start: Start coordinate of interpolation range (for shift/crop)
        h_end/w_end: End coordinate of interpolation range (for shift/crop)
    Returns:
        Interpolated 2D field array
    """
    h_old, w_old = field_old.shape
    # Original grid coordinates
    h_grid_old = np.linspace(0, h_old, h_old)
    w_grid_old = np.linspace(0, w_old, w_old)
    
    # Set default end coordinate as original shape (for resize/shift)
    h_end = h_old if h_end is None else h_end
    w_end = w_old if w_end is None else w_end
    # Generate new grid coordinates based on input params
    h_grid_new = np.linspace(h_start, h_end, h_new if h_new else h_old)
    w_grid_new = np.linspace(w_start, w_end, w_new if w_new else w_old)
    
    # Bicubic interpolation function
    interp_fun = interpolate.interp2d(w_grid_old, h_grid_old, field_old, kind='cubic')
    field_interp = interp_fun(w_grid_new, h_grid_new)
    return field_interp

# --------------------------
# Evaluation Metric: Cross Correlation (CC)
# --------------------------
def calculate_cc(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate cross correlation coefficient between prediction and ground truth
    Args:
        pred: Predicted 2D field array
        gt: Ground truth 2D field array
    Returns:
        Cross correlation coefficient (CC)
    """
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    epsilon = 1e-8  # Avoid division by zero
    
    numerator = np.sum(pred_flat * gt_flat)
    denominator = np.sqrt(np.sum(pred_flat**2)) * np.sqrt(np.sum(gt_flat**2))
    cc = numerator / (denominator + epsilon)
    return cc

# --------------------------
# Plotting Functions
# --------------------------
def plot_field_map(field: np.ndarray, title: str, cmap: str = 'bwr') -> None:
    """
    Plot 2D magnetic field map with normalized intensity
    Args:
        field: 2D field array to plot
        title: Plot title
        cmap: Colormap for the plot
    """
    v_abs = np.max(np.abs(field))
    plt.imshow(field / v_abs, cmap=cmap, vmin=-0.6, vmax=0.6)
    plt.title(title)

def plot_correlation_2d(pred: np.ndarray, gt: np.ndarray, 
                        x_label: str, y_label: str, 
                        bins: int = 50, vmax: float = None) -> None:
    """
    Plot 2D histogram for correlation between pred and gt, with CC line
    Args:
        pred: Predicted 2D field array
        gt: Ground truth 2D field array
        x_label/y_label: Axis labels for the 2D histogram
        bins: Number of bins for histogram
        vmax: Max value for colorbar clipping
    """
    cc = calculate_cc(pred, gt)
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # 2D histogram with log normalization
    h, x_edges, y_edges, img = plt.hist2d(pred_flat, gt_flat, bins=bins,
                                          cmap='jet', cmin=1, norm=LogNorm(clip=True))
    plt.colorbar()
    if vmax is not None:
        img.set_clim(vmax=vmax)
    
    # Plot y=x reference line with CC value
    val_min = min(np.min(pred_flat), np.min(gt_flat))
    val_max = max(np.max(pred_flat), np.max(gt_flat))
    plt.plot([val_min, val_max], [val_min, val_max], 
             color='purple', label=f'CC={cc:.4f}')
    
    plt.legend()
    plt.axis('equal')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

# --------------------------
# Main Process
# --------------------------
if __name__ == '__main__':
    # Configuration Parameters
    raw_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/coalignment/fast_mode_valid/'
    old_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/dataset/saveData/'
    save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/coalignment/correct_by_CC/'
    h_lr_std = 200  # Standard low resolution height (HMI)
    w_lr_std = 200  # Standard low resolution width (HMI)
    h_ups = 1.56    # Upsampling factor for height (HMI->SP)
    w_ups = 1.68    # Upsampling factor for width (HMI->SP)
    h_hr_std = int(h_lr_std * h_ups)  # Standard high resolution height (SP)
    w_hr_std = int(w_lr_std * w_ups)  # Standard high resolution width (SP)
    save_or_not = 1  # Save flag (unused in current code)

    # Load .npz file list
    file_list = [fn for fn in os.listdir(old_dir) if fn.endswith('.npz')]
    fn = file_list[10000]  # Select 11th file (index 10)
    rec_time = fn[:-4]  # Record time from file name
    print(f'Estimating the coalignment error of file: {fn}')

    # Load field data
    hmi_crop, hmi_coalign, sp_coalign, txy_crop = load_field(raw_dir, fn)
    h_hmi, w_hmi = hmi_crop.shape
    h_sp, w_sp = sp_coalign.shape
    HMIfield_cut_old, SPfield_cut_old, Txy_cut_old = load_old_field(old_dir, fn)

    # Bicubic interpolation between HMI and SP resolution
    hmi_interp2sp = bicubic_interp_2d(hmi_crop, h_new=h_sp, w_new=w_sp)
    sp_interp2hmi = bicubic_interp_2d(sp_coalign, h_new=h_hmi, w_new=w_hmi)

    # Plot 1: Original field comparison (3x3 subplots)
    plt.figure(figsize=(15, 8))
    plt.subplot(331)
    plot_field_map(hmi_crop, 'HMI')

    plt.subplot(332)
    plot_field_map(sp_interp2hmi, 'SP_interp2_HMI')

    plt.subplot(333)
    plot_correlation_2d(hmi_crop, sp_interp2hmi, 'HMI', 'SP_interp2_HMI')

    plt.subplot(334)
    plot_field_map(sp_coalign, 'SP')

    plt.subplot(335)
    plot_field_map(hmi_coalign, 'HMI_coalign')

    plt.subplot(336)
    plot_correlation_2d(sp_coalign, hmi_coalign, 'SP', 'HMI_coalign')

    plt.subplot(337)
    plot_field_map(sp_coalign, 'SP')

    plt.subplot(338)
    plot_field_map(hmi_interp2sp, 'HMI_interp2_SP')

    plt.subplot(339)
    plot_correlation_2d(sp_coalign, hmi_interp2sp, 'SP', 'HMI_interp2_SP')

    plt.suptitle(rec_time)

    # Find optimal shift (w/h) with maximum CC
    print('Finding x-shift and y-shift with maximum CC ...')
    max_cc = -1
    best_shift_w = 0
    best_shift_h = 0

    # Grid search shift range: [-10, 10] for width and height
    for shift_w in range(-10, 11):
        for shift_h in range(-10, 11):
            sp_sub = sp_interp2hmi.copy()
            hmi_sub = hmi_crop.copy()

            # Crop width according to shift
            if shift_w > 0:
                sp_sub = sp_sub[:, shift_w:]
                hmi_sub = hmi_sub[:, :-shift_w]
            elif shift_w < 0:
                sp_sub = sp_sub[:, :shift_w]
                hmi_sub = hmi_sub[:, -shift_w:]

            # Crop height according to shift
            if shift_h > 0:
                sp_sub = sp_sub[shift_h:, :]
                hmi_sub = hmi_sub[:-shift_h, :]
            elif shift_h < 0:
                sp_sub = sp_sub[:shift_h, :]
                hmi_sub = hmi_sub[-shift_h:, :]

            # Calculate CC for current shift
            curr_cc = calculate_cc(sp_sub, hmi_sub)
            if curr_cc > max_cc:
                max_cc = curr_cc
                best_shift_w = shift_w
                best_shift_h = shift_h

    print(f'Best shift: shift_w={best_shift_w}, shift_h={best_shift_h}, with highest cc={max_cc:.6f}')

    # Apply optimal shift to get cropped fields
    sp_bicubic_shift = sp_interp2hmi.copy()
    hmi_shift = hmi_crop.copy()
    txy_shift = txy_crop.copy()

    # Shift width
    if best_shift_w > 0:
        sp_bicubic_shift = sp_bicubic_shift[:, best_shift_w:]
        hmi_shift = hmi_shift[:, :-best_shift_w]
        txy_shift = txy_shift[:, :-best_shift_w]
    elif best_shift_w < 0:
        sp_bicubic_shift = sp_bicubic_shift[:, :-best_shift_w]
        hmi_shift = hmi_shift[:, -best_shift_w:]
        txy_shift = txy_shift[:, -best_shift_w:]

    # Shift height
    if best_shift_h > 0:
        sp_bicubic_shift = sp_bicubic_shift[best_shift_h:, :]
        hmi_shift = hmi_shift[:-best_shift_h, :]
        txy_shift = txy_shift[:-best_shift_h, :]
    elif best_shift_h < 0:
        sp_bicubic_shift = sp_bicubic_shift[:best_shift_h, :]
        hmi_shift = hmi_shift[-best_shift_h:, :]
        txy_shift = txy_shift[-best_shift_h:, :]

    # Shift information for plot annotation
    shift_info = f'x_shift={best_shift_w}\n y_shift={best_shift_h}'

    # Calculate shifted shape dimensions
    h_hmi_shift = h_hmi - np.abs(best_shift_h)
    w_hmi_shift = w_hmi - np.abs(best_shift_w)
    h_sp_shift = h_sp - int(np.abs(best_shift_h) * h_ups)
    w_sp_shift = w_sp - int(np.abs(best_shift_w) * w_ups)

    # Crop HMI field to standard LR size
    h_beg_hmi = (h_hmi_shift - h_lr_std) // 2
    h_end_hmi = h_beg_hmi + h_lr_std
    w_beg_hmi = (w_hmi_shift - w_lr_std) // 2
    w_end_hmi = w_beg_hmi + w_lr_std

    hmi_cut = hmi_shift[h_beg_hmi:h_end_hmi, w_beg_hmi:w_end_hmi]
    txy_cut = txy_shift[h_beg_hmi:h_end_hmi, w_beg_hmi:w_end_hmi]

    # Shift SP field with bicubic interpolation and crop to standard HR size
    sp_shift = bicubic_interp_2d(sp_coalign,
                                 h_start=best_shift_h * h_ups, h_end=h_sp + best_shift_h * h_ups,
                                 w_start=best_shift_w * w_ups, w_end=w_sp + best_shift_w * w_ups)

    # Calculate SP crop range by HMI fraction
    h_beg_frac = h_beg_hmi / h_hmi
    h_end_frac = h_end_hmi / h_hmi
    w_beg_frac = w_beg_hmi / w_hmi
    w_end_frac = w_end_hmi / w_hmi

    h_beg_sp = h_sp * h_beg_frac
    h_end_sp = h_sp * h_end_frac
    w_beg_sp = w_sp * w_beg_frac
    w_end_sp = w_sp * w_end_frac

    # Crop SP field to standard HR size with bicubic interpolation
    sp_cut = bicubic_interp_2d(sp_shift,
                               h_new=h_hr_std, w_new=w_hr_std,
                               h_start=h_beg_sp, h_end=h_end_sp,
                               w_start=w_beg_sp, w_end=w_end_sp)

    # Plot 2: Shifted and cropped field comparison (2x4 subplots)
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plot_field_map(hmi_crop, 'LR (HMI)')

    plt.subplot(234)
    plot_field_map(sp_coalign, 'HR (SP)')

    plt.subplot(232)
    plot_field_map(sp_interp2hmi + hmi_crop, 'Superimpose HR_Bicubic on LR')

    plt.subplot(235)
    plot_correlation_2d(sp_interp2hmi, hmi_crop, 'HR_Bicubic', 'LR', bins=50, vmax=None)

    plt.subplot(233)
    plot_field_map(sp_bicubic_shift + hmi_shift, 'Superimpose HR_Bicubic_shift on LR')

    plt.subplot(236)
    plot_correlation_2d(sp_bicubic_shift, hmi_shift, 'HR_Bicubic_shift', 'LR', bins=50, vmax=None)
    plt.text(0.95, 0.05, shift_info, transform=plt.gca().transAxes,
             ha='right', va='bottom', fontsize=10)

    plt.suptitle(rec_time)
    
    save_name = rec_time + '_superimpose.png'
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    
    plt.figure(figsize=(10, 8))

    plt.subplot(221)
    plot_field_map(HMIfield_cut_old, 'LR (HMI)')

    plt.subplot(223)
    plot_field_map(SPfield_cut_old, 'HR (SP)')

    plt.subplot(222)
    plot_field_map(hmi_cut, 'LR_shift (HMI)')

    plt.subplot(224)
    plot_field_map(sp_cut, 'HR_shift (SP)')

    plt.suptitle(rec_time)
    
    save_name = rec_time + '_comparison.png'
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    
    # plt.show()
    plt.close()

    # db