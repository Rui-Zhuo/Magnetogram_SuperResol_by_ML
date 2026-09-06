import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import interpolate
from scipy import fftpack
from scipy.ndimage import map_coordinates
from scipy.interpolate import RectBivariateSpline

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
        Tuple of 4 field arrays: HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_L3
    """
    file_path = os.path.join(target_dir, file_name)
    fields = np.load(file_path)
    HMIfield_crop = fields['HMIfield_crop']
    HMIfield_coalign = fields['HMIfield_coalign']
    SPfield_coalign = fields['SPfield_coalign']
    Txy_crop = fields['Txy_crop']
    return HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop

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
    plt.imshow(field / v_abs, cmap=cmap, vmin=-0.8, vmax=0.8)
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

def bilinear_interp(im, x, y):
    """双线性插值实现亚像素位置的像素值采样"""
    H, W = im.shape
    # 取整得到邻近四个像素的索引
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    # 边界裁剪，避免索引越界
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # 计算插值权重（亚像素偏移量）
    wx0 = x1 - x
    wx1 = x - x0
    wy0 = y1 - y
    wy1 = y - y0

    # 双线性插值计算像素值
    val = (wy0 * (wx0 * im[y0, x0] + wx1 * im[y0, x1]) +
           wy1 * (wx0 * im[y1, x0] + wx1 * im[y1, x1]))
    return val

def register_by_displacement(im2, dx, dy, output_shape):
    """根据位移场将im2配准到im1的空间位置（新建数组3）"""
    H, W = output_shape
    # 生成数组3的网格坐标（im1的原始位置）
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    # 逆向映射：im2中对应im1(x,y)的像素坐标 = (x+dx, y+dy)
    map_x = x_coords + dx  # 亚像素x坐标
    map_y = y_coords + dy  # 亚像素y坐标
    # 双线性插值采样im2的亚像素位置，得到配准后的数组3
    im3 = bilinear_interp(im2, map_x, map_y)
    return im3

def phase_correlation_subpixel(A, B):
    """
    亚像素级 phase correlation
    返回 (dy, dx)
    """

    FA = fftpack.fft2(A)
    FB = fftpack.fft2(B)

    R = FA * FB.conj()
    R /= np.abs(R) + 1e-12

    r = fftpack.ifft2(R).real

    # 找整像素峰值
    maxpos = np.unravel_index(np.argmax(r), r.shape)
    peak_y, peak_x = maxpos

    ny, nx = r.shape

    # 考虑循环边界
    if peak_y > ny // 2:
        peak_y -= ny
    if peak_x > nx // 2:
        peak_x -= nx

    # ---- 亚像素抛物线拟合 ----

    def parabola_interp(p_minus, p, p_plus):
        return 0.5 * (p_minus - p_plus) / (p_minus - 2*p + p_plus + 1e-12)

    # x方向
    x0 = maxpos[1]
    xm1 = (x0 - 1) % nx
    xp1 = (x0 + 1) % nx
    dx_sub = parabola_interp(r[maxpos[0], xm1],
                             r[maxpos[0], x0],
                             r[maxpos[0], xp1])

    # y方向
    y0 = maxpos[0]
    ym1 = (y0 - 1) % ny
    yp1 = (y0 + 1) % ny
    dy_sub = parabola_interp(r[ym1, maxpos[1]],
                             r[y0, maxpos[1]],
                             r[yp1, maxpos[1]])

    dy = peak_y + dy_sub
    dx = peak_x + dx_sub

    return dy, dx

def local_lct_field(A, B, window=32, step=16):
    """
    计算局部位移场（亚像素LCT）

    参数:
    A, B: 二维数组
    window: 窗口大小
    step: 滑动步长

    返回:
    dx_map, dy_map, x_grid, y_grid
    """

    ny, nx = A.shape

    grid_y = list(range(0, ny - window, step))
    grid_x = list(range(0, nx - window, step))

    dy_map = np.zeros((len(grid_y), len(grid_x)))
    dx_map = np.zeros((len(grid_y), len(grid_x)))

    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):

            subA = A[y:y+window, x:x+window]
            subB = B[y:y+window, x:x+window]

            dy, dx = phase_correlation_subpixel(subA, subB)

            dy_map[i, j] = dy
            dx_map[i, j] = dx

    return dx_map, dy_map, np.array(grid_x), np.array(grid_y)

def warp_by_displacement(B, dx_map, dy_map, x_grid, y_grid, window, step):
    """
    根据局部位移场对图像 B 做形变配准

    返回配准后的数组 C
    """

    ny, nx = B.shape

    # 位移场对应的网格点中心
    gx = x_grid + window // 2
    gy = y_grid + window // 2

    # 构建插值函数
    dx_interp = RectBivariateSpline(gy, gx, dx_map)
    dy_interp = RectBivariateSpline(gy, gx, dy_map)

    # 全分辨率坐标
    Y, X = np.mgrid[0:ny, 0:nx]

    # 插值得到每个像素位置的位移
    DX = dx_interp(Y[:, 0], X[0, :])
    DY = dy_interp(Y[:, 0], X[0, :])

    # 构建映射坐标
    coords = np.array([
        Y - DY,
        X - DX
    ])

    # 重采样
    C = map_coordinates(B, coords, order=3, mode='nearest')

    return C

def local_subpixel_registration(A, B, window=32, step=16):
    """
    完整流程：
    - 计算局部位移场
    - 根据位移场配准 B 到 A

    返回:
    C: 配准后的数组
    dx_map, dy_map: 位移场
    """

    dx_map, dy_map, xg, yg = local_lct_field(A, B, window, step)

    C = warp_by_displacement(B, dx_map, dy_map, xg, yg, window, step)

    return C, dx_map, dy_map

def plot_l3_comparison(HMI_L3, SP_L3, HMI_L4, SP_L4, rec_time, save_or_not):
        # Plot 1: Original field comparison (3x3 subplots)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(231)
    plot_field_map(HMI_L3, 'HMI-L3')

    plt.subplot(232)
    plot_field_map(SP_L4, 'SP-L4')

    plt.subplot(233)
    plot_correlation_2d(HMI_L3, SP_L4, 'HMI-L3', 'SP-L4')
    
    plt.subplot(234)
    plot_field_map(SP_L3, 'SP-L3')

    plt.subplot(235)
    plot_field_map(HMI_L4, 'HMI-L4')

    plt.subplot(236)
    plot_correlation_2d(SP_L3, HMI_L4, 'SP-L3', 'HMI-L4')

    plt.suptitle(rec_time)
    
    # if save_or_not:
    #     plt.savefig

def find_best_shift_by_cc(HMI_L3, SP_L4):
    
    print('Finding shift_h and shift_w with maximum CC ...')
    
    max_cc = -1
    best_shift_h = 0
    best_shift_w = 0

    # Grid search shift range: [-10, 10] for width and height
    for shift_h in range(-10, 11):
        for shift_w in range(-10, 11):
            SP_L51_sub = SP_L4.copy()
            HMI_L51_sub = HMI_L3.copy()

            # Crop height according to shift
            if shift_h > 0:
                SP_L51_sub = SP_L51_sub[shift_h:, :]
                HMI_L51_sub = HMI_L51_sub[:-shift_h, :]
            elif shift_h < 0:
                SP_L51_sub = SP_L51_sub[:shift_h, :]
                HMI_L51_sub = HMI_L51_sub[-shift_h:, :]

            # Crop width according to shift
            if shift_w > 0:
                SP_L51_sub = SP_L51_sub[:, shift_w:]
                HMI_L51_sub = HMI_L51_sub[:, :-shift_w]
            elif shift_w < 0:
                SP_L51_sub = SP_L51_sub[:, :shift_w]
                HMI_L51_sub = HMI_L51_sub[:, -shift_w:]

            # Calculate CC for current shift
            curr_cc = calculate_cc(SP_L51_sub, HMI_L51_sub)
            if curr_cc > max_cc:
                max_cc = curr_cc
                best_shift_h = shift_h
                best_shift_w = shift_w

    print(f'Best shift: shift_h={best_shift_h}, shift_w={best_shift_w}, with highest cc={max_cc:.6f}')

    return best_shift_h, best_shift_w

def cc_shift(HMI_L3, SP_L4, Txy_L3):
    # Find optimal shift (w/h) with maximum CC
    best_shift_h, best_shift_w = find_best_shift_by_cc(HMI_L3, SP_L4)
    
    # Apply optimal shift to get cropped fields
    SP_L51 = SP_L4.copy()
    HMI_L51 = HMI_L3.copy()
    Txy_L51 = Txy_L3.copy()

    # Shift width
    if best_shift_w > 0:
        SP_L51 = SP_L51[:, best_shift_w:]
        HMI_L51 = HMI_L51[:, :-best_shift_w]
        Txy_L51 = Txy_L51[:, :-best_shift_w]
    elif best_shift_w < 0:
        SP_L51 = SP_L51[:, :-best_shift_w]
        HMI_L51 = HMI_L51[:, -best_shift_w:]
        Txy_L51 = Txy_L51[:, -best_shift_w:]

    # Shift height
    if best_shift_h > 0:
        SP_L51 = SP_L51[best_shift_h:, :]
        HMI_L51 = HMI_L51[:-best_shift_h, :]
        Txy_L51 = Txy_L51[:-best_shift_h, :]
    elif best_shift_h < 0:
        SP_L51 = SP_L51[:best_shift_h, :]
        HMI_L51 = HMI_L51[-best_shift_h:, :]
        Txy_L51 = Txy_L51[-best_shift_h:, :]
    
    return SP_L51, HMI_L51, Txy_L51, best_shift_h, best_shift_w

def cut_std_grid(HMI_Lx, Txy_Lx, SP_L5x): 
    
    # Calculate shifted shape dimensions
    h_HMI_Lx, w_HMI_Lx = HMI_Lx.shape
      
    h_beg_HMI = (h_HMI_Lx - h_lr_std) // 2
    h_end_HMI = h_beg_HMI + h_lr_std
    w_beg_HMI = (w_HMI_Lx - w_lr_std) // 2
    w_end_HMI = w_beg_HMI + w_lr_std
    
    HMI_L7x = HMI_Lx[h_beg_HMI:h_end_HMI, w_beg_HMI:w_end_HMI]
    Txy_L7x = Txy_Lx[h_beg_HMI:h_end_HMI, w_beg_HMI:w_end_HMI]
    SP_L6x = SP_L5x[h_beg_HMI:h_end_HMI, w_beg_HMI:w_end_HMI]
    
    SP_L7x = bicubic_interp_2d(SP_L6x, h_new=h_hr_std, w_new=w_hr_std)
    
    return HMI_L7x, Txy_L7x, SP_L6x, SP_L7x

def plot_correct_results(HMI_L7, SP_L7, HMI_L71, SP_L71, HMI_L72, SP_L72, rec_time):
    plt.figure(figsize=(10, 8))

    plt.subplot(231)
    plot_field_map(HMI_L7, 'LR (HMI)')

    plt.subplot(234)
    plot_field_map(SP_L7, 'HR (SP)')

    plt.subplot(232)
    plot_field_map(HMI_L71, 'LR')

    plt.subplot(235)
    plot_field_map(SP_L71, 'HR_CC')

    plt.subplot(233)
    plot_field_map(HMI_L72, 'LR')

    plt.subplot(236)
    plot_field_map(SP_L72, 'HR_LCT')

    plt.suptitle(rec_time)

def plot_result_comparison(HMI_L7, HMI_L71, HMI_L72, SP_L6, SP_L61, SP_L62, rec_time, shift_info):
    
    # Plot 2: Shifted and cropped field comparison (2x4 subplots)
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plot_field_map(SP_L6 + HMI_L7, 'Superimpose HR on LR')

    plt.subplot(234)
    plot_correlation_2d(SP_L6, HMI_L7, 'HR', 'LR', bins=50, vmax=None)

    plt.subplot(232)
    plot_field_map(SP_L61 + HMI_L71, 'Superimpose HR_CC on LR')

    plt.subplot(235)
    plot_correlation_2d(SP_L61, HMI_L71, 'HR_CC', 'LR', bins=50, vmax=None)
    plt.text(0.95, 0.05, shift_info, transform=plt.gca().transAxes,
             ha='right', va='bottom', fontsize=10)

    plt.subplot(233)
    plot_field_map(SP_L62 + HMI_L72, 'Superimpose HR_LCT on LR')

    plt.subplot(236)
    plot_correlation_2d(SP_L62, HMI_L72, 'HR_LCT', 'LR', bins=50, vmax=None)

    plt.suptitle(rec_time)

def plot_result(HMI_L7, HMI_L72, SP_L6, SP_L62, rec_time):
    
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plot_field_map(SP_L6, 'HR')

    plt.subplot(223)
    plot_correlation_2d(SP_L6, HMI_L7, 'HR', 'LR', bins=50, vmax=None)
    
    plt.subplot(222)
    plot_field_map(SP_L62, 'HR_LCT (window=32)')

    plt.subplot(224)
    plot_correlation_2d(SP_L62, HMI_L72, 'HR_LCT', 'LR', bins=50, vmax=None)

    plt.suptitle(rec_time)

def plot_lct_dxdy_map(dx_map, dy_map, rec_time):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.imshow(dx_map, cmap='jet')
    plt.colorbar()
    plt.title('dx map')
    
    plt.subplot(122)
    plt.imshow(dy_map, cmap='jet')
    plt.colorbar()
    plt.title('dy map')
    
    plt.suptitle(rec_time)

def plot_single_map(field):
    v_abs = np.max(np.abs(field))
    plt.figure(figsize=(8,8))
    plt.imshow(field / v_abs, cmap='bwr', vmin=-0.8, vmax=0.8)
    plt.axis('off')

# --------------------------
# Main Process
# --------------------------
if __name__ == '__main__':
    # Configuration Parameters
    raw_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/coalignment/fast_mode_valid/'
    old_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/dataset/saveData/'
    save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/dataset/L72/'
    h_lr_std = 200  # Standard low resolution height (HMI)
    w_lr_std = 200  # Standard low resolution width (HMI)
    h_ups = 1.56    # Upsampling factor for height (HMI->SP)
    w_ups = 1.68    # Upsampling factor for width (HMI->SP)
    h_hr_std = int(h_lr_std * h_ups)  # Standard high resolution height (SP)
    w_hr_std = int(w_lr_std * w_ups)  # Standard high resolution width (SP)
    save_or_not = True # Save flag

    # Load .npz file list
    file_list = [fn for fn in os.listdir(old_dir) if fn.endswith('.npz')]
    error_list = [] 
    
    for i_file in range(0, 2500):
        try:
            fn = file_list[i_file]  # Select 11th file (index 10)
            rec_time = fn[:-4]  # Record time from file name
            
            if os.path.exists(os.path.join(save_dir + 'SaveData/', fn)):
                print(f'File {fn} already exists, skipping...')
                continue
            
            print(f'Estimating the coalignment error of file: {fn}')

            # Load field data
            HMI_L3, _, SP_L3, Txy_L3 = load_field(raw_dir, fn)
            h_HMI_L3, w_HMI_L3 = HMI_L3.shape
            h_SP_L3, w_SP_L3 = SP_L3.shape
            
            HMI_L7, SP_L7, Txy_L7 = load_old_field(old_dir, fn)
            h_HMI_L7, w_HMI_L7 = HMI_L7.shape

            # Bicubic interpolation between HMI and SP resolution
            HMI_L4 = bicubic_interp_2d(HMI_L3, h_new=h_SP_L3, w_new=w_SP_L3)
            SP_L4 = bicubic_interp_2d(SP_L3, h_new=h_HMI_L3, w_new=w_HMI_L3)
            SP_L6 = bicubic_interp_2d(SP_L7, h_new=h_HMI_L7, w_new=w_HMI_L7)
            
            ##########################################################################################

            window_size = 32
            SP_L52, dx_map, dy_map = local_subpixel_registration(HMI_L3, SP_L4, window=window_size, step=1)
            
            HMI_L72, Txy_L72, SP_L62, SP_L72 = cut_std_grid(HMI_L3, Txy_L3, SP_L52)
            
            ##########################################################################################
                
            plot_result(HMI_L7, HMI_L72, SP_L6, SP_L62, rec_time)
            if save_or_not:
                save_name = f'{rec_time}.png'
                save_path = os.path.join(save_dir + 'SaveFig/', save_name)
                plt.savefig(save_path)
                plt.close()
                
                save_name = f'{rec_time}.npz'
                np.savez(os.path.join(save_dir + 'SaveData/', save_name), 
                    HMIfield = HMI_L72, 
                    SPfield = SP_L72,
                    Txy = Txy_L72)
            else: 
                plt.show()
                
        except Exception as e:
            error_info = f"HMI: {fn} | Error: {str(e)}"
            error_list.append(error_info)
            continue
    
    if error_list:
        print(f"\nError files ({len(error_list)}):")
        for idx, error in enumerate(error_list, 1):
            print(f"  {idx}. {error}")