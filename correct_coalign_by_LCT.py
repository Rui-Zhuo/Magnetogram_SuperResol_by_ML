import os
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
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

def fourier_lct(im1, im2, win_size=15, search_size=30, step=None):
    """
    基于傅里叶变换的局部相关追踪法（Fourier-LCT）+ 配准恢复
    适配：非0-255、任意尺寸（矩形/偶数）、自动标准化，输出位移场+配准数组3
    修复：np.polyfit二次拟合的解包错误，增加鲁棒性校验
    """
    # 输入基础校验
    assert im1.shape == im2.shape, "两张数组尺寸必须一致！"
    assert search_size > win_size, "搜索窗口尺寸必须大于参考窗口尺寸！"
    step = win_size if step is None else step
    H, W = im1.shape
    # 初始化位移场
    dx = np.zeros((H, W), dtype=np.float32)
    dy = np.zeros((H, W), dtype=np.float32)
    # 转换为浮点型
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # 步骤1：按自身绝对值最大值标准化
    norm1 = np.max(np.abs(im1)) + 1e-8
    norm2 = np.max(np.abs(im2)) + 1e-8
    im1_norm = im1 / norm1
    im2_norm = im2 / norm2

    # 步骤2：窗口切片参数（兼容奇偶/矩形）
    win_half = win_size / 2
    search_half = search_size / 2
    y_range = range(int(win_half), H - int(win_half), step)
    x_range = range(int(win_half), W - int(win_half), step)

    # 滑动窗口计算位移场
    for y in y_range:
        for x in x_range:
            # 提取参考窗口和搜索窗口（严格尺寸校验）
            ref_y1, ref_y2 = int(y - win_half), int(y + win_half)
            ref_x1, ref_x2 = int(x - win_half), int(x + win_half)
            sea_y1, sea_y2 = int(y - search_half), int(y + search_half)
            sea_x1, sea_x2 = int(x - search_half), int(x + search_half)
            
            ref_win = im1_norm[ref_y1:ref_y2, ref_x1:ref_x2]
            search_win = im2_norm[sea_y1:sea_y2, sea_x1:sea_x2]
            
            if ref_win.shape != (win_size, win_size) or search_win.shape != (search_size, search_size):
                continue

            # 去均值预处理
            ref_win_cent = ref_win - np.mean(ref_win)
            search_win_cent = search_win - np.mean(search_win)

            # 零填充参考窗口
            ref_win_pad = np.zeros_like(search_win, dtype=np.float32)
            pad_half = (search_size - win_size) // 2
            ref_win_pad[pad_half:pad_half + win_size, pad_half:pad_half + win_size] = ref_win_cent

            # 频域互相关（卷积定理）
            f_ref = fft2(ifftshift(ref_win_pad))
            f_search = fft2(ifftshift(search_win_cent))
            f_corr = f_ref * np.conj(f_search)
            corr_raw = fftshift(np.real(ifft2(f_corr)))

            # 归一化修正
            norm_ref = np.sqrt(np.sum(ref_win_cent ** 2) + 1e-8)
            norm_search = np.sqrt(np.sum(search_win_cent ** 2) + 1e-8)
            corr_norm = corr_raw / (norm_ref * norm_search)

            # 整像素峰值检测
            y_max, x_max = np.unravel_index(np.argmax(corr_norm), corr_norm.shape)
            dy_int = y_max - (search_size // 2)
            dx_int = x_max - (search_size // 2)

            # 亚像素抛物线插值 —— 核心修复处
            if 1 <= y_max <= search_size - 2 and 1 <= x_max <= search_size - 2:
                patch = corr_norm[y_max-1:y_max+2, x_max-1:x_max+2]
                # 修复1：二次多项式拟合返回3个系数[a, b, c]，完整解包
                # y方向插值（ax²+bx+c）
                y_vals = [-1, 0, 1]
                y_patch_vals = np.max(patch, axis=1)  # 按行取最大，保证3个值
                a_y, b_y, c_y = np.polyfit(y_vals, y_patch_vals, 2)
                y_sub = -b_y / (2 * a_y) if abs(a_y) > 1e-8 else 0.0
                # x方向插值（ax²+bx+c）
                x_patch_vals = np.max(patch, axis=0)  # 按列取最大，保证3个值
                a_x, b_x, c_x = np.polyfit(y_vals, x_patch_vals, 2)
                x_sub = -b_x / (2 * a_x) if abs(a_x) > 1e-8 else 0.0
                # 最终亚像素位移
                dy_sub = dy_int + y_sub
                dx_sub = dx_int + x_sub
            else:
                dy_sub = dy_int
                dx_sub = dx_int

            # 赋值位移场
            dy[ref_y1:ref_y2, ref_x1:ref_x2] = dy_sub
            dx[ref_y1:ref_y2, ref_x1:ref_x2] = dx_sub

    # 步骤3：配准恢复，新建数组3
    im3 = register_by_displacement(im2, dx, dy, output_shape=(H, W))

    return dx, dy, im3

import numpy as np
from scipy import fftpack
from scipy.ndimage import map_coordinates


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

from scipy.interpolate import RectBivariateSpline


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


# -------------------------- 测试代码：全流程验证（无报错） --------------------------
if __name__ == "__main__":
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

    # # 1. 生成模拟测试数据：非0-255、矩形尺寸、亚像素位移
    # H, W = 200, 300  # 矩形非正方形
    # im1 = np.random.randn(H, W).astype(np.float32) * 80 + 20  # 数值范围~[-60,100]
    # # 生成im2：im1亚像素平移（dx=3.6, dy=2.4）+ 少量噪声
    # im2 = np.roll(np.roll(im1, 3, axis=1), 2, axis=0)
    # im2 = im2 + np.random.randn(H, W).astype(np.float32) * 2

    # 2. 运行Fourier-LCT全流程（无解包错误）
    # dx, dy, im3 = fourier_lct(
    #     im1=hmi_crop[2:,:],
    #     hmi_crop[:-2,:], #sp_interp2hmi,
    #     win_size=11,
    #     search_size=31,
    #     step=2
    # )
    
    # A, B 为你的两个二维数组
    C, dx_map, dy_map = local_subpixel_registration(hmi_crop, sp_interp2hmi, window=16, step=1)
    
    # Crop HMI field to standard LR size
    h_beg_hmi = (h_hmi - h_lr_std) // 2
    h_end_hmi = h_beg_hmi + h_lr_std
    w_beg_hmi = (w_hmi - w_lr_std) // 2
    w_end_hmi = w_beg_hmi + w_lr_std
    
    hmi_cut = hmi_crop[h_beg_hmi:h_end_hmi, w_beg_hmi:w_end_hmi]
    txy_cut = txy_crop[h_beg_hmi:h_end_hmi, w_beg_hmi:w_end_hmi]
    spfield_shift = bicubic_interp_2d(sp_interp2hmi, h_new=h_sp, w_new=w_sp)
    
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
    sp_cut = bicubic_interp_2d(spfield_shift,
                               h_new=h_hr_std, w_new=w_hr_std,
                               h_start=h_beg_sp, h_end=h_end_sp,
                               w_start=w_beg_sp, w_end=w_end_sp)

    # 3. 可视化全流程结果
    plt.figure(figsize=(20, 8))
    # 参考数组im1
    plt.subplot(2, 3, 1)
    plt.imshow(hmi_cut, cmap='gray')
    plt.title('1. Reference Array (im1) | Shape: {}×{}'.format(h_lr_std, w_lr_std))
    plt.axis('off')
    # 偏移数组im2
    plt.subplot(2, 3, 2)
    plt.imshow(SPfield_cut_old, cmap='gray')
    plt.title('2. Shifted Array (im2)')
    plt.axis('off')
    # 配准后数组im3
    plt.subplot(2, 3, 3)
    plt.imshow(sp_cut, cmap='gray')
    plt.title('3. Registered Array (im3) | Aligned with im1')
    plt.axis('off')
    # 位移场dx
    plt.subplot(2, 3, 4)
    plt.imshow(dx_map, cmap='jet')
    plt.colorbar(label='dx (pixels)', shrink=0.8)
    plt.title('4. Displacement Field dx (sub-pixel)')
    plt.axis('off')
    # 位移场dy
    plt.subplot(2, 3, 5)
    plt.imshow(dy_map, cmap='jet')
    plt.colorbar(label='dy (pixels)', shrink=0.8)
    plt.title('5. Displacement Field dy (sub-pixel)')
    plt.axis('off')
    # im1与im3的差值
    # diff = np.abs(hmi_crop[2:,:] - C)
    plt.subplot(2, 3, 6)
    # plt.imshow(diff, cmap='Reds')
    plot_correlation_2d(C, hmi_crop, 'HR_Bicubic_shift', 'LR', bins=50, vmax=None)
    # plt.colorbar(label='Absolute Difference', shrink=0.8)
    plt.title('6. |im1 - im3| | Smaller = Better Registration')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 4. 定量验证配准效果
    valid_mask = (dx != 0) & (dy != 0)
    dx_valid = dx[valid_mask]
    dy_valid = dy[valid_mask]
    diff_valid = diff[valid_mask]
    print("="*60)
    print("Displacement Field Statistics (Valid Area):")
    # print(f"有效追踪像素数：{np.sum(valid_mask)}/{H*W} ({np.sum(valid_mask)/(H*W)*100:.1f}%)")
    print(f"dx均值：{np.mean(dx_valid):.2f} 像素，标准差：{np.std(dx_valid):.2f}")
    print(f"dy均值：{np.mean(dy_valid):.2f} 像素，标准差：{np.std(dy_valid):.2f}")
    print("="*60)
    print("Registration Accuracy Statistics (Valid Area):")
    print(f"im1与im3差值均值：{np.mean(diff_valid):.2f}")
    print(f"im1与im3差值标准差：{np.std(diff_valid):.2f}")
    print(f"im1与im3最大差值：{np.max(diff_valid):.2f}")
    print("="*60)