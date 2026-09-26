"""Local phase-correlation kernels copied from the research pipeline."""
import numpy as np
from scipy import fftpack
from scipy.interpolate import RectBivariateSpline
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
