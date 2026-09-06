import numpy as np
import pandas as pd
from scipy import interpolate
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from constants import h_shape_lr, w_shape_lr, h_shape_hr, w_shape_hr, pixel_area_LR, pixel_area_HR

# ==================== Reference maps ====================
def get_bicubic_map(field_lr, h_shape_hr, w_shape_hr):
    # Bicubic map from interpolation of the LR map
    h_shape_lr, w_shape_lr = field_lr.shape
    h_arr_lr = np.linspace(0, h_shape_lr, h_shape_lr)
    w_arr_lr = np.linspace(0, w_shape_lr, w_shape_lr)

    h_arr_hr = np.linspace(0, h_shape_lr, h_shape_hr)
    w_arr_hr = np.linspace(0, w_shape_lr, w_shape_hr)

    interp_func = interpolate.interp2d(w_arr_lr, h_arr_lr, field_lr, kind='cubic')
    bicubic_map = interp_func(w_arr_hr, h_arr_hr)
    
    return bicubic_map

def get_empirical_map_from_dalda2017(bic_map):
    # Empirical relationship from Table 5 in Dalda et al. (2017)
    # Dividing region: weak plage < 175G < strong plage < 650G < penumbra < 1750G < umbra
    # Umbra:        B_HMI = 0.84 * B_SP - 0.26
    # Penumbra:     B_HMI = 0.95 * B_SP - 0.02
    # Strong plage: B_HMI = 0.32 * B_SP - 0.02
    # Weak plage:   B_HMI = 0.32 * B_SP - 0.00
    emp_map = bic_map.copy()
    bic_map_abs = np.abs(bic_map)
    emp_map[bic_map_abs >= 1750] = (bic_map[bic_map_abs >= 1750] + 0.26) / 0.84
    emp_map[(bic_map_abs >= 650) & (bic_map_abs < 1750)] = (bic_map[(bic_map_abs >= 650) & (bic_map_abs < 1750)] + 0.02) / 0.95
    emp_map[(bic_map_abs >= 175) & (bic_map_abs < 650)] = (bic_map[(bic_map_abs >= 175) & (bic_map_abs < 650)] + 0.02) / 0.32
    emp_map[bic_map_abs < 175] = (bic_map[bic_map_abs < 175] + 0.00) / 0.32
    return emp_map

def get_empirical_map_from_zhang2023(bic_map):
    # Empirical relationship from Zhang et al. (2023)
    # B_HMI = 0.93 * B_SP
    emp_map = bic_map / 0.93
    return emp_map

# ==================== Evaluation metrics ====================
def calc_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(gt - pred))
    return mae

def calc_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    # Calculate RMSE (Root Mean Squared Error)
    mse = mean_squared_error(gt, pred)
    rmse = np.sqrt(mse)
    return rmse

def calc_pearson_cc(pred: np.ndarray, gt: np.ndarray) -> float:
    # Calculate Pearson Correlation Coefficient 
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    pred_mean = pred_flat.mean()
    gt_mean = gt_flat.mean()

    numerator = np.sum((pred_flat - pred_mean) * (gt_flat - gt_mean))
    denom_pred = np.sqrt(np.sum((pred_flat - pred_mean) ** 2))
    denom_gt = np.sqrt(np.sum((gt_flat - gt_mean) ** 2))

    eps = 1e-8
    cc = numerator / (denom_pred * denom_gt + eps)
    return cc

def calc_R2(pred: np.ndarray, gt: np.ndarray) -> float:
    # Calculate R-squared (Coefficient of Determination)
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    ss_res = np.sum((gt_flat - pred_flat) ** 2)
    ss_tot = np.sum((gt_flat - gt_flat.mean()) ** 2)

    eps = 1e-8
    r2_score = 1 - (ss_res / (ss_tot + eps))
    return r2_score

def calc_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    # Calculate SSIM (Structural Similarity Index Measure)
    score = ssim(gt, pred, data_range=gt.max() - gt.min())
    return score

def calc_PSNR(pred: np.ndarray, gt: np.ndarray) -> float:
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = mean_squared_error(gt, pred)
    if mse == 0:
        return float('inf')
    max_pixel = gt.max()
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calc_net_flux(br_map: np.ndarray, pixel_area: float = 1.0) -> float:
    # Calculate net flux
    return np.sum(br_map * 1e-5) * pixel_area

def calc_unsigned_flux(br_map: np.ndarray, pixel_area: float = 1.0) -> float:
    # Calculate unsigned flux
    return np.sum(np.abs(br_map * 1e-5)) * pixel_area

# ==================== Evaluation functions ====================
def eval_metrics(HR_map, field_map):
    # Evaluate metrics, including RMSE, Pearson CC, R2, SSIM, and PSNR
    mae_field = calc_mae(field_map, HR_map)
    rmse_field = calc_rmse(field_map, HR_map)
    cc_field = calc_pearson_cc(field_map, HR_map)
    r2_field = calc_R2(field_map, HR_map)
    ssim_field = calc_ssim(field_map, HR_map)
    psnr_field = calc_PSNR(field_map, HR_map)
    
    return mae_field, rmse_field, cc_field, r2_field, ssim_field, psnr_field

def eval_flux(field_map):
    # Evaluate net and unsigned fluxes throughtout the whole map
    if field_map.shape == (h_shape_lr, w_shape_lr):
        flux_net = calc_net_flux(field_map, pixel_area_LR)
        flux_uns = calc_unsigned_flux(field_map, pixel_area_LR)
    elif field_map.shape == (h_shape_hr, w_shape_hr):
        flux_net = calc_net_flux(field_map, pixel_area_HR)
        flux_uns = calc_unsigned_flux(field_map, pixel_area_HR)
        
    return flux_net, flux_uns