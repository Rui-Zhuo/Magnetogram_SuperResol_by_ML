"""Original metric definitions, retained for comparison with existing tables."""
import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity as ssim

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
