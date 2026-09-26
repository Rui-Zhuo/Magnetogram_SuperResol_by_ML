"""Metrics in Gauss; legacy PSNR and projected pixel flux conventions."""
import numpy as np
from skimage.metrics import structural_similarity
from scipy.interpolate import RectBivariateSpline


def bicubic(field, shape):
    """Equivalent regular-grid replacement for removed scipy.interp2d."""
    h, w = field.shape
    return RectBivariateSpline(np.linspace(0, h, h), np.linspace(0, w, w), field)(
        np.linspace(0, h, shape[0]), np.linspace(0, w, shape[1]))


def evaluate(pred, gt):
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    if pred.shape != gt.shape or gt.ndim != 2 or min(gt.shape) < 7:
        raise ValueError('Metrics require matching 2-D maps at least 7 by 7')
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError('Non-finite samples must be explicitly excluded, not silently filled')
    error = pred - gt
    mae, rmse = np.abs(error).mean(), np.sqrt(np.mean(error**2))
    p, g = pred - pred.mean(), gt - gt.mean()
    cc = np.sum(p*g) / (np.sqrt(np.sum(p*p)*np.sum(g*g)) + 1e-8)
    r2 = 1 - np.sum(error**2) / (np.sum(g*g) + 1e-8)
    span, sigma = np.ptp(gt), gt.std()
    ssim = structural_similarity(gt, pred, data_range=span) if span > 0 else np.nan
    # Preserve the original table definition. This is NOT range-based PSNR.
    psnr = np.inf if rmse == 0 else 20*np.log10(gt.max()/rmse) if gt.max() > 0 else np.nan
    ratio = lambda x, y: float(x/y) if abs(y) > 1e-12 else np.nan
    return dict(MAE_G=mae, RMSE_G=rmse, MAE_over_sigma=ratio(mae, sigma),
                RMSE_over_sigma=ratio(rmse, sigma), CC=cc, R2=r2, SSIM=ssim,
                PSNR_dB=psnr, net_flux_ratio=ratio(pred.sum(), gt.sum()),
                unsigned_flux_ratio=ratio(np.abs(pred).sum(), np.abs(gt).sum()))


def flux_weber(field_gauss, pixel_area_m2):
    """Physical SI conversion: 1 G = 1e-4 T (legacy code used 1e-5)."""
    field = np.asarray(field_gauss, dtype=float)
    if pixel_area_m2 <= 0 or not np.isfinite(field).all():
        raise ValueError('Require finite field and positive projected pixel area')
    return dict(net_Wb=field.sum()*1e-4*pixel_area_m2,
                unsigned_Wb=np.abs(field).sum()*1e-4*pixel_area_m2)
