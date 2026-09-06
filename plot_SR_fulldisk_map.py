import os
import sunpy.map
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
from scipy.ndimage import convolve
from matplotlib.lines import Line2D
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm, SymLogNorm

from constants import h_ups, w_ups, h_HMI, w_HMI, h_HMI_edge, w_HMI_edge

def interp_linear(field_lr):
    h_ups, w_ups = 1.56, 1.68
    h_shape_lr, w_shape_lr = 4000, 4000
    h_shape_hr, w_shape_hr = int(h_shape_lr * h_ups), int(w_shape_lr * w_ups)
    
    nan_mask = np.isnan(field_lr)
    field_temp = np.nan_to_num(field_lr, nan=0.0)
    
    h_arr_lr = np.linspace(0, h_shape_lr, h_shape_lr)
    w_arr_lr = np.linspace(0, w_shape_lr, w_shape_lr)

    h_arr_hr = np.linspace(0, h_shape_lr, h_shape_hr)
    w_arr_hr = np.linspace(0, w_shape_lr, w_shape_hr)
    
    interp_func = interpolate.interp2d(w_arr_lr, h_arr_lr, field_temp, kind='cubic')
    field_interp = interp_func(w_arr_hr, h_arr_hr)
    
    nan_mask_hr = interpolate.interp2d(w_arr_lr, h_arr_lr, nan_mask.astype(float), kind='linear')(w_arr_hr, h_arr_hr)
    field_interp[nan_mask_hr > 0.5] = np.nan
    
    return field_interp

solar_phase = 'solar_minimum'
work_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/application/full-disk/'
fov_SP_or_not = 1 # 0 - SP FOV; 1 - show FOV
save_or_not = 1

cmap_fdk = 'gray'
cmap_sub = 'bwr'
# colors = ['blue', 'cyan', 'orange', 'red']
colors = ['darkblue', 'blue', 'cyan', 'orange', 'red', 'darkred']
# colors = ['cyan', 'skyblue', 'crimson', 'darkred']

if solar_phase == 'solar_maximum':
    
    REC_time = '20240505120000/'
    HMI_fn = 'hmi.M_720s.20240505_120000_TAI.3.magnetogram.fits'
    SP_fn = '20240505_115853_L2.1.fits'
    
    vmin_fdk, vmax_fdk = -200, 200
    norm_fdk = plt.Normalize(vmin=vmin_fdk, vmax=vmax_fdk)
    vmin_sub, vmax_sub = -2000, 2000
    norm_sub = plt.Normalize(vmin=vmin_sub, vmax=vmax_sub)
    levels = [-1500, -1000, -500, 500, 1000, 1500]
    # levels = [-1500, -750, 750, 1500]
    
    xmin_cut, xmax_cut = 1310, 1550
    ymin_cut, ymax_cut = 2410, 2650
    xmin_fov_cut, xmax_fov_cut = 2700, 2940 # SP FOV
    ymin_fov_cut, ymax_fov_cut =  980, 1220 # SP FOV

elif solar_phase == 'solar_minimum':
    
    # case 1 for solar minimum: 20190125010000
    REC_time = '20190125010000/'
    HMI_fn = 'hmi.M_720s.20190125_010000_TAI.3.magnetogram.fits'
    SP_fn = '20190125_010005_L2.1.fits'
    
    vmin_fdk, vmax_fdk = -50, 50
    norm_fdk = plt.Normalize(vmin=vmin_fdk, vmax=vmax_fdk)
    vmin_sub, vmax_sub = -600, 600
    norm_sub = plt.Normalize(vmin=vmin_sub, vmax=vmax_sub)
    
    levels = [-450, -300, -150, 150, 300, 450]
    
    xmin_cut, xmax_cut = 2300, 2450
    ymin_cut, ymax_cut = 1500, 1650
    xmin_fov_cut, xmax_fov_cut =  850, 1100 # SP FOV
    ymin_fov_cut, ymax_fov_cut = 1420, 1670 # SP FOV
    
    # # case 2 for solar minimum: 20190407062400
    # REC_time = '20190407062400/'
    # HMI_fn = 'hmi.M_720s.20190407_062400_TAI.3.magnetogram.fits'
    # SP_fn = '20190407_062200_L2.1.fits'
    
    # vmin_fdk, vmax_fdk = -50, 50
    # norm_fdk = plt.Normalize(vmin=vmin_fdk, vmax=vmax_fdk)
    # vmin_sub, vmax_sub = -50, 50
    # norm_sub = plt.Normalize(vmin=vmin_sub, vmax=vmax_sub)
    
    # levels = [-150, -100, -50, 50, 100, 150]
    # # levels = [-100, -50, -30, 30, 50, 100]
    # # levels = [-100, -50, 50, 100]
    
    # xmin_cut, xmax_cut = 2300, 2600
    # ymin_cut, ymax_cut = 2080, 2380
    # xmin_fov_cut, xmax_fov_cut = 1950, 2230 # SP FOV
    # ymin_fov_cut, ymax_fov_cut = 3550, 3830 # SP FOV

HMI_path = os.path.join(work_dir, REC_time, HMI_fn)
SP_path = os.path.join(work_dir, REC_time, SP_fn)

SR_dir = os.path.join(work_dir, REC_time)
SR_path = os.path.join(SR_dir, 'SR_map.npz')
SRF_path = os.path.join(SR_dir, 'SRF_map.npz')

HMI_map = sunpy.map.Map(HMI_path)
SP_map = fits.open(SP_path)
SR_map = np.load(SR_path)

HMIfield = np.fliplr(HMI_map.data)
SP_field = np.flipud(SP_map[4].data)
SRfield = SR_map['SRfield']

if os.path.exists(SRF_path):
    SRFfield = np.load(SRF_path)['SRFfield']
else:
    LINfield = interp_linear(HMIfield[w_HMI_edge:-w_HMI_edge, h_HMI_edge:-h_HMI_edge])
    SRFfield = np.where(np.isfinite(SRfield), SRfield, LINfield)
    np.savez(SRF_path, SRFfield=SRFfield)
    print(f'Saved SRF map to: {SRF_path}')

sun_center_hpc = HMI_map.center
sun_radius_arcsec = HMI_map.rsun_obs
theta = np.linspace(0, 2 * np.pi, 360)
x_world = sun_center_hpc.Tx + sun_radius_arcsec * np.cos(theta)
y_world = sun_center_hpc.Ty + sun_radius_arcsec * np.sin(theta)
sky_coords = SkyCoord(x_world, y_world, frame=HMI_map.coordinate_frame)
limb_x = HMI_map.world_to_pixel(sky_coords).x.value
limb_y = HMI_map.world_to_pixel(sky_coords).y.value

def save_pure_image(data, save_path, norm, cmap, levels=None, colors=None, color_rect='lime', limb=None, box=None):
    h, w = data.shape
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap=cmap, norm=norm)
    if levels is not None:
        ax.contour(data, levels=levels, colors=colors, linewidths=1)
    if limb is not None:
        ax.plot(limb[0], limb[1], 'y', linewidth=1)
    if box is not None:
        xb, yb, wb, hb = box
        ax.add_patch(plt.Rectangle((xb, yb), wb, hb, linewidth=1, edgecolor=color_rect, facecolor='none'))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')

    if save_or_not == 1:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

if fov_SP_or_not == 0:
    xmin_cut, xmax_cut = xmin_fov_cut, xmax_fov_cut
    ymin_cut, ymax_cut = ymin_fov_cut, ymax_fov_cut
    color_fov = 'purple'
else:
    color_fov = 'lime'
    
xstart = xmin_cut
ystart = ymin_cut
save_path = os.path.join(SR_dir, f'xstart.{xstart}.ystart.{ystart}.zoomout_HMI.png')
save_pure_image(
    HMIfield, save_path, norm=norm_fdk, cmap=cmap_fdk, 
    levels=None, colors=colors, color_rect=color_fov, 
    limb=(limb_x, limb_y),
    box=(xmin_cut, ymin_cut, xmax_cut-xmin_cut, ymax_cut-ymin_cut)
)

save_path = os.path.join(SR_dir, f'xstart.{xstart}.ystart.{ystart}.zoomout_SR.png')
limb_hr = ((limb_x-w_HMI_edge)*w_ups, (limb_y-h_HMI_edge)*h_ups)
box_hr = (
    (xmin_cut-w_HMI_edge)*w_ups,
    (ymin_cut-h_HMI_edge)*h_ups,
    (xmax_cut-xmin_cut)*w_ups,
    (ymax_cut-ymin_cut)*h_ups
)
save_pure_image(
    SRFfield, save_path, norm=norm_fdk, cmap=cmap_fdk,
    levels=None, colors=colors, color_rect=color_fov, 
    limb=limb_hr,
    box=box_hr
)

if fov_SP_or_not == 0:
    save_path = os.path.join(SR_dir, f'xstart.{xstart}.ystart.{ystart}.zoomin_SP.png')
    save_pure_image(SP_field, save_path, norm=norm_sub, cmap=cmap_sub, levels=levels, colors=colors)

HMIcut = HMIfield[ymin_cut:ymax_cut, xmin_cut:xmax_cut]
save_path = os.path.join(SR_dir, f'xstart.{xstart}.ystart.{ystart}.zoomin_HMI.png')
save_pure_image(HMIcut, save_path, norm=norm_sub, cmap=cmap_sub, levels=levels, colors=colors)

SRcut = SRfield[
    int((ymin_cut-h_HMI_edge)*h_ups) : int((ymax_cut-h_HMI_edge)*h_ups),
    int((xmin_cut-w_HMI_edge)*w_ups) : int((xmax_cut-w_HMI_edge)*w_ups)
]
save_path = os.path.join(SR_dir, f'xstart.{xstart}.ystart.{ystart}.zoomin_SR.png')
save_pure_image(SRcut, save_path, norm=norm_sub, cmap=cmap_sub, levels=levels, colors=colors)

if save_or_not == 1:
    # Colorbar for full-disk map
    fig, cax = plt.subplots(figsize=(6, 0.5))
    sm = plt.cm.ScalarMappable(norm=norm_fdk, cmap=cmap_fdk)
    sm.set_array([])
    
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=12)
    
    cax.set_xlabel('')
    cax.set_ylabel('')
    
    cb_path = os.path.join(SR_dir, f'colorbar_fulldisk.png')
    plt.savefig(cb_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    
    # Colorbar for subregion map
    fig, cax = plt.subplots(figsize=(6, 0.5))
    sm = plt.cm.ScalarMappable(norm=norm_sub, cmap=cmap_sub)
    sm.set_array([])
    
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=12)
    
    cax.set_xlabel('')
    cax.set_ylabel('')
    
    cb_path = os.path.join(SR_dir, f'colorbar_subregion.png')
    plt.savefig(cb_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    
    # Contour legend
    legend_elements = [
        Line2D([0], [0], color=c, lw=3, label=f'{lvl} G') 
        for lvl, c in zip(levels, colors)
    ]
    fig, ax = plt.subplots(figsize=(10, 0.5))
    ax.legend(handles=legend_elements, loc='center', ncol=6, fontsize=12)
    ax.axis('off')
    plt.savefig(os.path.join(SR_dir, f'legend_contour.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()