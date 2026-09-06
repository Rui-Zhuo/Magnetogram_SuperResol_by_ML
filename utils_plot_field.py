import os
import glob
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

def plot_field_map(field, title, cmap='bwr', vlim=None, symlog=False, linthresh=1.0, text=None):
    
    if vlim is None:
        vabs = np.max(np.abs(field))
        vmin, vmax = -vabs, vabs
    else:
        vmin, vmax = vlim[0], vlim[1]
    
    norm = None
    if symlog:
        norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        plt.imshow(field, cmap=cmap, norm=norm)
    else:    
        plt.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax)
    
    plt.colorbar(orientation='horizontal', shrink=0.9)
    plt.title(title)
    # plt.axis('equal')
    
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if text: 
        plt.text(x=0.99, y=0.01, s=text,
                 transform=plt.gca().transAxes, ha='right', va='bottom')

def plot_field_distribution(gt, pred, xlabel, ylabel, bins=50, vmax=None, symlog=False, text=None):
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    valid_gt = np.isfinite(gt_flat)
    valid_pred = np.isfinite(pred_flat)
    
    if not np.any(valid_gt) or not np.any(valid_pred):
        return 
    
    h, xedges, yedges, img = plt.hist2d(gt_flat, pred_flat, bins=bins, \
        cmap='jet', cmin=1, norm=LogNorm(clip=True))
    plt.colorbar(orientation='horizontal')
    
    # plt.colorbar()
    if vmax:
        img.set_clim(vmax=vmax)
        
    if symlog:
        plt.xscale('symlog', linthresh=1.0)
        plt.yscale('symlog', linthresh=1.0)
    else:
        plt.xscale('linear')
        plt.yscale('linear')
    
    plt.plot([-3000, 3000], [-3000, 3000], color='k')
    
    plt.axis('square')
    
    abs_max = max(np.max(np.abs(pred_flat)), np.max(np.abs(gt_flat)))
    plt.xlim(-abs_max*0.8, abs_max*0.8)
    plt.ylim(-abs_max*0.8, abs_max*0.8)     
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(f'{xlabel} .vs. {ylabel}')
    
    if text: 
        plt.text(x=0.99, y=0.01, s=text, 
                 transform=plt.gca().transAxes, ha='right', va='bottom')