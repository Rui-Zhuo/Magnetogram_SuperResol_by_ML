import os
import numpy as np
import pandas as pd

# ==================== Load dataset ====================
def load_L72_sample(npz_path):
    # Load L72 dataset sample from .npy file
    data = np.load(npz_path)
    
    HMIfield = data['HMIfield']
    SPfield = data['SPfield']
    Txy = data['Txy']
    
    data.close()
    return HMIfield, SPfield, Txy

# ==================== Load model output ====================
def load_output_sample(npz_path):
    # Load model output sample from .npz file
    data = np.load(npz_path)
    
    inp = data['inp']
    gt = data['gt']
    pred = data['pred']
    sample_idx = data['sample_idx']
    
    data.close()
    return inp, gt, pred, sample_idx

# ==================== Load application patches ====================  
