import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadDataset(target, fn):
    file_path = os.path.join(target, fn)
    fields = np.load(file_path)
    HMIfield_cut = fields['HMIfield_cut']
    SPfield_cut = fields['SPfield_cut']
    
    fields.close()
    
    return HMIfield_cut, SPfield_cut

def SaveFigure(target, fn, HMIfield, SPfield):
    cv2.imwrite(os.path.join(target+'LR/', fn), HMIfield)
    cv2.imwrite(os.path.join(target+'HR/', fn), SPfield)

data_dir = 'E:/Research/Work/Super_Resolution_Magnetogram/dataset/SaveData/'
save_dir = 'E:/Research/Work/Super_Resolution_Magnetogram/dataset/'
save_or_not = 1

for fn in os.listdir(data_dir):
    if fn.endswith('.npz'):
        HMIfield, SPfield = loadDataset(data_dir, fn)
        REC_time = fn[:-4]
        
        # Flipping array
        HMIfield_flip = np.flipud(HMIfield)
        SPfield_flip = np.flipud(SPfield)
        
        # Normalizing [-300, 3000] to [0, 255]
        HMI_norm = ((HMIfield_flip + 4000) / 8000 * 255).astype(np.uint8)
        SP_norm = ((SPfield_flip + 4000) / 8000 * 255).astype(np.uint8)
        
        # Saving as gray images
        SaveFigure(save_dir, REC_time + '.png', HMI_norm, SP_norm)