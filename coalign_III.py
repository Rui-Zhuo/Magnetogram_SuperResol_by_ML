import astropy.io.fits as fits
import astropy.units as u
import sunpy.map 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as ndimage
import os
import sys
import re
import pdb
from matplotlib.colors import LogNorm

def slitToPixLocation(slitpos):
    '''Given the slit position from SOT/SP, return the actual position
    locations of the columns.'''
    slitpos = slitpos[0][0]
    slitdiff = np.diff(slitpos)
    normalJump = np.median(slitdiff)
    actualPixLocation = (slitpos-slitpos[0])/normalJump
    return actualPixLocation

def slitDrop(X, slitpos):
    '''Given an image where the columns indicate slit positions, take only the
    actual ones. Makes the image smaller, or slitDrop(X,..).shape[1] <= 
    slitDrop(X,..).shape[1].'''
    actualPixLocation = slitToPixLocation(slitpos)
    return np.hstack([X[:,[int(i)]] for i in actualPixLocation])

def slitInterp(X, slitpos, mode='linear'):
    '''Given an image where the columns indicate slit indices, expand it so that
    the columns indicate slit positions. Makes the image bigger, or 
    slitInterp(X,slitpos).shape[1] >= slitInterp(X,slitpos).shape[1]'''
    actualPixLocation = slitToPixLocation(slitpos)
   
    XInterp = np.zeros((X.shape[0],int(np.max(actualPixLocation))+1))

    if mode == 'linear':
        for i in range(X.shape[0]):
            XInterp[i,:] = np.interp(np.arange(XInterp.shape[1]), actualPixLocation, X[i,:])
    elif mode == 'nearest':
        for i in range(X.shape[0]):
            interpnn = scipy.interpolate.interp1d(actualPixLocation, X[i,:],kind='nearest')
            XInterp[i,:] = interpnn(np.arange(XInterp.shape[1]))
    return XInterp

def affineXYToYX(A):
    #Given a XY affine matrix, convert it to assume YX
    return np.array([
        [A[1,1], A[1,0], A[1,2]],
        [A[0,1], A[0,0], A[0,2]],
        [A[2,0], A[2,1], A[2,2]]
        ])
        
def denanify(X):
    X[np.isnan(X)] = 0
    return X

def get_sorted_files(folder):
    all_items = os.listdir(folder)
    files = [
        os.path.join(folder, item) 
        for item in all_items 
        if os.path.isfile(os.path.join(folder, item))
    ]
    files.sort(key=lambda x: os.path.basename(x))
    return files

def normalize_hmi_filename(filename):
    """
    Normalize HMI filename by removing numeric suffix (.1/.3/.n) before .magnetogram.fits
    Example:
    - Input: hmi.m_720s.20140102_031200_TAI.1.magnetogram.fits
    - Output: hmi.m_720s.20140102_031200_TAI.magnetogram.fits
    """
    if filename.startswith('hmi.m_720s.') and filename.endswith('.magnetogram.fits'):
        pattern = r'\.(\d+)\.magnetogram\.fits$'
        normalized_name = re.sub(pattern, '.magnetogram.fits', filename)
        return normalized_name
    return filename

def find_hmi_file_by_normalized_name(srcHMI, target_fn):
    """
    Find HMI file in srcHMI directory by normalized filename (ignore .1/.3 suffix)
    Return: full path of the first matching file (or None if not found)
    """
    normalized_target = normalize_hmi_filename(target_fn)
    for file_name in os.listdir(srcHMI):
        if os.path.isfile(os.path.join(srcHMI, file_name)):
            normalized_file = normalize_hmi_filename(file_name)
            if normalized_file == normalized_target:
                return os.path.join(srcHMI, file_name)
    return None

def parse_download_match_by_line(match_log_path):
    """
    Parse download_match_*.txt by line number (skip header)
    Return: list of dicts -> [{'line_num': line_num, 'col2_fn': col2_fn, 'col3_fn': col3_fn}, ...]
    """
    match_lines = []
    skip_header = True
    line_num = 0     

    with open(match_log_path, 'r', encoding='utf-8') as f:
        # Fix: Correct enumerate usage (original code had bug here)
        for idx, line in enumerate(f, 1):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            if skip_header:
                if '-' * 80 in line_stripped or '-' * 120 in line_stripped:
                    skip_header = False
                continue
            
            line_num += 1
            parts = line_stripped.split('|')
            
            fnSP = parts[0].strip() if len(parts) >= 1 else ''
            fnHMI = parts[1].strip() if len(parts) >= 2 else ''

            match_lines.append({
                'line_num': line_num,
                'fnSP': fnSP,
                'fnHMI': fnHMI
            })

    return match_lines

if __name__ == '__main__':
    year_record = '2015'
    srcUpdate = os.path.join('E:/HinodeSOTSPLevel2Update/Main/', year_record)
    srcSPL2_rd = os.path.join('E:/Research/Work/Magnetogram_SuperResol_by_NN/SPL2_reduced/', year_record)
    srcSPL21 = os.path.join('E:/Research/Data/HINODE/SP/L2.1/', year_record)
    srcHMI = os.path.join('E:/Research/Data/SDO/HMI/FullDisk/Magnetogram/4096/', year_record)
    target = os.path.join('E:/Research/Work/Magnetogram_SuperResol_by_NN/coalignment/', year_record)
    
    fnmatch = 'download_match_' + year_record + '.txt'
    match_log = os.path.join('E:/Research/Data/SDO/HMI/FullDisk/Magnetogram/4096/', fnmatch)

    # Create target directories (including SaveFig)
    if not os.path.exists(target):
        os.mkdir(target)
    savefig_dir = os.path.join(target, 'SaveFig')
    if not os.path.exists(savefig_dir):
        os.mkdir(savefig_dir)
    savedata_dir = os.path.join(target, 'SaveData')
    if not os.path.exists(savedata_dir):
        os.mkdir(savedata_dir)

    # Get sorted files
    filesUpdate = get_sorted_files(srcUpdate)
    filesSPL2_rd = get_sorted_files(srcSPL2_rd)
    filesSPL21 = get_sorted_files(srcSPL21)
    filesHMI = get_sorted_files(srcHMI)
    
    # Parse match log
    match_lines = parse_download_match_by_line(match_log)
    avail_num = 0
    missing_list = []
    
    # Fix: Iterate over match_lines (not filesUpdate length)
    for pair in match_lines:
        line_num = pair['line_num']
        fnSP = pair['fnSP']  # For SPL2/SPL2_rd/SPL21
        fnHMI = pair['fnHMI'] 
        
        print(f'begin processing No.{line_num} pair: ')
        print(f'{fnSP} and {fnHMI}')
        
        # Construct file paths
        fnUpdate = os.path.join(srcUpdate, fnSP) if fnSP else ''
        fnSPL2_rd = os.path.join(srcSPL2_rd, fnSP) if fnSP else ''
        fnSPL21 = os.path.join(srcSPL21, fnSP) if fnSP else ''  # Fix: Original had fnSPL21 instead of fnSP
        
        # Find HMI file (ignore .1/.3 suffix)
        hmiFieldName = None
        if fnHMI:
            hmiFieldName = find_hmi_file_by_normalized_name(srcHMI, fnHMI)
        
        # Check file existence
        missing_file = []
        if fnSP and not os.path.exists(fnSPL2_rd):
            missing_file.append(f"SPL2_rd: {fnSPL2_rd}")
        if fnSP and not os.path.exists(fnSPL21):
            missing_file.append(f"SPL21: {fnSPL21}")
        if fnHMI and not hmiFieldName:
            missing_file.append(f"HMI: {fnHMI}")
        
        if missing_file:
            missing_list.extend(missing_file)
            continue
        
        avail_num += 1
        RecTime = os.path.basename(fnUpdate)[-20:-5] if fnUpdate else ''
        
        # Load data
        update = fits.open(fnUpdate)
        SPL2_rd = fits.open(fnSPL2_rd)
        SPL21 = fits.open(fnSPL21)

        # OLD X/Y Coordinates
        SP_XOLD = SPL2_rd[1].data
        SP_YOLD = SPL2_rd[2].data

        # Updated X/Y Coordinates
        SP_XNEW = update[38].data
        SP_YNEW = update[39].data

        # Calculate min/max points
        pointXMin = min(np.nanmin(SP_XOLD), np.nanmin(SP_XNEW))
        pointXMax = max(np.nanmax(SP_XOLD), np.nanmax(SP_XNEW))
        pointYMin = min(np.nanmin(SP_YOLD), np.nanmin(SP_YNEW))
        pointYMax = max(np.nanmax(SP_YOLD), np.nanmax(SP_YNEW))

        # Load slit position and field data
        SLITPOS = SPL2_rd[3].data
        SPfield_coalign = SPL21[4].data
        SPEXPAND_Br = slitInterp(SPfield_coalign, SLITPOS)
        
        # Load affine matrix from header
        affXform = np.array([
            [update[0].header['WARP00'], update[0].header['WARP01'], update[0].header['WARP02']],
            [update[0].header['WARP10'], update[0].header['WARP11'], update[0].header['WARP12']],
            [0.0, 0.0, 1.0]
            ])

        # Warping function from HMI to SPEXPAND coordinate system
        def fromHMItoSPEXPAND(X):
            return ndimage.affine_transform(denanify(X), affineXYToYX(affXform), 
                    output_shape=SPEXPAND_Br.shape, order=1)

        # Helper functions for visualization
        def plotField(fn, HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop, Tx_center, Ty_center, RecTime, vmin=-3000, vmax=3000):
            plt.figure(figsize=(10,8))
            
            plt.subplot(221)
            plt.pcolormesh(HMIfield_crop, cmap='bwr', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.axis('equal')
            plt.title(f'HMIfield_crop {HMIfield_crop.shape}')
            
            plt.subplot(222)
            plt.pcolormesh(HMIfield_coalign, cmap='bwr', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.axis('equal')
            plt.title(f'HMIfield_coalign {HMIfield_coalign.shape}')
            
            plt.subplot(223)
            plt.pcolormesh(SPfield_coalign, cmap='bwr', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.axis('equal')
            plt.title(f'SPfield_coalign {SPfield_coalign.shape}')
            
            plt.subplot(224)
            plt.pcolormesh(Txy_crop, cmap='jet')
            plt.colorbar()
            plt.axis('equal')
            plt.title('Angular distance to disk center')
            
            plt.suptitle(f'{RecTime} center at ({Tx_center:+.2f},{Ty_center:+.2f})')
            plt.savefig(os.path.join(savefig_dir, fn))
            plt.close()
        
        def saveField(fn, HMIfield_crop, HMIfield_coalign, SPfield_coalign, Txy_crop):
            np.savez(os.path.join(savedata_dir, fn), 
                    HMIfield_crop = HMIfield_crop, 
                    HMIfield_coalign = HMIfield_coalign,
                    SPfield_coalign = SPfield_coalign,
                    Txy_crop = Txy_crop)
        
        def compareField(fn, HMIfield_coalign, SPfield_coalign, RecTime, vmin=-3000, vmax=3000):
            plt.figure(figsize=(14,5))
            HMIfield = HMIfield_coalign.flatten()
            SPfield_coalign = SPfield_coalign.flatten()
            
            plt.subplot(121)
            plt.hist2d(SPfield_coalign, HMIfield, bins=[100, 100], range=[[vmin,vmax], [vmin, vmax]], 
                    cmap='jet', norm=LogNorm(vmin=1e-1))
            plt.plot([vmin,vmax], [vmin,vmax], color='k')
            plt.colorbar()
            plt.xlim([vmin, vmax])
            plt.ylim([vmin, vmax])
            plt.axis('equal')
            plt.xlabel('SPfield_coalign')
            plt.ylabel('HMIfield_coalign')
            
            plt.subplot(122)
            plt.hist(SPfield_coalign,  bins=np.arange(vmin, vmax, 100), fill=False, edgecolor='r', label='SPfield_coalign')
            plt.hist(HMIfield, bins=np.arange(vmin, vmax, 100), fill=False, edgecolor='b', label='HMIfield_coalign')
            plt.yscale('log')
            plt.legend()
            plt.xlabel('Field strength')
            plt.ylabel('COUNTS')
            
            plt.suptitle(RecTime)
            plt.savefig(os.path.join(savefig_dir, fn))
            plt.close()
        
        # Get HMI crop parameters from header
        regDate = update[0].header['PNTDATE']
        cropMinX = update[0].header['BNDMINX']
        cropMinY = update[0].header['BNDMINY']
        cropMaxX = update[0].header['BNDMAXX']
        cropMaxY = update[0].header['BNDMAXY']

        # Load HMI data (ignore .1/.3 suffix)
        HMIFieldMap = sunpy.map.Map(hmiFieldName)
        HMI_HMIfield = HMIFieldMap.data[::-1,::-1]

        # Get arcsec per pixel
        H, W = HMIFieldMap.data.shape[0], HMIFieldMap.data.shape[1]
        HMIX, HMIY = np.meshgrid(np.array(range(W)), np.array(range(H)))
        sc = HMIFieldMap.pixel_to_world(HMIX*u.pix, HMIY*u.pix)
        HMI_Tx = sc.Tx.arcsec[::-1,::-1]
        HMI_Ty = sc.Ty.arcsec[::-1,::-1]

        # Warp HMI data to SPEXPAND coordinate system
        SPEXPAND_HMIBr = fromHMItoSPEXPAND(HMI_HMIfield)
        SPEXPAND_TxRedo = fromHMItoSPEXPAND(HMI_Tx)
        SPEXPAND_TyRedo = fromHMItoSPEXPAND(HMI_Ty)

        # Convert to original coordinate system
        HMIfield_coalign = slitDrop(SPEXPAND_HMIBr, SLITPOS)
        SP_TxRedo = slitDrop(SPEXPAND_TxRedo, SLITPOS)
        SP_TyRedo = slitDrop(SPEXPAND_TyRedo, SLITPOS)
        
        # Calculate distance to solar disk center
        Tx_crop = HMI_Tx[cropMinY:cropMaxY, cropMinX:cropMaxX]
        Ty_crop = HMI_Ty[cropMinY:cropMaxY, cropMinX:cropMaxX]
        Txy_crop = np.sqrt(Tx_crop**2 + Ty_crop**2)
        
        crop_h, crop_w = Tx_crop.shape
        Tx_center = Tx_crop[crop_h//2, crop_w//2]
        Ty_center = Ty_crop[crop_h//2, crop_w//2]

        # Plot and save results
        plotField(f'{RecTime}.png', 
                HMI_HMIfield[cropMinY:cropMaxY, cropMinX:cropMaxX], 
                HMIfield_coalign, 
                SPfield_coalign, 
                Txy_crop, Tx_center, Ty_center,
                RecTime)
        
        saveField(f'{RecTime}.npz', 
                HMI_HMIfield[cropMinY:cropMaxY, cropMinX:cropMaxX], 
                HMIfield_coalign, 
                SPfield_coalign, 
                Txy_crop)
        
        compareField(f'{RecTime}_comparsion.png', 
                    HMIfield_coalign, 
                    SPfield_coalign, 
                    RecTime)
        
        update.close()
        SPL2_rd.close()
        SPL21.close()
    
    # Print missing files summary
    print(f"\n===== Processing Summary =====")
    print(f"Valid pairs processed: {avail_num}")
    print(f"Total pairs in log: {len(match_lines)}")
    if missing_list:
        print(f"\nMissing files ({len(missing_list)}):")
        for idx, missing_file in enumerate(missing_list, 1):
            print(f"  {idx}. {missing_file}")
    else:
        print("\nNo missing files!")