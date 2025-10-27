import astropy.io.fits as fits
import astropy.units as u
import sunpy.map 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as ndimage
import os
import sys
import pdb
from matplotlib.colors import LogNorm

def slitToPixLocation(slitpos):
    '''Given the slit position from SOT/SP, return the actual position
    locations of the columns.'''

    #the raw data structure out of astropy.fits means spData[..].data needs
    #to be [0][0]'d
    slitpos = slitpos[0][0]
    slitdiff = np.diff(slitpos)
    normalJump = np.median(slitdiff)
    actualPixLocation = (slitpos-slitpos[0])/normalJump
    return actualPixLocation


def slitDrop(X, slitpos):
    '''Given an image where the columns indicate slit positions, take only the
    actual ones. Makes the image smaller, or slitDrop(X,..).shape[1] <= 
    slitDrop(X,..).shape[1].
    
    X: the image
    slitpos: the slit positions
    
    '''
    actualPixLocation = slitToPixLocation(slitpos)
    return np.hstack([X[:,[int(i)]] for i in actualPixLocation])

def slitInterp(X, slitpos, mode='linear'):
    '''Given an image where the columns indicate slit indices, expand it so that
    the columns indicate slit positions. Makes the image bigger, or 
    slitInterp(X,slitpos).shape[1] >= slitInterp(X,slitpos).shape[1]
   
    X: the image
    slitpos: the slit positions
    mode: 
        'linear' -- linear interpolation (default) 
        'nearest' -- nearest neighbor 
    
    '''
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

if __name__ == '__main__':
    srcUpdate = 'E:/HinodeSOTSPLevel2Update/Main/201401-201404/'
    srcSPL2 = 'E:/Research/Data/HINODE/SP/L2/201401-201404/'
    srcSPL21 = 'E:/Research/Data/HINODE/SP/L2.1/201401-201404/'
    srcHMI = 'E:/Research/Data/SDO/HMI/Magnetogram/720s/201401-201404/' 
    target = 'E:/Research/Work/Super_Resolution_Magnetogram/coalignment/'

    if not os.path.exists(target):
        os.mkdir(target)

    # fnUpdate = RecTime + '.fits'
    # RecTime = '20140102_030739'
    # fnUpdate = RecTime + '.fits'
    # fnSPL2 = RecTime + '.fits'
    # fnSPL21 = RecTime + '_L2.1' + '.fits'
    # # HMI files should be exported by JSOC, 4096*4096
    # fnHMI = 'hmi.M_720s.20140102_031200_TAI.1.magnetogram' + '.fits'
    
    # fnUpdate = RecTime + '.fits'
    # RecTime = '20140102_030739'
    # fnUpdate = RecTime + '.fits'
    # fnSPL2 = RecTime + '.fits'
    # fnSPL21 = RecTime + '_L2.1' + '.fits'
    # # HMI files should be exported by JSOC, 4096*4096
    # fnHMI = 'hmi.M_720s.20140102_031200_TAI.1.magnetogram' + '.fits'
    
    # Sorting file folder
    filesUpdate = get_sorted_files(srcUpdate)
    filesSPL2 = get_sorted_files(srcSPL2)
    filesSPL21 = get_sorted_files(srcSPL21)
    filesHMI = get_sorted_files(srcHMI)
    
    # Iterating throughout all files
    for i_file in range(len(filesSPL2)):
        
        print(f'begin processing No.{i_file+1} files')
        
        fnUpdate = filesUpdate[i_file]
        RecTime = fnUpdate[-20:-5]
        fnSPL2 = filesSPL2[i_file]
        fnSPL21 = filesSPL21[i_file]
        fnHMI = filesHMI[i_file]
        
        update = fits.open(os.path.join(srcUpdate, fnUpdate))
        # print('\nStart Header information')
        # print(repr(update[0].header))
        # print('End header information\n')
        SPL2 = fits.open(os.path.join(srcSPL2, fnSPL2))
        SPL21 = fits.open(os.path.join(srcSPL21, fnSPL21))

        # There are three 'coordinate systems':
        # HMI: the original HMI grid
        # SPEXPAND: the SP data, expanded using slit information so columns are 
        #    slit positions, not indices
        # SP: the SP data, where columns are indices, not positions. This is how
        #    SOTSP is stored, but not how it should be used.

        #OLD X/Y Coordinates
        SP_XOLD = SPL2[38].data
        SP_YOLD = SPL2[39].data

        #Updated X/Y Coordinates
        SP_XNEW = update[38].data
        SP_YNEW = update[39].data

        # For many applications, the above's the only part needed. However, if you 
        # want to see the alignment with HMI, you can run the rest

        pointXMin = min(np.nanmin(SP_XOLD), np.nanmin(SP_XNEW))
        pointXMax = max(np.nanmax(SP_XOLD), np.nanmax(SP_XNEW))
        pointYMin = min(np.nanmin(SP_YOLD), np.nanmin(SP_YNEW))
        pointYMax = max(np.nanmax(SP_YOLD), np.nanmax(SP_YNEW))

        #load the field data; expand it to make it an image
        SLITPOS = SPL2[41].data

        SPfield_coalign = SPL21[4].data
        SPEXPAND_Br = slitInterp(SPfield_coalign, SLITPOS)
        
        #load a 3x3 affine matrix from the header
        affXform = np.array([
            [update[0].header['WARP00'], update[0].header['WARP01'], update[0].header['WARP02']],
            [update[0].header['WARP10'], update[0].header['WARP11'], update[0].header['WARP12']],
            [0.0, 0.0, 1.0]
            ])

        #do a warping from HMI to the SPEXPAND coordinate system
        def fromHMItoSPEXPAND(X):
            return ndimage.affine_transform(denanify(X), affineXYToYX(affXform), 
                    output_shape=SPEXPAND_Br.shape, order=1)

        #helper functions for visualization
        def plotField(fn, HMIfield_crop, HMIfield_coalign, SPfield_coalign, RecTime, vmin=-3000, vmax=3000):
            plt.figure(figsize=(16,5))
            
            plt.subplot(131)
            plt.pcolormesh(HMIfield_crop, cmap='bwr', vmin=vmin, vmax=vmax)
            plt.axis('equal')
            plt.title(f'HMIfield_crop {HMIfield_crop.shape}')
            
            plt.subplot(132)
            plt.pcolormesh(HMIfield_coalign, cmap='bwr', vmin=vmin, vmax=vmax)
            plt.axis('equal')
            plt.title(f'HMIfield_coalign {HMIfield_coalign.shape}')
            
            plt.subplot(133)
            plt.pcolormesh(SPfield_coalign, cmap='bwr', vmin=vmin, vmax=vmax)
            plt.axis('equal')
            plt.title(f'SPfield_coalign {SPfield_coalign.shape}')
            
            plt.suptitle(RecTime)
            plt.savefig(os.path.join(target+'SaveFig/', fn))
            plt.close()
        
        def saveField(fn, HMIfield_crop, HMIfield_coalign, SPfield_coalign):
            np.savez(os.path.join(target+'SaveData/', fn), 
                    HMIfield_crop = HMIfield_crop, 
                    HMIfield_coalign = HMIfield_coalign,
                    SPfield_coalign = SPfield_coalign)
        
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
            plt.savefig(os.path.join(target+'SaveFig/', fn))
            plt.close()
        
        #get the HMI Field, X_COORDINATE, Y_COORDINATE 
        regDate = update[0].header['PNTDATE']
        cropMinX = update[0].header['BNDMINX']
        cropMinY = update[0].header['BNDMINY']
        cropMaxX = update[0].header['BNDMAXX']
        cropMaxY = update[0].header['BNDMAXY']

        hmiFieldName = os.path.join(srcHMI, fnHMI)
        if not os.path.exists(hmiFieldName):
            print("Can't find hmi file %s" % hmiFieldName)
            print("You'll need the HMI file from which the pointing was derived")
            print('Please export hmi.B_720s[%s] from ' % regDate)
            print('  http://jsoc.stanford.edu/ajax/exportdata.html')
            sys.exit(1)

        HMIFieldMap = sunpy.map.Map(hmiFieldName)
        HMI_HMIfield = HMIFieldMap.data[::-1,::-1]

        #get the arcsec info per-pixel, flipping to account for the fact that
        #the transformation is from the flipped HMI system
        H, W = HMIFieldMap.data.shape[0], HMIFieldMap.data.shape[1]
        HMIX, HMIY = np.meshgrid(np.array(range(W)), np.array(range(H)))
        sc = HMIFieldMap.pixel_to_world(HMIX*u.pix, HMIY*u.pix)
        HMI_Tx = sc.Tx.arcsec[::-1,::-1]
        HMI_Ty = sc.Ty.arcsec[::-1,::-1]

        #warp them to the SP Expanded coordinate system
        #Note that *all* affine transformations refer to a map from HMI (flipped)
        #to SOTSP (expanded so columns are positions, not indices)
        SPEXPAND_HMIBr = fromHMItoSPEXPAND(HMI_HMIfield)
        SPEXPAND_TxRedo = fromHMItoSPEXPAND(HMI_Tx)
        SPEXPAND_TyRedo = fromHMItoSPEXPAND(HMI_Ty)

        #Go to the original, packaged coordinate system
        HMIfield_coalign = slitDrop(SPEXPAND_HMIBr, SLITPOS)
        SP_TxRedo = slitDrop(SPEXPAND_TxRedo, SLITPOS)
        SP_TyRedo = slitDrop(SPEXPAND_TyRedo, SLITPOS)

        #Plot and save the images
        saveField(RecTime + '.npz', 
                HMI_HMIfield[cropMinY:cropMaxY, cropMinX:cropMaxX], 
                HMIfield_coalign, 
                SPfield_coalign)
        
        plotField(RecTime + '.png', 
                HMI_HMIfield[cropMinY:cropMaxY, cropMinX:cropMaxX], 
                HMIfield_coalign, 
                SPfield_coalign, 
                RecTime)
        
        compareField(RecTime + '_comparsion.png', 
                    HMIfield_coalign, 
                    SPfield_coalign, 
                    RecTime)
        
        # plt.show()

   
