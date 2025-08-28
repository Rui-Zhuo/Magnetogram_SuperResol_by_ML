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

if __name__ == '__main__':
    srcUpdate = 'E:/HinodeSOTSPLevel2Update/Main/'
    srcSPL2 = 'E:/Research/Data/HINODE/SP/L2/'
    srcSPL21 = 'E:/Research/Data/HINODE/SP/L2.1/'
    srcHMI = 'E:/Research/Data/SDO/HMI/Magnetogram/720s/'
    
    RecTime = '20140101_032208'
    fnUpdate = RecTime + '.fits'
    fnSPL2 = RecTime + '.fits'
    fnSPL21 = RecTime + '_L2.1' + '.fits'
    # HMI files should be exported by JSOC, 4096*4096
    fnHMI = 'hmi.M_720s.20140101_032400_TAI.1.magnetogram.fits'

    target = 'SP21_HMI_coalignment/'

    if not os.path.exists(target):
        os.mkdir(target)

    update = fits.open(os.path.join(srcUpdate, fnUpdate))
    print('\nStart Header information')
    print(repr(update[0].header))
    print('End header information\n')
    SPL2 = fits.open(os.path.join(srcSPL2, fnSPL2))
    SPL21 = fits.open(os.path.join(srcSPL21, fnSPL21))

    #There are three 'coordinate systems':
    #HMI: the original HMI grid
    #SPEXPAND: the SP data, expanded using slit information so columns are 
    #   slit positions, not indices
    #SP: the SP data, where columns are indices, not positions. This is how
    #   SOTSP is stored, but not how it should be used.

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

    SP_Br = SPL21[4].data
    SPEXPAND_Br = slitInterp(SP_Br, SLITPOS)

    
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
    def savePointX(fn, X): plt.imsave(os.path.join(target, fn), X, 
                                vmin=pointXMin, vmax=pointXMax, cmap='tab20c')
    def savePointY(fn, X): plt.imsave(os.path.join(target, fn), X, 
                                vmin=pointYMin, vmax=pointYMax, cmap='tab20b')
    def saveField(fn, X): plt.imsave(os.path.join(target, fn), X**0.5, 
                                vmin=0, vmax=3000**0.5, cmap='plasma')
    def saveBr(fn, X): plt.imsave(os.path.join(target, fn), X, 
                                vmin=-3000, vmax=3000, cmap='bwr')

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
    HMI_Br = HMIFieldMap.data[::-1,::-1]

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
    SPEXPAND_HMIBr = fromHMItoSPEXPAND(HMI_Br)
    SPEXPAND_TxRedo = fromHMItoSPEXPAND(HMI_Tx)
    SPEXPAND_TyRedo = fromHMItoSPEXPAND(HMI_Ty)

    #Go to the original, packaged coordinate system
    SP_HMIBr = slitDrop(SPEXPAND_HMIBr, SLITPOS)
    SP_TxRedo = slitDrop(SPEXPAND_TxRedo, SLITPOS)
    SP_TyRedo = slitDrop(SPEXPAND_TyRedo, SLITPOS)


    ###
    #Save the images
    ###

    #the HMI data, before it's been warped. The SOTSP data should look like 
    #this, but with aspect ratio stretched a little and some non-rigid 
    #deformation due to evolution during acquisition
    saveBr('HMICROP_HMIField.png', HMI_Br[cropMinY:cropMaxY, cropMinX:cropMaxX])
    savePointX('HMICROP_Tx.png', HMI_Tx[cropMinY:cropMaxY, cropMinX:cropMaxX])
    savePointY('HMICROP_Ty.png', HMI_Ty[cropMinY:cropMaxY, cropMinX:cropMaxX])

    #Both are on the same grid, so they should look the same, modulo the 
    #non-rigid deformation during acquisition
    saveBr('SP_HMIField.png', SP_HMIBr)
    saveBr('SP_FIELD.png', SP_Br)

    #this is the x coordinate info from HMI, warped to SOTSP
    savePointX('SP_TxRedo.png', SP_TxRedo)
    savePointY('SP_TyRedo.png', SP_TyRedo)

    #this is the old x coordinate. These will be different. Notice that the start
    #and end of X/Y will be different than TxRedo
    savePointX('SP_XOLD.png', SP_XOLD)
    savePointY('SP_YOLD.png', SP_YOLD)

    #these are the new x coordinates. By construction, they match HMI's.
    savePointX('SP_XNEW.png', SP_XNEW)
    savePointY('SP_YNEW.png', SP_YNEW)
   
