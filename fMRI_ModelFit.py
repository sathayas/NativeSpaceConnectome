import os
import numpy as np
import pandas as pd
import nibabel as nib


# function to get sole image file name from the directory
def fname_nii(dirBase):
    listFiles = os.listdir(dirBase)
    niiFiles = [x for x in listFiles if '.nii' in x]
    return niiFiles[0]


#
# Inputs:
#    -fMRI (preprocessed)
#    -Node mask (dilated gray matter mask)
#    -GM segmentation image
#    -WM segmentation image
#    -CSF segmentation image
#    -Deep white matter mask
#    -Motion parameters
#


# Inputs - image files
dirPreprocBase = '/home/satoru/Projects/NativeSpaceConnectome/ProcessedData/Berlin_Margulies/sub77281/DerivativesNative'
ffMRI = os.path.join(dirPreprocBase,'rrest_roi.nii')
fNodeMask = os.path.join(os.path.join(dirPreprocBase,'_binarize0'),
                         fname_nii(os.path.join(dirPreprocBase,'_binarize0')))
fGMMask = os.path.join(os.path.join(dirPreprocBase,'_coreg0'),
                       fname_nii(os.path.join(dirPreprocBase,'_coreg0')))
fWMMask = os.path.join(os.path.join(dirPreprocBase,'_coreg1'),
                       fname_nii(os.path.join(dirPreprocBase,'_coreg1')))
fCSFMask = os.path.join(os.path.join(dirPreprocBase,'_coreg2'),
                        fname_nii(os.path.join(dirPreprocBase,'_coreg2')))
fDeepWMMask = os.path.join(os.path.join(dirPreprocBase,'_erodeDeep1'),
                           fname_nii(os.path.join(dirPreprocBase,'_erodeDeep1')))
fMoPar = os.path.join(dirPreprocBase,'rp_rest_roi.txt')



# Parameters



#
# Steps:
#    -Masking preprocessed fMRI
#    -Band-pass filtering
#    -Calculate global signals
#       -Brain parenchyma
#           -GM + WM image
#           -Threshold and binarize
#           -Mask and extract mean
#       -Deep WM
#           -Mask and extract mean
#       -CSF
#           -Threshold and binarize
#           -Mask and extract mean
#    -Regressing out global and motion
#       -Band-pass filtering of motion, global signals
#       -Actual regressing out
#    -Motion scrubbing
#       -Calculate FD
#       -Identify frames to be removed
#       -Remove frames & reconstruct time series
#
