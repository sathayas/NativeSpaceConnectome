import os
import numpy as np
import nibabel as nib


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


#
# Steps:
#    -Masking preprocessed fMRI
#    -Band-pass filtering
#    -Calculate global signals
#       -Brain parenchyma
#           -GM + WM image
#           -Mask and extract mean
#       -Deep WM
#           -Mask and extract mean
#       -CSF
#           -Mask and extract mean
#    -Regressing out global and motion
#       -Band-pass filtering of motion, global signals
#       -Actual regressing out
#    -Motion scrubbing
#       -Calculate FD
#       -Identify frames to be removed
#       -Remove frames & reconstruct time series
#
