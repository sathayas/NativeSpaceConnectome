import os
import numpy as np
import nibabel as nib
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import nipype.interfaces.spm as spm # importing SPM interface functions
import nipype.interfaces.fsl as fsl # importing FSL interface functions
from nipype import Node, MapNode, Workflow  # components to construct workflow
from nipype.interfaces.io import DataSink  # datasink
from nipype.algorithms.misc import Gunzip  # gunzip interface


##### Parameters and such
# Sites where rs-fMRI data originate
sites = ['Berlin_Margulies',
         'Leiden_2200',
         'Newark',
         'NewYork_b',
         'Oxford',
         'Queensland'
         ]
# kernel size for erosion / dilation
k_size = 1
# threshold for binarization
bTh = 0.6
# smoothing kernel witdh
smoFWHM = 6.0
# Directory where resting-state raw data reside
dataDir = '/home/satoru/Projects/Connectome/Data/1000FCP'
#dataDir = '/Users/sh45474/Documents/Research/Project/NativeSpaceConnectome/Data'
# template (it has to be tissue probability maps)
fTPM = '/usr/local/spm12/tpm/TPM.nii'
#fTPM = '/Users/sh45474/SoftwareTools/spm12/tpm/TPM.nii'
# brain mask in MNI space (from FSL)
fmask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz'
# Output directory (base)
outDirBase = '/home/satoru/Projects/NativeSpaceConnectome/ProcessedData'
#outDirBase = '/Users/sh45474/Documents/Research/Project/NativeSpaceConnectome/ProcessedData'


##### Choosing a single subject as to test the pipeline
# picked a site
iSite = sites[0]

# getting a list of subject at this site
dirSite = os.path.join(dataDir,iSite)
listFiles = os.listdir(os.path.join(dirSite,'Raw'))
listSubj = [s for s in listFiles if s[-5:].isnumeric()]

# picked a random subject
iSubj = listSubj[-1]

# and the directories for that subject
dirSubj = os.path.join(dirSite,'Raw',iSubj)
dirSubjFunc = os.path.join(dirSubj,'func')
dirSubjAnat = os.path.join(dirSubj,'anat')


# rs-fMRI image from the selected subject
imagefMRI = os.path.join(dirSubjFunc,'rest.nii.gz')
# voxel size for the fMRI data
voxfMRI = list(nib.load(imagefMRI).header['pixdim'][1:4].astype(float))

# an T1 image for the same subject
imageT1 = os.path.join(dirSubjAnat,'mprage_anonymized.nii.gz')



##### Creating the output directory
# directory for the site
outDirSite = os.path.join(outDirBase, iSite)
# if the directory doesn't exist, create it
if not os.path.exists(outDirSite):
    os.makedirs(outDirSite)

# directory for the subject
outDirSubj = os.path.join(outDirSite, iSubj)
# if the directory doesn't exist, create it
if not os.path.exists(outDirSubj):
    os.makedirs(outDirSubj)

# finally, output directory for the results in Native space
outDir = outDirSubj
# if the directory doesn't exist, create it
if not os.path.exists(outDir):
    os.makedirs(outDir)



#
#    T1-related nodes
#
# gunzip node
gunzip_T1w = Node(Gunzip(in_file=imageT1),
                  name="gunzip_T1w")

# Segmentation, native space
segNative = Node(spm.NewSegment(),
                 name='segNative')




#
#   fMRI pre-processing
#
# skip dummy scans
extract = Node(fsl.ExtractROI(in_file=imagefMRI,  # input image
                              t_min=4,            # first 4 volumes are deleted
                              t_size=-1,
                              output_type='NIFTI'),  # forces output to be .nii
               name="extract")

# motion correction aka realignment
realign = Node(spm.Realign(),
               name="realign")




#
#    Tissue map nodes
#
# coregistration, tissue images (native) to fMRI (native)
coreg = MapNode(spm.Coregister(cost_function='nmi',
                               jobtype='estwrite',
                               write_interp=0),
                name="coreg",
                iterfield=['apply_to_files'],
                nested=True)

# smoothing
smooth = MapNode(fsl.maths.IsotropicSmooth(fwhm=smoFWHM),
                 name='smooth',
                 iterfield=['in_file'],
                 nested=True)

# thresholding
thresh = MapNode(fsl.maths.Threshold(thresh=bTh),
                 name='thresh',
                 iterfield=['in_file'],
                 nested=True)

# filling in the holes, with fslmaths
fillHoles = MapNode(fsl.maths.MathsCommand(args='-fillh'),
                    name='fillHoles',
                    iterfield=['in_file'],
                    nested=True)

# dilation with fslmaths
dilate = MapNode(fsl.maths.DilateImage(operation='max',
                                        kernel_shape='boxv',
                                        kernel_size=k_size),
                  name='dilate',
                  iterfield=['in_file'],
                  nested=True)

# erosion with fslmaths
erode = MapNode(fsl.maths.ErodeImage(kernel_shape='boxv',
                                      kernel_size=k_size),
                 name='erode',
                 iterfield=['in_file'],
                 nested=True)

# binarization, with fslmaths
binarize = MapNode(fsl.maths.MathsCommand(args='-bin'),
                    name='binarize',
                    iterfield=['in_file'],
                    nested=True)

# erosion to identify deep tissue with fslmaths
erodeDeep = MapNode(fsl.maths.ErodeImage(),
                    name='erodeDeep',
                    iterfield=['in_file'],
                    nested=True)


# DataSink to collect outputs
datasink = Node(DataSink(base_directory=outDir),
                name='datasink')


# creating a workflow
Native = Workflow(name="Native", base_dir=outDir)

# connecting the nodes to the main workflow
Native.connect(extract, 'roi_file', realign, 'in_files')
Native.connect(gunzip_T1w, 'out_file', segNative, 'channel_files')
Native.connect(gunzip_T1w, 'out_file', coreg, 'source')
Native.connect(realign, 'mean_image', coreg, 'target')
Native.connect(segNative, 'native_class_images', coreg, 'apply_to_files')
Native.connect(coreg, 'coregistered_files', dilate, 'in_file')
Native.connect(dilate, 'out_file', fillHoles, 'in_file')
Native.connect(fillHoles, 'out_file', erode, 'in_file')
Native.connect(erode, 'out_file', smooth, 'in_file')
Native.connect(smooth, 'out_file', thresh, 'in_file')
Native.connect(thresh, 'out_file', binarize, 'in_file')
Native.connect(coreg, 'coregistered_files', erodeDeep, 'in_file')

# connections to the datasink
Native.connect(realign, 'realignment_parameters',
                    datasink, 'DerivativesNative.@mcPar')
Native.connect(realign, 'mean_image',
                    datasink, 'DerivativesNative.@mean_fMRI')
Native.connect(realign, 'realigned_files',
                    datasink, 'DerivativesNative.@moCorfMRI')
Native.connect(coreg, 'coregistered_files',
                    datasink, 'DerivativesNative.@tissueSeg')
Native.connect(binarize, 'out_file',
                    datasink, 'DerivativesNative.@binaryGMMask')
Native.connect(erodeDeep, 'out_file',
                    datasink, 'DerivativesNative.@deepWMMask')





# running the workflow
Native.run()
