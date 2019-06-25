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

# finally, output directory for the results in MNI space
outDir = outDirSubj
# if the directory doesn't exist, create it
if not os.path.exists(outDir):
    os.makedirs(outDir)



#
#    T1 Normalization nodes
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


# coregistration, T1w (native) to fMRI (native)
coreg = Node(spm.Coregister(cost_function='nmi'),
                name="coreg")



# creating a workflow
MNI = Workflow(name="MNI", base_dir=outDir)

# connecting the nodes to the main workflow
MNI.connect(extract, 'roi_file', realign, 'in_files')
MNI.connect(gunzip_T1w, 'out_file', segNative, 'channel_files')
MNI.connect(gunzip_T1w, 'out_file', coreg, 'source')
MNI.connect(realign, 'mean_image', coreg, 'target')
MNI.connect(segNative, 'native_class_images', coreg, 'apply_to_files')


# running the workflow
MNI.run()
