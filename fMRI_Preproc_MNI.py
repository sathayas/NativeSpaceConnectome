import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nipype.interfaces.spm as spm # importing SPM interface functions
import nipype.interfaces.fsl as fsl # importing FSL interface functions
from nipype import Node, Workflow  # components to construct workflow
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
# template (it has to be tissue probability maps)
fTPM = '/usr/local/spm12/tpm/TPM.nii'
# brain mask in MNI space (from FSL)
fmask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz'
# Output directory (base)
outDirBase = '/home/satoru/Projects/NativeSpaceConnectome/ProcessedData'



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

# an T1 image for the same subject
imageT1 = os.path.join(dirSubjFunc,'mprage_skullstripped.nii.gz')



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
outDir = os.path.join(outDirSubj,'MNI')
# if the directory doesn't exist, create it
if not os.path.exists(outDir):
    os.makedirs(outDir)



#
#    T1 Normalization nodes
#
# gunzip node
gunzip_T1w = Node(Gunzip(in_file=imageT1),
              name="gunzip_T1w")

# Normalize - normalizes structural images to the MNI template
normalize = Node(spm.Normalize12(jobtype='estwrite',
                                 tpm=fTPM,
                                 write_bounding_box=[[-90, -120, -70],
                                                     [90, 90, 105]]),
                 name="normalize")





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

# co-registration node
coreg = Node(spm.Coregister(cost_function='nmi',
                            jobtype='estimate'),
             name="coreg")

# smoothing node
smooth = Node(spm.Smooth(fwhm=[6,6,6]),
              name='smooth')

# gunzip node, FSL brain mask
gunzip_mask = Node(Gunzip(in_file=fmask),
                   name="gunzip_mask")

# Reslice the FSL template to match fMRI
reslice = Node(spm.utils.Reslice(),  # FSL mask image needs to be resliced
               name='reslice')

# masking the fMRI with a brain mask
applymask = Node(fsl.ApplyMask(),
                 name='applymask')

# DataSink to collect outputs
datasink = Node(DataSink(base_directory=outDir),
                name='datasink')

# creating a workflow
preprocfMRI = Workflow(name="PreprocfMRI", base_dir=outDir)

# connecting the nodes to the main workflow
preprocfMRI.connect(extract, 'roi_file', realign, 'in_files')
preprocfMRI.connect(, 'roi_file', realign, 'in_files')


preprocfMRI.connect([(extract, realign, [('roi_file', 'in_files')])])
preprocfMRI.connect([(gunzip_T1w, coreg, [('out_file', 'target')])])
preprocfMRI.connect([(realign, coreg, [('mean_image', 'source')])])
preprocfMRI.connect([(realign, coreg, [('realigned_files', 'apply_to_files')])])
preprocfMRI.connect([(gunzip_T1w, normalize, [('out_file', 'image_to_align')])])
preprocfMRI.connect([(coreg, normalize, [('coregistered_files', 'apply_to_files')])])
preprocfMRI.connect([(normalize, smooth, [('normalized_files', 'in_files')])])
preprocfMRI.connect([(smooth, reslice, [('smoothed_files', 'space_defining')])])
preprocfMRI.connect([(gunzip_mask, reslice, [('out_file', 'in_file')])])
preprocfMRI.connect([(reslice, applymask, [('out_file', 'mask_file')])])
preprocfMRI.connect([(smooth, applymask, [('smoothed_files', 'in_file')])])


# connections to the datasink
preprocfMRI.connect([(realign, datasink, [('realignment_parameters',
                                           'Combo_Preproc_fMRI.@mcPar')])])
preprocfMRI.connect([(normalize, datasink, [('normalized_image',
                                             'Combo_Preproc_fMRI.@T1_standard')])])
preprocfMRI.connect([(normalize, datasink, [('normalized_files',
                                             'Combo_Preproc_fMRI.@fMRI_standard')])])
preprocfMRI.connect([(applymask, datasink, [('out_file',
                                             'Combo_Preproc_fMRI.@SmoothfMRI_standard')])])

# writing out graphs
preprocfMRI.write_graph(graph2use='orig', dotfilename='graph_orig.dot')

# showing the graph
plt.figure(figsize=[10,6])
img=mpimg.imread(os.path.join(outDir,"PreprocfMRI_combo","graph_orig_detailed.png"))
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()

# running the workflow
preprocfMRI.run()
