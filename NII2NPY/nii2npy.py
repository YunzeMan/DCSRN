import numpy as np
import os
import glob
import nibabel

outdir = '../HCP_NPY'
if not os.path.exists(outdir):
    os.makedirs(outdir)

files = glob.glob("../HCP/*/*/*/*/*.nii")
for filepath in files:
    file = nibabel.load(filepath).get_data().transpose(1, 0, 2)
    print('  Data shape is ' + str(file.shape) + ' .')
    filename_ = filepath.split('/')[-1].split('.')[0]
    filename = filename_ + '.npy'
    file1 = os.path.join(outdir, filename)
    np.save(file1, file)
    print('File ' + filename_ + ' is saved in ' + file1 + ' .')

