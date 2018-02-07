import numpy as np
import os
import glob

outdir = '../HCP_NPY_Augment'
if not os.path.exists(outdir):
    os.makedirs(outdir)

files = glob.glob("../HCP_NPY/*.npy")
for filepath in files:
    file = np.array(np.load(filepath), dtype = np.float32)
    print('  Data shape is ' + str(file.shape) + ' .')
    for i in range(0, 200):
        x = int(np.floor((file.shape[0] - 64) * np.random.rand(1))[0])
        y = int(np.floor((file.shape[1] - 64) * np.random.rand(1))[0])
        z = int(np.floor((file.shape[2] - 64) * np.random.rand(1))[0])
        file_aug = file[x:x+64, y:y+64, z:z+64]
        filename_ = filepath.split('/')[-1].split('.')[0]
        filename_ = filename_ + '_' + str(i) + '.npy'
        filename = os.path.join(outdir, filename_)
        np.save(filename, file_aug)
    print('All sliced files of ' + filepath + ' are saved.')