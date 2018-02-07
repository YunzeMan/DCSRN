from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import logging

import tensorflow as tf

from tf_dcsrn import dcsrn, util, image_util
from data_util.parse_annotations import *
from data_util.image_helper import *

output_path = "./snapshots/"
# path of dataset, here is the HCP dataset
dataset_NIH = "/home/mk/Data/HCP_NPY_Augment/"

#preparing data loading, you may want to explicitly note the glob search path on you data 
data_provider = image_util.MedicalImageDataProvider()

# setup & training
net = dcsrn.DCSRN(channels=1)
trainer = dcsrn.Trainer(net)
path = trainer.train(data_provider, output_path)

print("\nTraining process is over.\n")

# verification, randomly test 4 images
test_provider = image_util.MedicalImageDataProvider()
test_x, test_y = test_provider(4)
result = net.predict(path, test_x)