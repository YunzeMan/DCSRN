from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

from tf_dcsrn import dcsrn, util, image_util
from data_util.parse_annotations import *
from data_util.image_helper import *


output_path = "./snapshots/"
# NIH数据集位置 path of other PASCAL VOC dataset, if you want to train with 2007 and 2012 train datasets 
dataset_NIH = "/home/mk/Data/NIH/PROCESSED_DATA/2D"
# 存放所有bounding box的位置
dataset_BB = "/home/mk/Data/NIH/PROCESSED_DATA/BB"
# 存放所有mask的位置
dataset_LABEL = "/home/mk/Data/NIH/PROCESSED_DATA/LABELS"



images = get_all_images(dataset_NIH)
annotations = get_all_annotations_mask(dataset_LABEL)
bbox = get_all_annotations_bb(dataset_BB)

data_num = len(images) 

#preparing data loading
data_provider = image_util.MedicalImageDataProvider(images, annotations, bbox)

#setup & training
net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
trainer = unet.Trainer(net)
# path = trainer.train(data_provider, output_path, training_iters=32, epochs=100)
path = trainer.train(data_provider, output_path, training_iters=data_num, epochs=10)

#verification 先随便test四张
test_provider = image_util.MedicalImageDataProvider(images, annotations)
test_x, test_y = test_provider(4)
prediction = net.predict(path, test_x)

unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))

img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "prediction.jpg")