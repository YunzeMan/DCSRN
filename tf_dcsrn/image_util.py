'''
author: MANYZ
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import sys
sys.path.append("../")
import glob
import numpy as np
from PIL import Image
from FFT import LRbyFFT


class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 1
    n_class = 2
    

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_hrlr_data(self):
        hr_data = self._next_data()
            
        hr_data = self._process_data(hr_data)
        lr_data = self._get_lrdata(hr_data)
        
        hr_data, lr_data = self._post_process(hr_data, lr_data)
        
        ny = hr_data.shape[0]
        nx = hr_data.shape[1]
        nz = hr_data.shape[2]

        return hr_data.reshape(1, ny, nx, nz, self.channels), lr_data.reshape(1, ny, nx, nz, self.channels),
    
    def _get_lrdata(self, hr_data):
        lr_data = LRbyFFT.getLR(hr_data)
        return lr_data
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, hr_data, lr_data):
        """
        Post processing hook that can be used for data augmentation
        
        :param hr_data: the high resolution data array
        :param lr_data: the low resolution label array
        """
        return hr_data, lr_data
    
    def __call__(self, n):
        hr_data, lr_data = self._load_hrlr_data()
        nx = hr_data.shape[1]
        ny = hr_data.shape[2]
        nz = hr_data.shape[3]
    
        X = np.zeros((n, nx, ny, nz, self.channels))
        Y = np.zeros((n, nx, ny, nz, self.channels))
    
        X[0] = hr_data
        Y[0] = lr_data
        for i in range(1, n):
            hr_data, lr_data = self._load_hrlr_data()
            X[i] = hr_data
            Y[i] = lr_data
    
        return Y, X

class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/1.tif' and 'train/1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("../train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif', shuffle_data=True, n_class = 2):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        
        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
    
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        
        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)
    
        return img,label


class MedicalImageDataProvider(BaseDataProvider):
    """
    Generic data provider for high resolution and low resolution images, supports gray scale.
    Assumes that the high resolution images are stored in the same folder
    and images are of the same shape
    e.g. 'HCP_mgh_1035_MR_MPRAGE_GradWarped_and_Defaced_Br_20140919135823853_S227866_I444361_9.npy'

    Usage:
    data_provider = MedicalImageDataProvider("../../HCP_NPY_Augment/*.npy")
    
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    """
    
    def __init__(self, search_path = '../HCP_NPY_Augment/*.npy', a_min=None, a_max=None, shuffle_data=True):
        super(MedicalImageDataProvider, self).__init__(a_min, a_max)
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.channels = 1

        self.data_files = self._find_data_files(search_path)

        assert len(self.data_files) > 0, "No training files"
        print("Number of 3D files used: %s" % len(self.data_files))

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files]
    
    def _load_file(self, path, dtype=np.float32):
       return np.array(np.load(path), dtype)

    def _increment_fileidx(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
        
    def _next_data(self):
        self._increment_fileidx()
        image_name = self.data_files[self.file_idx]
        img = self._load_file(image_name, np.float32)
        while np.amax(img) <= 0:
            self._increment_fileidx()
            image_name = self.data_files[self.file_idx]
            img = self._load_file(image_name, np.float32)

        return img