import numpy as np
import os
import glob
from PIL import Image
import sys
sys.path.append('../')
  
def get_all_images():
    filepaths = glob.glob('../../HCP_NPY/*.npy')
    images = []
    for filename in filepaths:
        img_npy = np.load(filename)
        img = np.array(img_npy, dtype = np.float32)
        images.append(img)
    print("All images are saved in images, shape " + str(np.shape(images)))
    return images

# Get all images from the NPY files
images = get_all_images()

def getLR(hr_data):
    imgfft = np.fft.fftn(hr_data)
    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2
    imgfft[x_center-20 : x_center+20, y_center-20 : y_center+20, z_center-20 : z_center+20] = 0
    imgifft = np.fft.ifftn(imgfft)
    img_out = abs(imgifft)

    return img_out

'''  
# TODO Figure out how to perform FFT in python
for image in images:
    image -= np.amin(image)
    image /= np.amax(image)
    d = image.shape[0]
    for i in range(100, 101):
        img = image[i, :, :]
        origin_name = './' + str(i) + '.png'
        save_name = './' + str(i) + '_ifft.png'

        imgfft = np.fft.fft2(img)
        x_center = imgfft.shape[0] // 2
        y_center = imgfft.shape[1] // 2

        imgfft[x_center-70 : x_center+70, y_center-50 : y_center+50] = 0

        imgifft = np.fft.ifft2(imgfft)
        imgifft = np.real(imgifft)
        imgifft -= np.amin(imgifft)
        imgifft /= np.amax(imgifft)
        imgifft *= 255        

        img -= np.amin(img)
        img /= np.amax(img)
        img *= 255
        # [100:150, 100:150]
        Image.fromarray(img.round().astype(np.uint8)).save(origin_name, 'PNG', dpi=[300,300], compression_level = 0)

        Image.fromarray(imgifft.round().astype(np.uint8)).save(save_name, 'PNG', dpi=[300,300], compression_level = 0)
        print(ssim.get_ssim(img, imgifft).mean())

# TODO perform truncating op
# ...
# TODO perform iFFT and get the LR images
# ...
'''