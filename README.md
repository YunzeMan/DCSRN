# DCSRN-Tensorflow
version 0.1.0 - Feb 7 2018 - by Yunze MAN

This is an implementation of [Brain MRI Super Resolution Using 3D Deep Densely Connected Neural Networks](https://arxiv.org/pdf/1801.02728.pdf) with Tensorflow. The paper is based on [Densely connected convolutional networks](https://arxiv.org/pdf/1608.06993.pdf), it takes a low resolution 3D MRI image, performs a super resolution process and then output the high resolution image. This repository is only on test phase right now, any contributions helping with bugs and compatibility issues are welcomed.

- [DCSRN-Tensorflow](#dcsrn-tensorflow)
    - [1. Introduction](#1-introduction)
    - [2. Installation](#2-installation)
        - [2.1 Prerequisites](#21-prerequisites)
        - [2.2 Requirements](#22-requirements)
    - [3. Usage](#3-usage)
        - [3.1 Data preparation](#31-data-preparation)
        - [3.2 Run the code](#32-run-the-code)
    - [4. Visualization](#4-visualization)
    - [5. Questions on paper](#5-questions-on-paper)
    - [6. Contact Information](#6-contact-information)

## 1. Introduction

![Imgur](https://imgur.com/ECl1t1T.png)
This repo DCSRN is a code package of paper [Brain MRI Super Resolution Using 3D Deep Densely Connected Neural Networks](https://arxiv.org/pdf/1801.02728.pdf) The paper propose a 3D Densely Connected Super-Resolution Networks (DCSRN), derived from [Densely connected convolutional networks](https://arxiv.org/pdf/1608.06993.pdf). After trainning on the public dataset HCP, this method achieves a better performance in terms of **SSIM**, **PSNR** and **NRMSE**. So far, the implementation only support SSIM performance.
The encapsulation structure of this project follows [tf-unet](https://github.com/jakeret/tf_unet), which is well designed and easy to understand. 

## 2. Installation
#### 2.1 Prerequisites
###### 2.1.1 Please make sure that your machine is equipped with modern GPUs that support CUDA.
    Modern GPU allow training and testing phases to execute 50x times faster or more.
###### 2.1.2 Please make sure that python (3.6 is recommended) is installed 
    Insufficient compatibility guaranteed for Python 2
**Anaconda3 is recommended.**

#### 2.2 Requirements
###### 2.2.1 Please refer to `requirements.txt`
    conda install numpy nibabel glob Pillow matplotlib
Anaconda is recommended to create an environment for the project to run with.
###### 2.2.2 Please install new version of tensorflow
    There are new features of Tensorflow in the current and coming codes, 
    so a relatively new version of tensorflow is needed.

## 3. Usage

#### 3.1 Data preparation
###### 3.1.1 Download HCP data from http://www.humanconnectomeproject.org/data/
    You may want to sign up to get the permission to the public dataset first.
    There are several datasets provided by HCP,
        Smaller ones contain 35 subjects' MRI
        Larger ones contains 1113 subjects' MRI.
    Either one is OK to use, just pay attention to the volume of data

###### 3.1.2 Use code to transform NII data into NPY format
    HCP dataset has the format
        $HCP/mgh_$NUMBER/A_LONG_CODE/$TIME_STRING/A_NUMBER/Filename.nii
    Put nii2npy.py under $HCP and run: 
        python nii2npy.py
    You are suppose to get HCP_NPY/ in the same dir with $HCP

###### 3.1.3 Do data Augmentation
    Put data_augment.py under $HCP and run:
        python data_augment.py
    You are suppose to get HCP_NPY_Augment/ in the same dir with $HCP
This data augmentation process generates 100 64x64x64 volumes from each subject's 3D MRI. According to the paper, the patches are generated randomly in the image.

#### 3.2 Run the code
    cd to DCSRN/ and run:
        python dcsrn_test.py
    snapshots will be saved at /DCSRN/snapshots/

## 4. Visualization
**Original Slice:**

![Imgur](https://i.imgur.com/fN6vaN2.png)

**After FFT:**

![Imgur](https://imgur.com/AWn6gGA.png)

## 5. Questions on paper
1. The paper use a low-pass filtering-like method to generate low resolution images, the method meet with the nature of MRI, which makes it valid, but paper also mentions that this method avoid checkerboard artifact. However, I think since DenseNet does not involve deconvolution process, there is nothing to do with checkerboard artifact in this process. Not sure if I'm right.
2. The second question is about the graph in the paper. The bypass-like links don't concat before bn+elu+conv units, but concat after them. This is not consistent with the original DenseNet, I don't see why paper use this way of connection. I view it as a minor typo and my implementation follows the DenseNet's way of connection.  

## 6. Contact Information

If you encounter any problems in using these codes, please open an issue in this repository.
You may also contact Yunze MAN (yunze.man97@gmail.com).

Thanks for your interest! Have fun!