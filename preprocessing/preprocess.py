"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# taken and/or modified from # https://github.com/facebookresearch/fastMRI/tree/main/fastmri/data 
import os
from pathlib import Path
from typing import Dict
import h5py
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as etree
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.utils.data as torch_data
import fastmri
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms, mri_data
from fastmri.data.mri_data import et_query

def get_data(path_data):
    # open all HDF5 files
    hf = h5py.File(path_data)
    # get MRI k-space
    volume_kspace = hf['kspace'][()]
    # get ismrmrd header
    et_root = etree.fromstring(hf["ismrmrd_header"][()])
    return volume_kspace, et_root

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.
    Source: modified from https://github.com/facebookresearch/fastMRI/blob/0c9fd544c1dc2e5de0b576ab0fb0349c1225924b/fastmri/data/transforms.py#L139
    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.
    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[1]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[1]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

class accelerateMRI(torch_data.Dataset):
    def __init__(self, path_file, center_fractions, accelerations, seed=42):
        super().__init__()
        self.path_file = path_file
        # load kspace volumes
        self.volume_kspace, self.et_root = get_data(self.path_file)
        # transform the data into appropriate format
        self.volume_kspace = transforms.to_tensor(self.volume_kspace)
        # define mask function
        self._mask_fnc = RandomMaskFunc(center_fractions=[center_fractions],
                                        accelerations=[accelerations],
                                        seed=seed)
        # Apply the mask to the k-space 
        self.mask_kspace, self.mask, _ = transforms.apply_mask(self.volume_kspace, self._mask_fnc)

        # extract target image width, height from ismrmrd header
        enc = ["encoding", "encodedSpace", "matrixSize"]
        self.crop_size = (
            int(et_query(self.et_root, enc + ["x"])),
            int(et_query(self.et_root, enc + ["y"])),
        )
        print('file', self.path_file)
           
    # convert k-space into image space
    def transform_slice(self, img_slice, type=None):
        
        # return accelerated image (from mask_kspace)
        if type == 'accelerate':
            # Apply Inverse Fourier Transform to get the complex image
            slice_image = fastmri.ifft2c(self.mask_kspace)
            # Compute absolute value to get a real image
            slice_image_abs = fastmri.complex_abs(slice_image)
            # Combine coils into the full image with root-sum-of-squares transforms
            sampled_image_rss = fastmri.rss(slice_image_abs, dim=0)
            # crop input image
            print('image shape before cropping: ', sampled_image_rss.shape)
            image_crop = center_crop(sampled_image_rss, self.crop_size)
            print('cropped image shape', image_crop.shape)
            
        else:
            # Apply Inverse Fourier Transform to get the complex image
            slice_image = fastmri.ifft2c(self.volume_kspace)
            # Compute absolute value to get a real image
            slice_image_abs = fastmri.complex_abs(slice_image)
            # Combine coils into the full image with root-sum-of-squares transforms
            sampled_image_rss = fastmri.rss(slice_image_abs, dim=0)
            # crop input image
            print('image shape before cropping: ', sampled_image_rss.shape)
            image_crop = center_crop(sampled_image_rss, self.crop_size)
            print('cropped image shape', image_crop.shape)
                
        # Convert back to numpy array
        return np.abs(image_crop.numpy())
    
    def __getitem__(self, idx):
        # get image at idx
        img = self.volume_kspace[idx]
        # get accelerated image
        img_transformed = self.transform_slice(img, type='accelerate')
        # get original complex image at idx
        orig_img = self.transform_slice(img)
        
        return self.path_file, orig_img, img_transformed
    
def normalize_pix(img_arr):
    return (np.maximum(img_arr, 0)/img_arr.max()) * 255.0

# SCRIPT STARTS HERE
data_path = 'fastMRI/multicoil_train'
# test with 2 files only -- crashes with full dataset...
files = [data_path+'/'+'file_brain_AXFLAIR_200_6002442.h5', data_path+'/'+'file_brain_AXFLAIR_200_6002543.h5']

# load data
# kspace_volumes = [get_kspace(data_path+'/'+files) for files in os.listdir(data_path)] # ERROR: zsh: killed python.....
# kspace_volume = [get_kspace(f) for f in files]

accelerations = [4, 8]
center_fractions = [0.08, 0.04]
slice_idx = 10

# select a random idx from list of files
# file_idx = np.random.randint(0, len(os.listdir(data_path)))
file_idx = np.random.randint(0, len(files))

# plot different accelerations at slice
transformed_img = []
for center_frac in center_fractions:
    fix, axis = plt.subplots(1, len(accelerations)+1, figsize=(20, 9))
    
    # start i count from 1 -- index 0 for original image
    for i, accel in enumerate(accelerations, 1):
        # init class    
        accel_MRI = accelerateMRI(files[file_idx], center_frac, accel, seed=42)
        # return transformed complex image
        name, original_img, transf_img = accel_MRI.__getitem__(file_idx)
        transformed_img.append(transf_img)
        # get file name only
        fname = os.path.basename(name)
        
        # plot each acceleration at slice 0
        axis[i].imshow(normalize_pix(transf_img[slice_idx]), cmap='gray')
        axis[i].set_title(f'x{accel} acceleration, {center_frac} center fractions', fontsize=12)
    
    # plot original in first subplot position
    # get first slice
    arrimg = original_img[slice_idx]
    # normalize values
    img2d = normalize_pix(arrimg)
    axis[0].imshow(img2d, cmap='gray')
    axis[0].set_title(f'{fname} (original)', fontsize=12)
    
    plt.show()

