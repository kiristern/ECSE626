#%%
import os
import pathlib 
from typing import Dict
import h5py
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as etree
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.utils.data as torch_data
import fastmri
from fastmri.data import subsample
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms, mri_data
from fastmri.data.mri_data import et_query

        
data_dir = '../fastMRI/tiny'

#%%  
# create a k-space mask for transforming input data using RandomMaskFunction
# mask_func = subsample.RandomMaskFunc(
#     center_fractions = [0.08, 0.04],
#     accelerations = [4, 8],
#     seed=42
# )
mask_func = subsample.RandomMaskFunc(
    center_fractions = [0.08],
    accelerations = [4],
    seed=42
)

def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the data into appropriate format
    # Here we simply mask the k-space and return the result
    kspace = transforms.to_tensor(kspace)
    # returns: tuple((masked_kspace, mask), num_low_frequencies)
    masked_kspace = transforms.apply_mask(kspace, mask_func)
    # masked_kspace = transforms.apply_mask(kspace, mask_func)
    return masked_kspace

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

def do_reconstruction(masked_kspace, crop_size=(320, 320)):
    # apply inverse fourier transform to get complex image
    ift = fastmri.ifft2c(masked_kspace)
    # compute abs value to get a real image
    ift_abs = fastmri.complex_abs(ift)
    # combine coils into the full image with root-sum-of-squares transform
    rss = fastmri.rss(ift_abs, dim=0)
    
    return np.abs(rss.numpy())

# apply transform to all slices for each image file in the dataset folder
dataset = mri_data.SliceDataset(
    root=pathlib.Path(data_dir),
    transform=data_transform,
    challenge='multicoil'
)

print(dir(dataset))
print(dataset.__dict__)

#%%
for masked_kspace in dataset:
    print('masked_kspace[0]: ', masked_kspace[0]) # masked_kspace
    print('masked_kspace[1]: ', masked_kspace[1]) # mask
#%%
# plots the sampled kspace mask
for i, j in enumerate(dataset):
    print('index: ', i, 'j[0].shape: ', j[0].shape)
    complx = fastmri.complex_abs(j[0][15])
    print('complx shape at slice 16: ', complx.shape) # torch.Size([640, 320])
    plt.imshow(np.log(np.abs(complx) + 1e-9), cmap='gray')
            
#%%
# plot reconstructed image for each slice
for i, j in enumerate(dataset):
    print('index: ', i, 'j[0].shape: ', j[0].shape)
    rss = do_reconstruction(j[0])
    # crop image
    cropped = center_crop(rss, (320, 320))
    print('cropped rss_img shape: ', cropped.shape)
    plt.imshow(cropped, cmap='gray')
    plt.show()
    
# %%
for masked_kspace in dataset:
    plt.imshow(do_reconstruction(masked_kspace[0]), cmap='gray')
    
    
#%%
# SliceDataset exploration
print('total number of slices in dataset: ', len(dataset)) # 80 --> 5 files in tiny folder; each with 16 channels
print('datatype: ', type(dataset[0])) # tuple: ()
temp = dataset[0][0] # first idx of tuple from first slice from dataset 
print('shape of slice 0: ', temp.shape) # torch.Size([16, 640, 320, 2]) --> (slices, height, width, coils)
slice10 = temp[10] # take slice10
print('slice10 from slice1 shape: ', slice10.shape) # torch.Size([640, 320, 2])
# plot abs val of k-space for each coil
plt.subplot(131), plt.imshow(np.log(np.abs(slice10) + 1e-9)[:,:,0], cmap='gray')
plt.subplot(132), plt.imshow(np.log(np.abs(slice10) + 1e-9)[:,:,1], cmap='gray')
plt.show()

# apply inverse fourier transform to get complex image
img_FT = fastmri.ifft2c(slice10)
print('complex image shape: ', img_FT.shape) # torch.Size([640, 320, 2])
# compute absolute val to get a real image
img_abs = fastmri.complex_abs(img_FT)
print('abs val img shape: ', img_abs.shape) # torch.Size([640, 320])
# plot
plt.imshow(img_abs, cmap='gray')
plt.show()

#%%
# plot all first 16 slices from dataset slice 0
img_ft = []
for i, slices in enumerate(temp):
    img_ft.append(fastmri.ifft2c(slices))
print('img_ft len: ', len(img_ft))
abs_img = []
for i, slices in enumerate(img_ft):
    abs_img.append(fastmri.complex_abs(slices))

plt.figure(figsize=(10,10))
for i, slices in enumerate(abs_img):
    plt.subplot(4, 4, i+1)
    plt.imshow(abs_img[i], cmap='gray')
plt.show()

# %%
