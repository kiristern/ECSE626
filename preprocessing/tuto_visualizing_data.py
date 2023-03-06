# apply acceralation on data following: https://github.com/facebookresearch/fastMRI/blob/main/fastMRI_tutorial.ipynb
import h5py
import numpy as np
from matplotlib import pyplot as plt
import os
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
from PIL import ImageDraw, Image, ImageEnhance
import xml.etree.ElementTree as etree
from fastmri.data.mri_data import et_query


pth = 'fastMRI/multicoil_train'
print('number of files: ', len(os.listdir(pth)))

hf = [h5py.File(pth+'/'+i) for i in os.listdir(pth)]
print('Keys: ', list([hf[i].keys() for i in range(len(hf))]))
print('Attrs: ', [dict(hf[i].attrs) for i in range(len(hf))])

# Multi-coil MRIs k-space has the following shape: (number of slices, number of coils, height, width)
# Single-coil MRIs, k-space has the following shape: (number of slices, height, width)
# MRIs are acquired as 3D volumes, the first dimension is the number of 2D slices
img = hf[0]
volume_kspace = img['kspace'][()]
print(volume_kspace.dtype)
print(volume_kspace.shape)

et_root = etree.fromstring(img["ismrmrd_header"][()])
# extract target image width, height from ismrmrd header
enc = ["encoding", "encodedSpace", "matrixSize"]
crop_size = (
    int(et_query(et_root, enc + ["x"])),
    int(et_query(et_root, enc + ["y"])),
)
print('image (height, width):', crop_size)

# Choosing the 10-th slice of volume[0]
slice_kspace = volume_kspace[10]
print(slice_kspace.shape)

# view the absolute value of k-space
def show_coils(data, slice_nums, cmap=None):
    plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap='gray')
    plt.show()

# This shows coils 0, 3 and 5
show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 3, 5])

# The fastMRI repo contains some utlity functions to convert k-space into image space. 
# These functions work on PyTorch Tensors. 
# The to_tensor function can convert Numpy arrays to PyTorch Tensors.
import fastmri
from fastmri.data import transforms as T

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

slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image
print('before crop image shape: ', slice_image_abs.shape)
crop_img = center_crop(slice_image_abs, crop_size) # crop input image
print('cropped img size: ', crop_img.shape)

show_coils(crop_img, [0, 3, 5], cmap='gray')

# As we can see, each coil in a multi-coil MRI scan focusses on a different region of the image. 
# These coils can be combined into the full image using the Root-Sum-of-Squares (RSS) transform.
slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')
plt.show()

# So far, we have been looking at fully-sampled data. 
# We can simulate under-sampled data by creating a mask and applying it to k-space.

from fastmri.data.subsample import RandomMaskFunc
mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])  # Create the mask function object

masked_kspace, mask, _ = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space

sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)
cropped = center_crop(sampled_image_rss, crop_size)

plt.imshow(np.abs(cropped.numpy()), cmap='gray')
plt.show()

# also see: https://github.com/facebookresearch/fastMRI/blob/main/fastmri_examples/annotation/fastmri_plus_viz.ipynb
img_data = img["reconstruction_rss"][:]
print("image shape", img_data.shape)

# display an imageslice
arrimg = np.squeeze(img_data[0,:,:])
img2d_scaled = (np.maximum(arrimg,0)/arrimg.max()) * 255.0
img2d_scaled = Image.fromarray(np.uint8(img2d_scaled))
plt.figure(figsize=(10,10))
plt.imshow(img2d_scaled, cmap='gray')
plt.show()

# click link to view more