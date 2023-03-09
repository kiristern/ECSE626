#%%
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
from fastmri.data import subsample
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms, mri_data
from fastmri.data.mri_data import et_query

        
data_dir = '../fastMRI/tiny'
# if sudden issue with running script, try rm .DS_Store file -- it messes up import!

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
    return masked_kspace, kspace

# apply transform to all slices for each image file in the dataset folder
dataset = mri_data.SliceDataset(
    root=Path(data_dir),
    transform=data_transform,
    challenge='multicoil'
)

print(dir(dataset))
print(dataset.__dict__)


# for i, (masked_kspace, kspace) in enumerate(dataset):
#     print('masked_kspace[0]: ', masked_kspace[0]) # masked_kspace
#     print()
#     print('masked_kspace[1]: ', masked_kspace[1]) # mask
#     print()
#     print('kspace original', kspace)

# plots the sampled kspace mask
# for i, (j, k) in enumerate(dataset):
#     print('index: ', i, 'j[0].shape: ', j[0].shape)
#     complx = fastmri.complex_abs(j[0][15])
#     print('complx shape at slice 16: ', complx.shape) # torch.Size([640, 320])
#     plt.imshow(np.log(np.abs(complx) + 1e-9), cmap='gray')

#%%
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

# # plot reconstructed image for each slice
# for i, (j, k) in enumerate(dataset):
#     print('index: ', i, 'j[0].shape: ', j[0].shape)
#     rss = do_reconstruction(j[0])
#     # crop image
#     cropped = center_crop(rss, (320, 320))
#     print('cropped rss_img shape: ', cropped.shape)
#     plt.imshow(cropped, cmap='gray')
#     plt.show()
    

# for (masked_kspace, kspace) in dataset:
#     plt.imshow(do_reconstruction(masked_kspace[0]), cmap='gray')
    
#%%
# SliceDataset exploration
print('total number of slices in dataset: ', len(dataset)) # 48 --> 3 files in tiny folder; each with 16 channels
print('datatype: ', type(dataset[0])) # tuple: ()
temp = dataset[0][0][0] # first idx of tuple from first slice from dataset 
print('shape of slice 0: ', temp.shape) # torch.Size([16, 640, 320, 2]) --> (slices, height, width, coils)
slice10 = temp[10] # take slice10
print('slice10 from slice1 shape: ', slice10.shape) # torch.Size([640, 320, 2])
# plot abs val of k-space for each coil
plt.subplot(131), plt.imshow(np.log(np.abs(slice10) + 1e-9)[:,:,0], cmap='gray')
plt.subplot(132), plt.imshow(np.log(np.abs(slice10) + 1e-9)[:,:,1], cmap='gray')
plt.show()

# # apply inverse fourier transform to get complex image
# img_FT = fastmri.ifft2c(slice10)
# print('complex image shape: ', img_FT.shape) # torch.Size([640, 320, 2])
# # compute absolute val to get a real image
# img_abs = fastmri.complex_abs(img_FT)
# print('abs val img shape: ', img_abs.shape) # torch.Size([640, 320])
# # plot
# plt.imshow(img_abs, cmap='gray')
# plt.show()


# # plot all first 16 slices from dataset slice 0
# img_ft = []
# for i, slices in enumerate(temp):
#     img_ft.append(fastmri.ifft2c(slices))
# print('img_ft len: ', len(img_ft))
# abs_img = []
# for i, slices in enumerate(img_ft):
#     abs_img.append(fastmri.complex_abs(slices))

# plt.figure(figsize=(10,10))
# for i, slices in enumerate(abs_img):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(abs_img[i], cmap='gray')
# plt.show()




# # get filename for each slice in dataset
# fname = [dataset.__dict__["examples"][idx][0] for idx in range(len(dataset))]
# fname

# # extract file basename
# for i, j in enumerate(dataset):
#     print(os.path.basename(dataset.__dict__["examples"][i][0]))

#%%
# create a dictionary for data
# data_dict = {}

# # get the names of filenames from data
# fnames = [os.path.basename(dataset.__dict__["examples"][i][0]) for i in range(len(dataset))]
# # convert fnames list to a set, to get unique filename instances only
# fname = set(fnames)

# for filename in fname:
#     # add unique filenames to dictionary
#     data_dict[filename] = {}
#     # add subdicts for each file
#     data_dict[filename] = {'acceleration': {}, 'original_rss': {}}
      
#     count = 0 # init count to zero for the current filename 
#     for i, (masked_kspace, kspace) in enumerate(dataset):
#         # add masked_kspace array to each slice if filename is the same
#         if os.path.basename(dataset.__dict__["examples"][i][0]) == filename:
#             count += 1 # increment count for the current filename
#             # add reconstructed data slices for each file
#             data_dict[filename]['acceleration'].setdefault(f'slice{count}', do_reconstruction(masked_kspace[0]))
#             # add original rss data for each slice in the file
#             data_dict[filename]['original_rss'].setdefault(f'slice{count}', do_reconstruction(kspace))
#%%
# view keys in dictionary
# print('filenames: ', data_dict.keys())

# print keys for each sub-dictionary
# for key, sub_dict in data_dict.items():
#     print(f"Dataset type for filename '{key}':")
#     for sub_key, subsub_dict in sub_dict.items():
#         print(f"Slices for dataset '{sub_key}':")
#         for subsub_key in subsub_dict.keys():
#             print(subsub_key)
#         print() # add an empty line between sub-dicts


###############################
#%%
class doReconstruction():    
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.mask_function = subsample.RandomMaskFunc(
                                                    center_fractions = [0.08],
                                                    accelerations = [4],
                                                    seed=42
                                                )
        self.apply_transform = self.data_transform
        self.dataset = mri_data.SliceDataset(
            root=Path(self.data_dir),
            transform=self.apply_transform,
            challenge='multicoil',
            )
        self.crop_size = (320, 320)

    # create a k-space mask for transforming input data using RandomMaskFunction
    def data_transform(self, kspace, mask, target, data_attributes, filename, slice_num):
        # Transform the data into appropriate format
        # Here we simply mask the k-space and return the result
        kspace = transforms.to_tensor(kspace)
        masked_kspace = transforms.apply_mask(kspace, self.mask_function)
        return masked_kspace, kspace
    
    def center_crop(self, data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
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

        return data[w_from:w_to, h_from:h_to]
    
    # function to reconstruct a complex image
    def reconstruct(self, kspace):
        # apply inverse fourier transform to get complex image
        ift = fastmri.ifft2c(kspace)
        # compute abs value to get a real image
        ift_abs = fastmri.complex_abs(ift)
        # combine coils into the full image with root-sum-of-squares transform
        rss = fastmri.rss(ift_abs, dim=0)
        # crop the image
        cropped = self.center_crop(rss, self.crop_size)
        
        return np.abs(cropped.numpy())
    
    def __getitem__(self, dataset):
        dataset = self.dataset
        
        # create a dictionary for data
        data_dict = {}
        
        # get the names of filenames from data
        fnames = [os.path.basename(dataset.__dict__["examples"][i][0]) for i in range(len(dataset))]
        # convert fnames list to a set, to get unique filename instances only
        fname = set(fnames)
        
        for filename in fname:
            # add unique filenames to dictionary
            data_dict[filename] = {}
            # add subdicts for each file
            data_dict[filename] = {'acceleration': {}, 'original_rss': {}}
            
            count = 0 # init count to zero for the current filename 
            for i, (masked_kspace, kspace) in enumerate(dataset):
                # add masked_kspace array to each slice if filename is the same
                if os.path.basename(dataset.__dict__["examples"][i][0]) == filename:
                    count += 1 # increment count for the current filename
                    # add reconstructed data slices for each file
                    data_dict[filename]['acceleration'].setdefault(f'slice{count}', self.reconstruct(masked_kspace[0]))
                    # add original rss data for each slice in the file
                    data_dict[filename]['original_rss'].setdefault(f'slice{count}', self.reconstruct(kspace))
            
        return data_dict
 

#%%
data_dir = '../fastMRI/tiny'

# get dataset dictionary
data = doReconstruction(data_dir)['data_dir']

#%%
# plot both accelerated and original slices
def plot_slices(data_dict, filename, slice_num):
    # Get the acceleration and original sub-dictionaries for the specified filename
    accel_dict = data_dict[filename]['acceleration']
    orig_dict = data_dict[filename]['original_rss']

    # Get the acceleration and original arrays for the specified slice number
    accel_slice = accel_dict.get(f'slice{slice_num}')
    orig_slice = orig_dict.get(f'slice{slice_num}')

    # Plot the acceleration and original arrays side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Slice {slice_num} from {filename}")
    ax1.imshow(accel_slice, cmap='gray')
    ax1.set_title("Acceleration")
    ax2.imshow(orig_slice, cmap='gray')
    ax2.set_title("Original")
    plt.show()
    
for files in data:
    plot_slices(data, files, slice_num=9)
    

#%%
# stack the slices to make a 3D volume
def make3Dvolume(data):
    for f, file_data in data.items():
        for data_type, slices in file_data.items():
            stacked_slices = []
            # create volume by stacking array slices into 1 object
            for s, arr in slices.items():
                stacked_slices.append(arr)
            # update dictionary to remove the slices sub-keys    
            file_data[data_type] = np.array(stacked_slices)
    return data

vol_dict = make3Dvolume(data)
#%%
plt.subplot(121), plt.imshow(vol_dict['file_brain_AXFLAIR_200_6002442.h5']['acceleration'][0], cmap='gray')
plt.subplot(122), plt.imshow(vol_dict['file_brain_AXFLAIR_200_6002442.h5']['original_rss'][0], cmap='gray')
plt.show()

#%% 
# print keys for each sub-dictionary
for key, sub_dict in vol_dict.items():
    print(f"Dataset type for filename '{key}':")
    for sub_key, subsub_dict in sub_dict.items():
        print(f"Slices for dataset '{sub_key}':")
        print(subsub_dict.shape)
    print() # add an empty line between sub-dicts
# %%
from torch.utils.data import Dataset, DataLoader

class fastDataset(Dataset):
    def __init__(self, data_dict: Dict, is3D: bool = False):
        self.data_dict = data_dict
        self.is3D = is3D 
        # return data_dict as 3D volume or not
        if self.is3D == True:
            self.data_dict = self.make3Dvolume(self.data_dict)
    
    def __len__(self) -> int:
        """
        Return length of the dataset
        """
        if self.is3D == True:
            return len(self.data_dict.keys()) # get the number of files in dictionary
        else:
            total_slices = 0
            for file_data in self.data_dict.values():
                # from acceleration array (doesn't matter since original_rss is the same len)
                slices = file_data.get('acceleration', {})
                # add total number of slices from acceleration data to total slice count
                total_slices += len(slices)
            return total_slices
            
    # stack the slices to make a 3D volume
    def make3Dvolume(self, data_dict):
        for file, file_data in self.data_dict.items():
            for data_type, slices in file_data.items():
                stacked_slices = []
                # create volume by stacking array slices into 1 object
                for slice_data in slices:
                    stacked_slices.append(slice_data)
                # update dictionary to remove the slices sub-keys    
                file_data[data_type] = np.array(stacked_slices)
        return data_dict
    
    
# %%
data2d = fastDataset(data_dict)
print('len data2d', len(data2d))
data3d = fastDataset(data_dict, is3D=True)
print('len data3d', len(data3d))

print(data3d.__dict__)
print(data3d['file_brain_AXFLAIR_200_6002442.h5']['acceleration'])
# %%