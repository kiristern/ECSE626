import os
import shutil
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print(os.getcwd())

pth = "fastMRI/multicoil_train/"

hf = [h5py.File(pth+'/'+i) for i in os.listdir(pth)]
# extract filenames from HDF5 files
filenames = [os.path.basename(f.filename) for f in hf]

# reserve 20% of data for test set
train_val_samples, test_samples = train_test_split(filenames, test_size=0.2, random_state=42)
# reserve 20% of data for val set
train_samples, val_samples = train_test_split(train_val_samples, test_size=0.2, random_state=42)

print(len(train_samples), len(val_samples), len(test_samples))

output = "fastMRI/accel4"

for sample in tqdm(train_samples):
    if not os.path.exists(output+'/train'):
        os.makedirs(output+'/train')
    shutil.copy(os.path.join(pth, sample), os.path.join(output+'/train', sample))

for sample in tqdm(val_samples):
    if not os.path.exists(output+'/val'):
        os.makedirs(output+'/val')
    shutil.copy(os.path.join(pth, sample), os.path.join(output+'/val', sample))

for sample in tqdm(test_samples):
    if not os.path.exists(output+'/test'):
        os.makedirs(output+'/test')
    shutil.copy(os.path.join(pth, sample), os.path.join(output+'/test', sample))