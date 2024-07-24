import ants
import torch
import torchio as tio
import random
import numpy as np
import torchio.transforms as transforms

import warnings  # Import the warnings module

# Suppress the specific RuntimeWarning from torchio
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Output shape.*!= target shape.*Fixing with CropOrPad")


ADC_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/dwi/sub-1_ses-0001_ADC.nii.gz'
DWI_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/dwi/sub-1_ses-0001_dwi.nii.gz'
FLAIR_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/anat/sub-1_ses-0001_FLAIR.nii.gz'
mask_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/derivatives/sub-1/ses-0001/sub-1_ses-0001_msk.nii.gz'

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)  # Set seed for random module

data=[ADC_path, DWI_path, FLAIR_path, mask_path]
data=[ants.image_read(data[i]).numpy() for i in range(len(data))]
new_data = []
for i in range(len(data)-1):
    new_data.append(tio.ScalarImage(tensor=torch.tensor(data[i]).unsqueeze(0)))
new_data.append(tio.LabelMap(tensor=torch.tensor(data[-1]).unsqueeze(0)))

# Assuming new_data contains 3 intensity images and 1 label map, adjust the keys as necessary
subject_dict = {
    'ADC': new_data[0],
    'DWI': new_data[1],
    'FLAIR': new_data[2],
    'mask': new_data[3],  # Assuming the last one is the label map
}

# Create a torchio.Subject
subject = tio.Subject(subject_dict)

#nnU-Net data augmentation
num1 = random.uniform(0.5, 1.5)
num2 = random.uniform(0.5, 1.5)
std_values = sorted([num1, num2])

def RandomBlur(x):
    prob_modality = 0.5
    do_blur = random.random() < prob_modality
    if do_blur:
        std = random.uniform(std_values[0], std_values[1])
        x = tio.RandomBlur(std=std)(x)
    return x

def RandomBrightness(x):
    factor = random.uniform(0.7, 1.3)
    x = x * factor
    return x

def RandomContrast(x):
    factor = random.uniform(0.65, 1.5)
    original_min = x.data.min().item()  # Get the original minimum intensity
    original_max = x.data.max().item()  # Get the original maximum intensity
    x.data *= factor
    x.data = x.data.clip(original_min, original_max)  # Clip the voxel intensities to the original range
    return x

def RandomLowResolution(x):
    modality_prob = 0.5
    do_low_res = random.random() < modality_prob
    if do_low_res:
        anisotropy_transform = tio.RandomAnisotropy(
            axes=(0, 1, 2),
            downsampling=(1, 2),
        )
        x = anisotropy_transform(x)
    return x

def RandomGamma(x):
    prob_prior_transform = 0.15
    do_prior_transform = random.random() < prob_prior_transform
    transform= tio.RandomGamma(log_gamma=(0.7, 1.5))

    mask = x.data != 0

    #Normalise image to [0, 1]
    x.data = (x.data - x.data.min()) / (x.data.max() - x.data.min() + 1e-6)

    if do_prior_transform:
        x = 1 - transform(1 - x)
    
    x = transform(x)

    #Scale back to original range
    x.data = x.data * (x.data.max() - x.data.min()) + x.data.min()
    x.data = x.data * mask
    return x

transforms_nnUNet = tio.Compose([
        tio.RandomAffine(
            scales=(0.7, 1.4),  # only scaling from U(0.7,1.4)
            isotropic=True,
            default_pad_value=0,
            p=0.16
        ),
        tio.RandomAffine(
            degrees=30,  # This will be interpreted as (-30, 30) for each axis
            isotropic=True,
            default_pad_value=0,
            p=0.16
        ),
        tio.RandomNoise(     # Add Gaussian noise with random parameters
            mean=0, 
            std=(0, 0.1),
            exclude=['label'], 
            p=0.15
        ),
        tio.Lambda(RandomBlur,
                    p=0.2,
                    ),
        tio.Lambda(RandomBrightness, 
                   types_to_apply=[tio.INTENSITY],
                   p=0.15
        ),
        tio.Lambda(RandomContrast, 
                   types_to_apply=[tio.INTENSITY],
                   p=0.15
        ),
        tio.Lambda(RandomLowResolution, 
                   p=0.25
        ),
        tio.Lambda(RandomGamma,
                    types_to_apply=[tio.INTENSITY],
                    p=0.15
        ),
        tio.RandomFlip(
                    axes=(0, 1, 2), 
                    p=0.5
        ),
])

transformed = transforms_nnUNet(subject)

# Save the transformed images
for modality in transformed:
    image_tensor = transformed[modality].data
    ants_image = ants.from_numpy(image_tensor.numpy().squeeze())
    ants.image_write(ants_image, f'transformed_{modality}.nii.gz')