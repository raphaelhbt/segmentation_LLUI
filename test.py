# Trash file to test stuff, not the final code or anything

import ants
import torch
import torchio as tio
import random
import numpy as np
import torchio.transforms as transforms
ADC_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/dwi/sub-1_ses-0001_ADC.nii.gz'
DWI_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/dwi/sub-1_ses-0001_dwi.nii.gz'
FLAIR_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/anat/sub-1_ses-0001_FLAIR.nii.gz'
mask_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/derivatives/sub-1/ses-0001/sub-1_ses-0001_msk.nii.gz'

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)  # Set seed for random module

paths=[ADC_path, DWI_path, FLAIR_path, mask_path]
imgs=[]
for i in paths:
    img = ants.image_read(i).numpy()
    imgs.append(img)

def RandomRotateScale(list_of_images):
    transform = tio.OneOf({
        tio.RandomAffine(
            scales=(0.7, 1.4),
            degrees=(-30, 30, -30, 30, -30, 30),
            isotropic=True,
            default_pad_value=0,
            p=0.08
        ),
        tio.RandomAffine(
            scales=(0.7, 1.4),
            isotropic=True,
            default_pad_value=0,
            p=0.16
        ),
        tio.RandomAffine(
            degrees=(-30, 30, -30, 30, -30, 30),
            isotropic=True,
            default_pad_value=0,
            p=0.16
        ),
    }, p=1.0)

    transformed = []
    for img in list_of_images:
        transformed_img = transform(img)
    return transformed

def RandomGaussianNoise(list_of_images):
    transform = tio.RandomNoise(
                mean=0, 
                std=(0, 0.1), 
                p=0.15
            )
    
    transformed = []
    for img in list_of_images:
        transformed.append(transform(img))
    return transformed

def RandomGaussianBlur(list_of_images):
    sample_prob=0.15
    do_blur = random.random() < sample_prob

    if do_blur:
        transformed_sample = []
        for img in list_of_images:
            kernel_width = random.uniform(0.5, 1.5)
            transform = tio.RandomBlur(
                std=(kernel_width, kernel_width),
                p=0.5
            )
            transformed_sample.append(transform(img))
        return transformed_sample
    else:
        return list_of_images

def RandomBrightness(list_of_images):
    prob = 0.15
    factor = random.uniform(0.7, 1.3)
    transformed_sample = []
    if random.random() < prob:
        for i in range(len(list_of_images)):
            img = list_of_images[i].numpy().squeeze(0)
            transformed_sample.append((img * factor)) 
            transformed_sample[i] = tio.ScalarImage(tensor=torch.tensor(transformed_sample[i]).unsqueeze(0))    
    return list_of_images

def RandomContrast(list_of_images):
    prob = 0.15
    factor = random.uniform(0.7, 1.3)
    transformed_sample = []
    if random.random() < prob:
        for i in range(len(list_of_images)):
            img = list_of_images[i].numpy().squeeze(0)
            transformed_sample.append((img * factor)) 
            transformed_sample[i] = np.clip(transformed_sample[i], img.min(), img.max())
            transformed_sample[i] = tio.ScalarImage(tensor=torch.tensor(transformed_sample[i]).unsqueeze(0)) 
    return list_of_images
 
def RandomLowResolution(list_of_images):
    sample_prob = 0.25
    modality_prob = 0.5
    do_low_res = random.random() < sample_prob

    if do_low_res:
        transformed_sample = []
        factor = random.uniform(1, 2)
        for i in range(len(list_of_images)):
            do_modality = random.random() < modality_prob
            if do_modality:
                original_image = list_of_images[i]
                original_shape = original_image.shape

                # Downsample
                resample_transform = tio.Resample(
                    target=(factor, factor, factor),
                    image_interpolation='nearest',  # nearest neighbor for downsampling
                )
                downsampled = resample_transform(original_image)

                # Ensure the final shape matches the original using CropOrPad
                final_image = tio.CropOrPad(target_shape=original_shape[1:])(downsampled)
                transformed_sample.append(final_image)
            else:
                transformed_sample.append(list_of_images[i])
        return transformed_sample
    else:
        return list_of_images

def RandomGamma(list_of_images):
    prob= 0.15
    prob_prior_transform = 0.15

    do_it = random.random() < prob
    do_it_prior_transform = random.random() < prob_prior_transform

    if do_it:
        transform = tio.RandomGamma(log_gamma=(0.7, 1.5))
        for i in range(len(list_of_images)):
            mask = list_of_images[i].numpy().squeeze(0) != 0

            # Normalize image to [0,1]
            img_min, img_max = list_of_images[i].numpy().min(), list_of_images[i].numpy().max()
            normalized_img = (list_of_images[i] - img_min) / (img_max - img_min + 1e-6)

            if do_it_prior_transform:
                normalized_img = 1 - transform(1 - normalized_img)
            
            normalized_img = transform(normalized_img)

            # Scale back to original value range 
            scaled_img = normalized_img * (img_max - img_min) + img_min
            clipped_img = torch.clamp(torch.tensor(scaled_img), img_min, img_max)
            list_of_images[i] = clipped_img * torch.tensor(mask)
    return list_of_images

def RandomMirror(list_of_images):
    # Initialize the transformation
    transform = tio.RandomFlip(axes=(0, 1, 2), p= 0.5) 
    
    # Apply the same transformation to all images
    transformed_images = []
    for img in list_of_images:
        torch.manual_seed(0)  
        transformed_tensor = transform(img)  
        transformed_images.append(transformed_tensor)
        
    return transformed_images
    
transform = transforms.Compose([
    lambda data: RandomRotateScale(data),
    lambda data: RandomGaussianNoise(data),
    lambda data: RandomGaussianBlur(data),
    lambda data: RandomBrightness(data),
    lambda data: RandomContrast(data),
    lambda data: RandomLowResolution(data),
    lambda data: RandomGamma(data),
    lambda data: RandomMirror(data)
    ])


for i in range(len(imgs)):
    imgs[i] = tio.ScalarImage(tensor=torch.tensor(imgs[i]).unsqueeze(0))
transformed_imgs = transform(imgs)
for i in range(len(transformed_imgs)):
    transformed_imgs[i] = transformed_imgs[i].numpy().squeeze(0)












for i in range(len(transformed_imgs)):
    ants.image_write(ants.from_numpy(transformed_imgs[i]), f'transformed_img_{i}.nii.gz')


 

