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

data=[ADC_path, DWI_path, FLAIR_path, mask_path]
data=[ants.image_read(data[i]).numpy() for i in range(len(data))]
new_data = []
for i in range(len(data)):
    new_data.append(tio.ScalarImage(tensor=torch.tensor(data[i]).unsqueeze(0)))

def RandomRotateScale(list_of_images):
    """
    Randomly rotate and scale the images with a probability of 0.16 for each transformation and 0.08 for the combined transformation.
    The angle is sampled from a uniform distribution between -30 and 30 degrees.
    The scale is sampled from a uniform distribution between 0.7 and 1.4.

    Args:
    list_of_images: list of torchio images

    Returns:
    transformed_img: list of transformed torchio images
    """
    prob_both = 0.08
    prob_scale = 0.16
    prob_rotate = 0.16

    do_both = random.random() < prob_both
    do_scale = random.random() < prob_scale
    do_rotate = random.random() < prob_rotate

    both = tio.RandomAffine(
            scales=(0.7, 1.4),
            degrees=(-30, 30, -30, 30, -30, 30),
            isotropic=True,
            default_pad_value=0
        )
    scale = tio.RandomAffine(
            scales=(0.7, 1.4),
            isotropic=True,
            default_pad_value=0
        )
    rotate = tio.RandomAffine(
            degrees=(-30, 30, -30, 30, -30, 30),
            isotropic=True,
            default_pad_value=0
        )
    if do_both:
        for img in list_of_images:
            img = both(img)
    elif do_scale:
        for img in list_of_images:
            img = scale(img)
    elif do_rotate:
        for img in list_of_images:
            img = rotate(img)
    return list_of_images

def RandomGaussianNoise(list_of_images):
    """
    Randomly add Gaussian noise to the images with a probability of 0.15. 
    The standard deviation is sampled from a uniform distribution between 0 and 0.1.

    Args:
    list_of_images: list of torchio images

    Returns:
    transformed: list of transformed torchio images
    """
    prob = 0.15
    do_noise = random.random() < prob

    transform = tio.RandomNoise(
                mean=0, 
                std=(0, 0.1), 
            )

    if do_noise:
        for i in range(len(list_of_images)-1):  # Don't apply noise to the mask
            # Create a mask where the image is not equal to 0
            mask = list_of_images[i].tensor.squeeze(0) != 0
            
            # Apply the transformation
            transformed_image = transform(list_of_images[i])
            
            # Extract tensor from the transformed image
            transformed_tensor = transformed_image.tensor
            
            # Apply the mask: keep original values where the mask is False (image == 0)
            masked_tensor = transformed_tensor * mask.float().unsqueeze(0)
            
            # Create a new ScalarImage with the masked tensor
            transformed_image = tio.ScalarImage(tensor=masked_tensor)
            
            list_of_images[i] = transformed_image
    return list_of_images

def RandomGaussianBlur(list_of_images):
    """
    Add random Gaussian blur to the images with a probability of 0.2. If this augmentation
    is triggered in a sample, blurring is applied with a probability of 0.5 for each of the
    associated modalities. The kernel width is sampled from a uniform distribution between 0.5 and 1.5.

    Args:
    list_of_images: list of torchio images

    Returns:
    transformed_sample: list of transformed torchio images
    """
    sample_prob=0.2
    modality_prob=0.5
    
    do_blur = random.random() < sample_prob
    do_modality = random.random() < modality_prob

    kernel_width = random.uniform(0.5, 1.5)
    transform = tio.RandomBlur(
                    std=(kernel_width, kernel_width),
                )
    if do_blur:
        if do_modality:
            for i in range(len(list_of_images) - 1): # Don't apply blur to the mask
                list_of_images[i] = transform(list_of_images[i])
    return list_of_images

def RandomBrightness(list_of_images):
    """
    Randomly adjust the brightness of the images with a probability of 0.15.
    The factor is sampled from a uniform distribution between 0.7 and 1.3.

    Args:
            list_of_images: list of torchio images
    
    Returns:
    list_of_images: list of torchio images
    """
    prob = 0.15
    factor = random.uniform(0.7, 1.3)
    if random.random() < prob:
        for i in range(len(list_of_images) - 1): # Don't apply brightness to the mask
            img = list_of_images[i].numpy().squeeze(0)
            list_of_images[i] = (img * factor) 
            list_of_images[i] = tio.ScalarImage(tensor=torch.tensor(list_of_images[i]).unsqueeze(0))
    return list_of_images

def RandomContrast(list_of_images):
    """
    Randomly adjust the contrast of the images with a probability of 0.15.
    The factor is sampled from a uniform distribution between 0.7 and 1.3.
    Following multiplication, the values are clipped to their original value range.

    Args:
    list_of_images: list of torchio images

    Returns:
    list_of_images: list of torchio images
    """
    prob = 0.15
    factor = random.uniform(0.7, 1.3)

    if random.random() < prob:
        for i in range(len(list_of_images) - 1): # Don't apply contrast to the mask
            img = list_of_images[i].numpy().squeeze(0)
            list_of_images[i] = (img * factor)
            list_of_images[i] = np.clip(list_of_images[i], img.min(), img.max())
            list_of_images[i] = tio.ScalarImage(tensor=torch.tensor(list_of_images[i]).unsqueeze(0))
    return list_of_images
 
def RandomLowResolution(list_of_images):
    """
    Randomly downsample the images with a probability of 0.25. If this augmentation is triggered in a sample,
    downsampling is applied with a probability of 0.5 for each of the associated modalities.
    The downsampling factor is sampled from a uniform distribution between 1 and 2. The interpolation method is nearest neighbor for downsampling.

    Args:
    list_of_images: list of torchio images

    Returns:
    transformed_sample: list of transformed torchio images
    """
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
    """
    Randomly adjust the gamma of the images with a probability of 0.15. If this augmentation is triggered in a sample,
    gamma adjustment is applied with a probability of 0.15 for each of the associated modalities.
    The gamma factor is sampled from a uniform distribution between 0.7 and 1.5.
    The patch intensities are first normalized to [0,1] before applying the gamma transformation. With a probability of 0.15,
    the gamma transformation is applied to the inverted intensities of the image before being applied again.

    Args:
    list_of_images: list of torchio images

    Returns:
    list_of_images: list of torchio images
    """
    prob= 0.15
    prob_prior_transform = 0.15

    do_it = random.random() < prob
    do_it_prior_transform = random.random() < prob_prior_transform

    if do_it:
        transform = tio.RandomGamma(log_gamma=(0.7, 1.5))
        for i in range(len(list_of_images) - 1): # Don't apply gamma to the mask
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
    """
    Randomly mirror the images with a probability of 0.15. If this augmentation is triggered in a sample,
    mirroring is applied with a probability of 0.5 for each of the associated modalities.

    Args:
    list_of_images: list of torchio images

    Returns:
    list_of_images: list of torchio images
    """
    # Initialize the transformation
    transform = tio.RandomFlip(axes=(0, 1, 2), p= 0.5) 
    
    # Apply the same transformation to all images
    transformed_images = []
    for i in range(len(list_of_images)):
        torch.manual_seed(0)  
        transformed_tensor = transform(list_of_images[i])  
        transformed_images.append(transformed_tensor)
        
    return transformed_images
 
    """
    Randomly mirror the images with a probability of 0.15. If this augmentation is triggered in a sample,
    mirroring is applied with a probability of 0.5 for each of the associated modalities.

    Args:
    list_of_images: list of torchio images

    Returns:
    list_of_images: list of torchio images
    """
    # Initialize the transformation
    transform = tio.RandomFlip(axes=(0, 1, 2), p= 0.5) 
    
    # Apply the same transformation to all images
    transformed_images = []
    for i in range(len(list_of_images)):
        torch.manual_seed(0)  
        transformed_tensor = transform(list_of_images[i])  
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

transformed_imgs = transform(new_data)
for i in range(len(transformed_imgs)):
    transformed_imgs[i] = transformed_imgs[i].numpy().squeeze(0)


for i in range(len(transformed_imgs)):
    ants.image_write(ants.from_numpy(transformed_imgs[i]), f'transformed_img_{i}.nii.gz')

