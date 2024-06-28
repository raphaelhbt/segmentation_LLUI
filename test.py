import torchio as tio
import random
import ants
import numpy as np
import torch


FLAIR_path = '/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/anat/sub-1_ses-0001_FLAIR.nii.gz'
ADC_path = '/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/dwi/sub-1_ses-0001_ADC.nii.gz'
DWI_path = '/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/dwi/sub-1_ses-0001_dwi.nii.gz'

path=[ADC_path, FLAIR_path, DWI_path]
torch.manual_seed(0)
# Load the images
image=[]
for i in range(len(path)):
    image.append(ants.image_read(path[i]).numpy())


def RandomMirror(list_of_images):
    # Initialize the transformation
    transform = tio.RandomFlip(axes=(0, 1, 2), p=0.5)
    
    # Apply the same transformation to all images
    transformed_images = []
    for img in list_of_images:
        img_tensor = torch.tensor(img).unsqueeze(0)  
        torch.manual_seed(0)  
        transformed_tensor = transform(img_tensor)  
        transformed_np = transformed_tensor.squeeze(0).numpy()  
        transformed_images.append(transformed_np)
        print(transformed_tensor.shape)
        print(transformed_np.shape)
        
    return transformed_images



# Save the images
transformed = RandomMirror(image)
for i in range(len(transformed)):
    # Convert TorchIO image to PyTorch tensor, then to NumPy array
    img_np = transformed[i]
    ants.image_write(ants.from_numpy(img_np), f'transformed_{i}.nii.gz')





 

