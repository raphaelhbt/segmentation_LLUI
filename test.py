import ants
import torchio as tio
import random
import numpy as np
import torch

image_path = r'/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/anat/sub-1_ses-0001_FLAIR.nii.gz'

# Load the image with ants
image = ants.image_read(image_path)

# Convert the ants image to a numpy array
image_np = image.numpy()
image_tensor = torch.from_numpy(image_np).unsqueeze(0)
# Convert the torch tensor to a torchio ScalarImage
image_tio = tio.ScalarImage(tensor=image_tensor)

degrees = (-30, 30, -30, 30, -30, 30)
scales = (0.7, 1.4, 0.7, 1.4, 0.7, 1.4)

transform = tio.RandomAffine(
    scales=scales,
    degrees=degrees,  # Use a tuple representing the range for degrees
    default_pad_value=0,  # Use 0 to fill the background with zeros
)

# Apply the transformation to the torchio image
transformed = transform(image_tio)

# Convert the transformed torchio image back to an ants image for plotting
transformed_np = transformed.numpy().squeeze()
print(transformed_np.shape)
transformed_ants = ants.from_numpy(transformed_np)

ants.image_write(transformed_ants, 'transformed_image.nii.gz')
