import ants
from scipy.ndimage import zoom
import numpy as np

FLAIR=ants.image_read('/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/anat/sub-1_ses-0001_FLAIR.nii.gz')

FLAIR=FLAIR.numpy()

def scale(image):
    # Generate a random scale parameter
    scale = np.random.uniform(0.7, 1.4)

    # Scale the image
    scaled_image = zoom(image, scale)

    # Initialize an empty array with the original image size
    output_image = np.zeros_like(image)

    # If the scaled image is larger than the original image, crop it
    if scale > 1:
        start_x = (scaled_image.shape[0] - image.shape[0]) // 2
        start_y = (scaled_image.shape[1] - image.shape[1]) // 2
        start_z = (scaled_image.shape[2] - image.shape[2]) // 2

        output_image = scaled_image[
            start_x:start_x + image.shape[0],
            start_y:start_y + image.shape[1],
            start_z:start_z + image.shape[2]
        ]
    # If the scaled image is smaller than the original image, pad it
    else:
        pad_x = (image.shape[0] - scaled_image.shape[0]) // 2
        pad_y = (image.shape[1] - scaled_image.shape[1]) // 2
        pad_z = (image.shape[2] - scaled_image.shape[2]) // 2

        output_image[
            pad_x:pad_x + scaled_image.shape[0],
            pad_y:pad_y + scaled_image.shape[1],
            pad_z:pad_z + scaled_image.shape[2]
        ] = scaled_image

    return output_image

FLAIR=scale(FLAIR)
print(FLAIR.shape)
FLAIR=ants.from_numpy(FLAIR)
FLAIR.plot()




