import numpy as np
import patchify as pt
import ants

FLAIR=ants.image_read('/home/user/Documents/raph/preprocessed_datasets/ISLES2022/sub-1/ses-0001/anat/sub-1_ses-0001_FLAIR.nii.gz')
FLAIR=FLAIR.numpy()
def extract_patches(FLAIR):
    FLAIR = (FLAIR - FLAIR.min()) / (FLAIR.max() - FLAIR.min())
    print(FLAIR.min(), FLAIR.max())
    pad_sizes = [(int((max_size - img_size) / 2), int((max_size - img_size) / 2)) for max_size, img_size in zip((256, 256, 256), FLAIR.shape)]
    FLAIR_padded = np.pad(FLAIR, pad_sizes, mode='constant')

    FLAIR_patches = pt.patchify(FLAIR_padded, (128, 128, 128), step=64)

    FlAIR_patch_list = []
    coords_list = []

    for i in range(FLAIR_patches.shape[0]):
        for j in range(FLAIR_patches.shape[1]):
            for k in range(FLAIR_patches.shape[2]):
                FlAIR_patch_list.append(FLAIR_patches[i, j, k, :, :, :])
                coords_list.append((i * 64, j * 64, k * 64))  # Step size is 64

    return np.array(FlAIR_patch_list), np.array(coords_list)

FLAIR_patches, coord_patch_list = extract_patches(FLAIR)

#for i in range(len(FLAIR_patches)):
#    print(FLAIR_patches.shape)
#    ants.from_numpy(FLAIR_patches[i]).plot()

def reconstruct_segmented_image(predictions, image_shape, patch_size, coords, original_shape):
    d, h, w = image_shape
    pd, ph, pw = patch_size
    segmented_image = np.zeros(image_shape)
    count_map = np.zeros(image_shape)
    
    for (patch, (z, y, x)) in zip(predictions, coords):
        segmented_image[z:z+pd, y:y+ph, x:x+pw] += patch
        count_map[z:z+pd, y:y+ph, x:x+pw] += 1
    
    # Adjust for overlap
    segmented_image /= count_map
    print(segmented_image.shape)
    d2, h2, w2 = int((d-original_shape[0])/2), int((h-original_shape[1])/2), int((w-original_shape[2])/2)
    return segmented_image[d2:d-d2, h2:h-h2, w2:w-w2]

segmented_image = reconstruct_segmented_image(FLAIR_patches, [256, 256, 256], [128, 128, 128], coord_patch_list, FLAIR.shape)

print(segmented_image.shape)
ants.from_numpy(segmented_image).plot()