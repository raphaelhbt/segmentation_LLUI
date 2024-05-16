import ants

temp_save_location = r'/home/user/Documents/raph/temp/sub-strokecase0001_ses-0001_adc_registered.nii.gz'

path= r'/home/user/Documents/raph/temp/sub-strokecase0001_ses-0001_adc_registered.nii.gz'
path2= r'/home/user/Documents/raph/temp/sub-strokecase0001_ses-0001_dwi_registered.nii.gz'
path3= r'/home/user/Documents/raph/temp/sub-strokecase0001_ses-0001_FLAIR_registered.nii.gz'
path4= r'/home/user/Documents/raph/temp/sub-strokecase0001_ses-0001_msk_registered.nii.gz'

paths = [path, path2, path3, path4]

def zscore_normalisation(img_paths):
    new_img_paths = []
    for i in range(len(img_paths)-1): # Exclude the mask
        new_img_paths.append(img_paths[i].replace('registered.nii.gz', 'zscore.nii.gz'))
        img = ants.image_read(img_paths[i])
        
        # Extract non-zero pixels
        non_zero_pixels = img.numpy()[img.numpy() != 0]
        
        # Calculate mean and standard deviation only on non-zero pixels
        mean = non_zero_pixels.mean()
        std = non_zero_pixels.std()
        
        # Apply z-score normalization only to non-zero pixels
        img_np = img.numpy()
        img_np[img_np != 0] = (img_np[img_np != 0] - mean) / std
        
        # Create an ANTs image from the normalized numpy array
        normalized_img = ants.from_numpy(img_np, has_components=True)
        
        # Write the normalized image to the new path
        ants.image_write(normalized_img, new_img_paths[i])

zscore_normalisation(paths)

