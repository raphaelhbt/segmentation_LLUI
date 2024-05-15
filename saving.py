import os
import torch
import ants

def save_as_ants_images(subject_id, session_id, parameters, save_location, adc, dwi, flair, mask):
    for parameter in parameters:
        # Save the data tensor to a file
        save_path_adc_dwi = os.path.join(save_location, f"{subject_id}", f"ses-{session_id}", "dwi", f"{subject_id}_ses-{session_id}_{parameter}.nii.gz")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(save_path_adc_dwi), exist_ok=True)

        if parameter == 'adc':
            adc_ants = ants.from_numpy(adc)  # Convert numpy array to ANTs image
            ants.image_write(adc_ants, save_path_adc_dwi)  # Save ANTs image to file
        elif parameter == 'dwi':
            dwi_ants = ants.from_numpy(dwi)  # Convert numpy array to ANTs image
            ants.image_write(dwi_ants, save_path_adc_dwi)  # Save ANTs image to file

    # Save the mask and flair tensors to files
    save_path_flair = os.path.join(save_location, f"{subject_id}", f"ses-{session_id}", "anat", f"{subject_id}_ses-{session_id}_FLAIR.nii.gz")
    save_path_mask = os.path.join(save_location, "derivatives", f"{subject_id}", f"ses-{session_id}", f"{subject_id}_ses-{session_id}_msk.nii.gz")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path_flair), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_mask), exist_ok=True)

    flair_ants = ants.from_numpy(flair)  # Convert numpy array to ANTs image
    mask_ants = ants.from_numpy(mask)  # Convert numpy array to ANTs image

    ants.image_write(flair_ants, save_path_flair)  # Save ANTs image to file
    ants.image_write(mask_ants, save_path_mask)  # Save ANTs image to file
