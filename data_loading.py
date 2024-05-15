import ants
import os
import numpy as np

def load_data(bids_dir, subject_id, parameters, session_id):
    
    for parameter in parameters:
        # Assuming the image file names follow a specific pattern, adjust this part as needed
        file_path = os.path.join(bids_dir, f"{subject_id}", f"ses-{session_id}", "dwi", f"{subject_id}_ses-{session_id}_{parameter}.nii.gz")

        # Check if the NIfTI file exists
        if not os.path.exists(file_path):
            print(f"Warning: NIfTI file not found for subject {subject_id}, session {session_id}, and parameter {parameter}. Skipping.")
            continue
        
        # Load the image using nibabel
        image_data = ants.image_read(file_path).numpy()
        
        # Put the data in the appropriate list
        if parameter == 'adc':
            adc_data=image_data
        elif parameter == 'dwi':
            dwi_data=image_data

    #LOADING MASK AND FLAIR IMAGES
    mask_file_path = os.path.join(bids_dir, "derivatives", f"{subject_id}", f"ses-{session_id}", f"{subject_id}_ses-{session_id}_msk.nii.gz")
    flair_file_path = os.path.join(bids_dir, f"{subject_id}", f"ses-{session_id}", "anat", f"{subject_id}_ses-{session_id}_FLAIR.nii.gz")

    # Check if the NIfTI Mask and FLAIR files exist
    if not os.path.exists(mask_file_path):
        print(f"Warning: NIfTI Mask file not found for subject {subject_id} and session {session_id}. Skipping.")
        return None, None, None, None

    if not os.path.exists(flair_file_path):
        print(f"Warning: NIfTI FLAIR file not found for subject {subject_id} and session {session_id}. Skipping.")
        return None, None, None, None

    # Load the image using nibabel
    mask_data = ants.image_read(mask_file_path).numpy()
    flair_data = ants.image_read(flair_file_path).numpy()


    
    return adc_data, dwi_data, flair_data, mask_data
