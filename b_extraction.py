import os
import ants

def brain_extraction(bids_dir, parameters,save_location, subject_id, session_id):

    # Create the new BIDS
    adc_path, adc_brainExtracted_path, dwi_path, dwi_brainExtracted_path, flair_path, flair_brainExtracted_path = create_new_bids(bids_dir, parameters, save_location, subject_id, session_id)

    # Perform the brain extraction
    os.system("bet " + adc_path + " " + adc_brainExtracted_path + " -R")
    os.system("bet " + dwi_path + " " + dwi_brainExtracted_path + " -R")
    os.system("bet " + flair_path + " " + flair_brainExtracted_path + " -R")

    return adc_brainExtracted_path, dwi_brainExtracted_path, flair_brainExtracted_path




def create_new_bids(bids_dir, parameters, save_location, subject_id, session_id):
    for parameter in parameters:

        # Create the path to the NIfTI file (DWI)
        file_path = os.path.join(bids_dir, f"{subject_id}", f"ses-{session_id}", "dwi", f"{subject_id}_ses-{session_id}_{parameter}.nii.gz")
    
        # Create the new bids path
        save_path_adc_dwi = os.path.join(save_location, f"{subject_id}", f"ses-{session_id}", "dwi", f"{subject_id}_ses-{session_id}_{parameter}.nii.gz")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(save_path_adc_dwi), exist_ok=True)

        if parameter == 'adc':
            # Copy the ADC image to the new location
            adc_ants = ants.image_read(file_path)  # Read image from file
            ants.image_write(adc_ants, save_path_adc_dwi)  # Save ANTs image to file

            adc_path= file_path
            adc_brainExtracted_path = save_path_adc_dwi

        elif parameter == 'dwi':
            # Copy the DWI image to the new location
            dwi_ants = ants.image_read(file_path) # Read image from file
            ants.image_write(dwi_ants, save_path_adc_dwi) # Save ANTs image to file

            dwi_path= file_path
            dwi_brainExtracted_path = save_path_adc_dwi

        # Check if the NIfTI file exists
        if not os.path.exists(file_path):
            print(f"Warning: NIfTI file not found for subject {subject_id}, session {session_id}, and parameter {parameter}. Skipping.")
            continue

    # Create the new bids path for the flair image
    # Save the flair tensor to files
    flair_path = os.path.join(bids_dir, f"{subject_id}", f"ses-{session_id}", "anat", f"{subject_id}_ses-{session_id}_FLAIR.nii.gz")
    flair_brainExtracted_path= os.path.join(save_location, f"{subject_id}", f"ses-{session_id}", "anat", f"{subject_id}_ses-{session_id}_FLAIR.nii.gz")
    os.makedirs(os.path.dirname(flair_brainExtracted_path), exist_ok=True)

    
    # Copy the FLAIR image to the new location
    flair_ants = ants.image_read(flair_path)  # Read image from file
    ants.image_write(flair_ants, flair_brainExtracted_path)  # Save ANTs image to file

    # Check if the NIfTI file exists
    if not os.path.exists(flair_path):
        print(f"Warning: NIfTI FLAIR file not found for subject {subject_id} and session {session_id}. Skipping.")
        return None


    # Create the new bids path for the mask image
    # Save the mask tensor to files
    mask_path = os.path.join(bids_dir, "derivatives", f"{subject_id}", f"ses-{session_id}", f"{subject_id}_ses-{session_id}_msk.nii.gz")
    mask_path2 = os.path.join(save_location, "derivatives", f"{subject_id}", f"ses-{session_id}", f"{subject_id}_ses-{session_id}_msk.nii.gz")
    os.makedirs(os.path.dirname(mask_path2), exist_ok=True)

    # Copy the mask image to the new location
    mask_ants = ants.image_read(mask_path)  # Read image from file
    ants.image_write(mask_ants, mask_path2)  # Save ANTs image to file

    # Check if the NIfTI file exists
    if not os.path.exists(mask_path):
        print(f"Warning: NIfTI mask file not found for subject {subject_id} and session {session_id}. Skipping.")
        return None

    return adc_path, adc_brainExtracted_path, dwi_path, dwi_brainExtracted_path, flair_path, flair_brainExtracted_path