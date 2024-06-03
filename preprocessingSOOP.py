import os
import pandas as pd
import ants
import subprocess
import gc
import time
import psutil
import numpy as np

# Paths definitions
bids_dir_WS = r"/home/user/Documents/raph/raw_datasets/SOOP/ds004889"
preprocessed_dir_WS = r"/home/user/Documents/raph/preprocessed_datasets/SOOP"
template_address = r'/home/user/Documents/raph/code/template_registration/MNI152_T1_1mm_brain.nii.gz'
temp_save_location = r'/home/user/Documents/raph/temporary/SOOP_temp'
parameters = ['ADC', 'TRACE', 'FLAIR', 'T1w', 'mask']
choice_brain_extraction = input("Do you want to perform the brain extraction ? (Y/N) : ")

def brain_extraction(bids_dir, parameters, temp_save_location, subject_id):
    img_paths = retrieve_img_paths(bids_dir, parameters, subject_id)
    if img_paths is None or len(img_paths) != len(parameters):
        raise ValueError(f"Error retrieving image paths for subject {subject_id}")
    
    out_paths = retrieve_out_paths(img_paths, temp_save_location, subject_id, parameters)
    if len(out_paths) != len(parameters):
        raise ValueError(f"Error creating output paths for subject {subject_id}")

    for i in range(1, len(img_paths)-1): 
        if parameters[i] == 'TRACE':
            os.system("bet " + img_paths[i] + " " + out_paths[i] + " -R -m")
        else:
            os.system("bet " + img_paths[i] + " " + out_paths[i] + " -R")

    if parameters[0] == 'ADC':
        mask_path = os.path.join(temp_save_location, f"{subject_id}_ses-0001_TRACE_brainExtracted_mask.nii.gz")
        command = ["fslmaths", img_paths[0], "-mul", mask_path, out_paths[0]]
        subprocess.run(command)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
    return out_paths

def bias_field_correction(img_paths, template_address):
    corrected_paths = []
    for i in range(len(img_paths)-1): 
        if i == 2 or i == 3:
            corrected_paths.append(downsample_image(img_paths[i], template_address))
        else:
            new_img_path = img_paths[i].replace('brainExtracted.nii.gz', 'corrected.nii.gz')
            img = ants.image_read(img_paths[i])
            img_corrected = ants.n4_bias_field_correction(img)
            ants.image_write(img_corrected, new_img_path)
            corrected_paths.append(new_img_path)
    corrected_paths.append(img_paths[-1])
    return corrected_paths


def registration(img_paths, template_address):
    template = ants.image_read(template_address)
    registered_paths = []

    registration_DWI = ants.registration(fixed=template, moving=ants.image_read(img_paths[1]), type_of_transform='Rigid', interpolator='lanczosWindowedSinc')
    registered_DWI = ants.apply_transforms(fixed=template, moving=ants.image_read(img_paths[1]), transformlist=registration_DWI['fwdtransforms'], interpolator='lanczosWindowedSinc')
    new_img_path = img_paths[1].replace('corrected.nii.gz', 'registered.nii.gz')
    registered_img_resampled = ants.resample_image_to_target(registered_DWI, template, interp_type='lanczosWindowedSinc')
    ants.image_write(registered_img_resampled, new_img_path)
    registered_paths.append(new_img_path)

    registration_ADC = ants.apply_transforms(fixed=template, moving=ants.image_read(img_paths[0]), transformlist=registration_DWI['fwdtransforms'], interpolator='lanczosWindowedSinc')
    new_img_path = img_paths[0].replace('corrected.nii.gz', 'registered.nii.gz')
    registered_img_resampled = ants.resample_image_to_target(registration_ADC, template, interp_type='lanczosWindowedSinc')
    ants.image_write(registered_img_resampled, new_img_path)
    registered_paths.append(new_img_path)

    registration_FLAIR2DWI = ants.registration(fixed=ants.image_read(img_paths[1]), moving=ants.image_read(img_paths[2]), type_of_transform='Rigid', interpolator='lanczosWindowedSinc')
    registered_FLAIR2DWI = registration_FLAIR2DWI['warpedmovout']
    registered_FLAIR = ants.apply_transforms(fixed=template, moving=registered_FLAIR2DWI, transformlist=registration_DWI['fwdtransforms'], interpolator='lanczosWindowedSinc')
    new_img_path = img_paths[2].replace('corrected.nii.gz', 'registered.nii.gz')
    registered_img_resampled = ants.resample_image_to_target(registered_FLAIR, template, interp_type='lanczosWindowedSinc')
    ants.image_write(registered_img_resampled, new_img_path)
    registered_paths.append(new_img_path)

    registration_T1w = ants.registration(fixed=template, moving=ants.image_read(img_paths[3]), type_of_transform='Rigid', interpolator='lanczosWindowedSinc')
    registration_T1w = ants.apply_transforms(fixed=template, moving=ants.image_read(img_paths[3]), transformlist=registration_DWI['fwdtransforms'], interpolator='lanczosWindowedSinc')
    new_img_path = img_paths[3].replace('corrected.nii.gz', 'registered.nii.gz')
    registered_img_resampled = ants.resample_image_to_target(registration_T1w, template, interp_type='lanczosWindowedSinc')
    ants.image_write(registered_img_resampled, new_img_path)
    registered_paths.append(new_img_path)

    registered_mask = ants.apply_transforms(fixed=template, moving=ants.image_read(img_paths[4]), transformlist=registration_DWI['fwdtransforms'], interpolator='nearestNeighbor')
    new_img_path = img_paths[4].replace('brainExtracted.nii.gz', 'registered.nii.gz')
    registered_img_resampled = ants.resample_image_to_target(registered_mask, template, interp_type='nearestNeighbor')
    ants.image_write(registered_img_resampled, new_img_path)
    registered_paths.append(new_img_path)

    # Reorder paths and images to maintain the original order
    registered_paths[0], registered_paths[1] = registered_paths[1], registered_paths[0]

    return registered_paths

def zscore_normalisation(img_paths):
    normalized_paths = []
    for img_path in img_paths[:-1]: 
        new_path = img_path.replace('registered.nii.gz', 'zscore.nii.gz')
        img = ants.image_read(img_path)
        img_np = img.numpy()

        mean = img_np.mean()
        std = img_np.std()

        normalized_img_np = (img_np - mean) / std
        normalized_img = ants.from_numpy(normalized_img_np, spacing=img.spacing, origin=img.origin, direction=img.direction)
        ants.image_write(normalized_img, new_path)
        normalized_paths.append(new_path)
    return normalized_paths

def get_patient_ids(file_path):
    with open(file_path, 'r') as file:
        df = pd.read_csv(file, sep='\t')
        patient_ids = df['participant_id'].tolist()
    return patient_ids

def retrieve_img_paths(bids_dir, parameters, subject_id):
    img_paths = []
    for parameter in parameters:
        if parameter == 'ADC' or parameter == 'TRACE':
            file_path = os.path.join(bids_dir, f"{subject_id}", "dwi", f"{subject_id}_rec-{parameter}_dwi.nii.gz")
            img_paths.append(file_path)
        elif parameter == 'FLAIR' or parameter == 'T1w':
            file_path = os.path.join(bids_dir, f"{subject_id}", "anat", f"{subject_id}_{parameter}.nii.gz")
            img_paths.append(file_path)
        elif parameter == 'mask':
            mask_path = os.path.join(bids_dir, "derivatives", "lesion_masks", f"{subject_id}", "dwi", f"{subject_id}_space-TRACE_desc-lesion_{parameter}.nii.gz")
            img_paths.append(mask_path)
        else:
            print(f"Warning: NIfTI {parameter} file not found for subject {subject_id}. Skipping.")
            return None
    return img_paths

def retrieve_out_paths(img_paths, temp_save_location, subject_id, parameters):
    out_paths = []
    os.makedirs(temp_save_location, exist_ok=True)

    for i in range(len(img_paths)):
        if i == 1:
            out_path = os.path.join(temp_save_location, f"{subject_id}_ses-0001_dwi_brainExtracted.nii.gz")
        else:
            out_path = os.path.join(temp_save_location, f"{subject_id}_ses-0001_{parameters[i]}_brainExtracted.nii.gz")
        ants.image_write(ants.image_read(img_paths[i]), out_path)
        out_paths.append(out_path)
    return out_paths

def save_images(subject_id, parameters, save_location, img_paths):
    for i in range(len(img_paths)):
        if 'ADC' in parameters[i]:
            save_path = os.path.join(save_location, f"{subject_id}", "ses-0001", "dwi", f"{subject_id}_ses-0001_{parameters[i]}.nii.gz")
        elif 'TRACE' in parameters[i]:
            save_path = os.path.join(save_location, f"{subject_id}", "ses-0001", "dwi", f"{subject_id}_ses-0001_dwi.nii.gz")
        elif 'FLAIR' in parameters[i] or 'T1w' in parameters[i]:
            save_path = os.path.join(save_location, f"{subject_id}", "ses-0001", "anat", f"{subject_id}_ses-0001_{parameters[i]}.nii.gz")
        elif 'mask' in parameters[i]:
            save_path = os.path.join(save_location, "derivatives", "lesion_masks", f"{subject_id}", "ses-0001", f"{subject_id}_ses-0001_msk.nii.gz")
        else:
            continue
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ants.image_write(ants.image_read(img_paths[i]), save_path)

def downsample_image(image_path, template_path):
    new_img_path = []
    new_img_path = image_path.replace('brainExtracted.nii.gz', 'corrected.nii.gz')
    img = ants.image_read(image_path)
    downsampled_img = ants.resample_image_to_target(img, ants.image_read(template_path), interp_type='lanczosWindowedSinc')
    ants.image_write(downsampled_img, new_img_path)
    return new_img_path

# MAIN
subject_ids = get_patient_ids(os.path.join(bids_dir_WS, 'participants.tsv'))
print(f"Found {len(subject_ids)} subjects in the dataset.")

for subject_id in subject_ids:
    try:
        temp_img_paths = retrieve_img_paths(bids_dir_WS, parameters, subject_id)
        if temp_img_paths is None:
            print(f"Skipping subject {subject_id}: Image paths not retrieved correctly.")
            continue

        address1 = os.path.join(bids_dir_WS, subject_id)
        address2 = os.path.join(bids_dir_WS, 'derivatives', 'lesion_masks', subject_id)

        if not os.path.exists(address1) or not os.path.exists(address2):
            print(f"Skipping subject {subject_id}: Folders do not exist at both addresses.")
            continue

        if not os.path.exists(os.path.join(address2, "dwi", f"{subject_id}_space-TRACE_desc-lesion_mask.nii.gz")):
            print(f"Skipping subject {subject_id}: Mask file not found.")
            continue

        if choice_brain_extraction == "Y":
            temp_img_paths = brain_extraction(bids_dir_WS, parameters, temp_save_location, subject_id)
            print('Brain extraction done')
        else:
            temp_img_paths = retrieve_out_paths(temp_img_paths, temp_save_location, subject_id, parameters)
            print('Brain extraction done')

        temp_img_paths = list(bias_field_correction(temp_img_paths, template_address))
        if len(temp_img_paths) != len(parameters):
            raise ValueError(f"Bias field correction did not return the expected number of paths for subject {subject_id}")
        print('Bias field correction done')

        temp_img_paths = list(registration(temp_img_paths, template_address))
        if len(temp_img_paths) != len(parameters):
            raise ValueError(f"Registration did not return the expected number of paths for subject {subject_id}")
        print('Registration done')

        temp_img_paths = list(zscore_normalisation(temp_img_paths))
        if len(temp_img_paths) != len(parameters) : #-1
            raise ValueError(f"Z-score normalization did not return the expected number of paths for subject {subject_id}")
        print('Intensity normalization done')

        save_images(subject_id, parameters, preprocessed_dir_WS, temp_img_paths)
        print(f"Subject {subject_id} preprocessed successfully.")

        for filename in os.listdir(os.path.join(preprocessed_dir_WS, f"{subject_id}", "ses-0001", "dwi")):
            # Check if 'TRACE' is in the filename
            if 'TRACE' in filename:
                # Construct the full path to the file
                file_path = os.path.join(os.path.join(preprocessed_dir_WS, f"{subject_id}", "ses-0001", "dwi"), f"{subject_id}_ses-0001_TRACE.nii.gz")
                os.remove(file_path)
    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
    
    # Release memory after processing each subject
    del temp_img_paths
    gc.collect()
    time.sleep(1)  # Small delay to ensure file system has completed operations
