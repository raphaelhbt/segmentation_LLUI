# SETTINGS
import os
import pandas as pd
import ants
import numpy as np

# On titouan's workstation do source ~/.bash_profile before running the script

# Paths definitions
# Path to the ISLES 2022 dataset and the preprocessed directory. Be careful it only works on the workstation !!!
bids_dir_WS = r"/home/user/Documents/raph/raw_datasets/ISLES-2022"
preprocessed_dir_WS = r"/home/user/Documents/raph/preprocessed_datasets/ISLES2022"

# Path to the template image used for registration
template_address = r'/home/user/Documents/raph/code/template_registration/MNI152_T1_1mm_brain.nii.gz'

# Path to the temporary directory where the preprocessed images will be saved
temp_save_location = r'/home/user/Documents/raph/temporary/ISLES_temp'

# Global variables
session_id = "0001"  # Only 0001 is available for the ISLES 2022 dataset
parameters = ['adc', 'dwi', 'FLAIR', 'msk']
choice_brain_extraction = input("Do you want to perform the brain extraction ? (Y/N) : ")   # Perform the brain extraction or not ? Y/N


# FUNCTIONS DEFINITIONS
def brain_extraction(bids_dir, parameters, temp_save_location, subject_id, session_id):

    # Retrieve the paths of the images
    img_paths= retrieve_img_paths(bids_dir, parameters, subject_id, session_id)

    # Retrieve the paths of the output images
    out_paths= retrieve_out_paths(img_paths, temp_save_location, subject_id, session_id, parameters)

    # Perform the brain extraction
    for i in range(len(img_paths)-1): # -1 because we don't want to perform brain extraction on the mask
        os.system("bet " + img_paths[i] + " " + out_paths[i] + " -R")

    return out_paths

def bias_field_correction(img_paths):
    new_img_paths = []
    for i in range(len(img_paths)-1): # Skip the mask
        new_img_paths.append(img_paths[i].replace('brainExtracted.nii.gz', 'corrected.nii.gz'))
        img = ants.image_read(img_paths[i])
        img_corrected = ants.n4_bias_field_correction(img)
        ants.image_write(img_corrected, new_img_paths[i])
    new_img_paths.append(img_paths[-1])  # Add the mask to the list
    return new_img_paths

def registration(img_paths, template_address):
    new_img_paths = []
    registered_img = []
    template = ants.image_read(template_address)

    # DWI
    registration_DWI = ants.registration(fixed=template, moving=ants.image_read(img_paths[1]), type_of_transform='Rigid', interpolator='lanczosWindowedSinc')
    registered_DWI = ants.apply_transforms(fixed=template, moving=ants.image_read(img_paths[1]), transformlist=registration_DWI['fwdtransforms'], interpolator='lanczosWindowedSinc')
    registered_img.append(registered_DWI)
    new_img_paths.append(img_paths[1].replace('corrected.nii.gz', 'registered.nii.gz'))

    # ADC
    registered_ADC = ants.apply_transforms(fixed=template, moving=ants.image_read(img_paths[0]), transformlist=registration_DWI['fwdtransforms'], interpolator='lanczosWindowedSinc')
    registered_img.append(registered_ADC)
    new_img_paths.append(img_paths[0].replace('corrected.nii.gz', 'registered.nii.gz'))

    # FLAIR (via DWI)
    registration_FLAIR2DWI = ants.registration(fixed=ants.image_read(img_paths[1]), moving=ants.image_read(img_paths[2]), type_of_transform='Rigid', interpolator='lanczosWindowedSinc')
    registered_FLAIR2DWI = registration_FLAIR2DWI['warpedmovout']
    registered_FLAIR = ants.apply_transforms(fixed=template, moving=registered_FLAIR2DWI, transformlist=registration_DWI['fwdtransforms'], interpolator='lanczosWindowedSinc')
    registered_img.append(registered_FLAIR)
    new_img_paths.append(img_paths[2].replace('corrected.nii.gz', 'registered.nii.gz'))

    # MASK
    registered_mask = ants.apply_transforms(fixed=template, moving=ants.image_read(img_paths[3]), transformlist=registration_DWI['fwdtransforms'], interpolator='nearestNeighbor')
    registered_img.append(registered_mask)
    new_img_paths.append(img_paths[3].replace('brainExtracted.nii.gz', 'registered.nii.gz'))

    # MASK to have 0 when out of the brain and not really small values (due to the interpolation)
    new_mask=create_mask(img_paths[2], output_path='/home/user/Documents/raph/temporary/ISLES_temp/mask4registration.nii.gz')
    registration_mask4registration2DWI = ants.registration(fixed=ants.image_read(img_paths[2]), moving=ants.image_read(new_mask), type_of_transform='Rigid', interpolator='nearestNeighbor')
    registered_mask4registration2DWI = registration_mask4registration2DWI['warpedmovout']
    registered_mask4registration = ants.apply_transforms(fixed=template, moving=registered_mask4registration2DWI, transformlist=registration_DWI['fwdtransforms'], interpolator='nearestNeighbor')

    # Reorder paths and images to maintain the original order
    new_img_paths[0], new_img_paths[1] = new_img_paths[1], new_img_paths[0]
    registered_img[0], registered_img[1] = registered_img[1], registered_img[0]

    for i in range(len(new_img_paths)):
        # Ensure that the registered images are resampled to match the template
        registered_img_resampled = ants.resample_image_to_target(registered_img[i], template, interp_type='lanczosWindowedSinc' if i != 3 else 'nearestNeighbor')
        registered_clean = registered_mask4registration * registered_img_resampled
        ants.image_write(registered_clean, new_img_paths[i])  

    return new_img_paths

def create_mask(image_path, output_path):
    image = ants.image_read(image_path)
    bool_mask = image.numpy() > 0
    masked_image_array = np.where(bool_mask, 1, 0).astype('float32')
    masked_image = ants.from_numpy(masked_image_array, origin=image.origin, spacing=image.spacing, direction=image.direction)
    ants.image_write(masked_image, output_path)
    return output_path

def zscore_normalisation(img_paths):
    new_img_paths = []
    for img_path in img_paths[:-1]:  # Exclude the mask
        new_path = img_path.replace('registered.nii.gz', 'zscore.nii.gz')
        new_img_paths.append(new_path)
        img = ants.image_read(img_path)
        img_np = img.numpy()

        # Calculate mean and standard deviation for the >0 values
        mask=np.where(img_np>0)

        mean = img_np[mask].mean()
        std = img_np[mask].std()

        # Initialize a copy of the image for normalized values
        normalized_img_np = np.zeros_like(img_np)

        # Perform z-score normalization only on >0 values
        normalized_img_np[mask] = (img_np[mask] - mean) / max(std, 1e-8)

        # Create an ANTs image from the normalized numpy array
        normalized_img = ants.from_numpy(normalized_img_np, spacing=img.spacing, origin=img.origin, direction=img.direction)

        # Write the normalized image to the new path
        ants.image_write(normalized_img, new_path)

    new_img_paths.append(img_paths[-1])  # Add the mask to the list
    return new_img_paths


#Helpers functions
def get_patient_ids(file_path):
    """
    Get all patient IDs from a TSV file.

    Parameters:
        file_path (str): The path to the TSV file containing patient IDs.

    Returns:
        list: A list containing all patient IDs extracted from the TSV file.
    """
    
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(file_path, sep='\t')
    
    # Extract the 'participant_id' column into a list
    patient_ids = df['participant_id'].tolist()
    
    return patient_ids

def retrieve_img_paths(bids_dir, parameters, subject_id, session_id):
    img_paths = []

    for parameter in parameters:
        if parameter == 'adc' or parameter == 'dwi':
            # Create the path to the NIfTI file (DWI & ADC)
            file_path = os.path.join(bids_dir, f"{subject_id}", f"ses-{session_id}", "dwi", f"{subject_id}_ses-{session_id}_{parameter}.nii.gz")

            img_paths.append(file_path)
    
    # Create the path to the NIfTI file (FLAIR)
    flair_path = os.path.join(bids_dir, f"{subject_id}", f"ses-{session_id}", "anat", f"{subject_id}_ses-{session_id}_FLAIR.nii.gz")
    img_paths.append(flair_path)

    # Create the path to the NIfTI file (MASK)
    mask_path = os.path.join(bids_dir, "derivatives", f"{subject_id}", f"ses-{session_id}", f"{subject_id}_ses-{session_id}_msk.nii.gz")
    img_paths.append(mask_path)

    for i in range(len(img_paths)):
        if not os.path.exists(img_paths[i]):
            print(f"Warning: NIfTI {parameter[i]} file not found for subject {subject_id} and session {session_id}. Skipping.")
            return None

    return img_paths

def retrieve_out_paths(img_paths, temp_save_location, subject_id, session_id, parameters):
    out_paths = []
    os.makedirs(temp_save_location, exist_ok=True)

    for i in range(len(img_paths)):
        out_paths.append(os.path.join(temp_save_location, f"{subject_id}_ses-{session_id}_{parameters[i]}_brainExtracted.nii.gz"))
        ants.image_write(ants.image_read(img_paths[i]), out_paths[i])
    return out_paths

def save_images(subject_id, session_id, parameters, save_location, img_paths):
    subject_id2 = subject_id
    parts = subject_id2.split("strokecase")  # Split the ID at 'strokecase'
    number_part = int(parts[1])  # Convert the numerical part to an integer to remove leading zeros
    subject_id2 = "sub-" + str(number_part)  # Construct the new ID and assign it back
    
    for i in range(len(img_paths)):
        
        if 'dwi' in img_paths[i] or 'adc' in img_paths[i]:
            save_path = os.path.join(save_location, f"{subject_id2}", f"ses-{session_id}", "dwi", f"{subject_id2}_ses-{session_id}_{parameters[i]}.nii.gz")

        elif 'FLAIR' in img_paths[i]:
            save_path = os.path.join(save_location, f"{subject_id2}", f"ses-{session_id}", "anat", f"{subject_id2}_ses-{session_id}_FLAIR.nii.gz")

        elif 'msk' in img_paths[i]:
            save_path = os.path.join(save_location, "derivatives", f"{subject_id2}", f"ses-{session_id}", f"{subject_id2}_ses-{session_id}_msk.nii.gz")

        else:
            print(f"Warning: NIfTI {parameters[i]} file not found for subject {subject_id2} and session {session_id}. Skipping.")
            return None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ants.image_write(ants.image_read(img_paths[i]), save_path)  # Save ANTs image to file



# MAIN
subject_ids = get_patient_ids(os.path.join(bids_dir_WS, 'participants.tsv'))
for subject_id in subject_ids:
    if choice_brain_extraction == "Y":

        # Perform the brain extraction only for the datasets for which it hasn't been done yet (you can comment this line if you don't want to perform the brain extraction for the dataset)
        temp_img_paths = brain_extraction(bids_dir_WS, parameters, temp_save_location, subject_id, session_id)

    else:
        bids_img_paths = retrieve_img_paths(bids_dir_WS, parameters, subject_id, session_id)
        temp_img_paths = retrieve_out_paths(bids_img_paths, temp_save_location, subject_id, session_id, parameters)

    # Perform the bias field correction
    temp_img_paths=bias_field_correction(temp_img_paths)

    # Perform the registration
    temp_img_paths=registration(temp_img_paths, template_address)

    # Perform intensity normalization
    temp_img_paths=zscore_normalisation(temp_img_paths)

    # Save the images
    save_images(subject_id, session_id, parameters, preprocessed_dir_WS, temp_img_paths)
    print(f"Subject {subject_id} preprocessed successfully.")