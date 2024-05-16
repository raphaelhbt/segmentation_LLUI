# SETTINGS
import os
import pandas as pd
import ants

# On titouan's workstation do source ~/.bash_profile before running the script

# Paths definitions
# Path to the ISLES 2022 dataset and the preprocessed directory. Be careful it only works on the workstation !!!
bids_dir_WS = r"/home/user/Documents/raph/ISLES-2022"
preprocessed_dir_WS = r"/home/user/Documents/raph/Preprocessed_images"

# Path to the template image used for registration
template_address = r'/home/user/Documents/raph/u_net/template_registration/MNI152_T1_1mm_brain.nii.gz'

# Path to the temporary directory where the preprocessed images will be saved
temp_save_location = "/home/user/Documents/raph/temp"

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

def registration(img_paths, template_address, parameters):
    new_img_paths = []
    data=[]
    # Load the template image using ANTs
    template = ants.image_read(template_address)

    # Create ANTs images from numpy arrays
    imgs = {}
    for i in range(len(img_paths)):
        imgs[f"{parameters[i]}_img"] = ants.image_read(img_paths[i])
        new_img_paths.append(img_paths[i].replace('corrected.nii.gz', 'registered.nii.gz'))
        
    # DWI
    registration_DWI = ants.registration(fixed=template, moving=imgs['dwi_img'], type_of_transform='Rigid')
    data.append(registration_DWI['warpedmovout'])

    # ADC
    data.append(ants.apply_transforms(fixed=template, moving=imgs['adc_img'], interpolator='lanczosWindowedSinc',
                                           transformlist=registration_DWI['fwdtransforms']))

    # FLAIR (via DWI)
    registration_FLAIR2DWI = ants.registration(fixed=imgs['dwi_img'], moving=imgs['FLAIR_img'], type_of_transform='Rigid')

    data.append(ants.apply_transforms(fixed=template, moving=imgs['FLAIR_img'],
                                              interpolator='lanczosWindowedSinc',
                                              transformlist=registration_DWI['fwdtransforms'] +
                                                           registration_FLAIR2DWI['fwdtransforms']))
    # MASK
    data.append(ants.apply_transforms(fixed=template, moving=imgs['msk_img'], interpolator='nearestNeighbor',
                                         transformlist=registration_DWI['fwdtransforms']))

    data[0], data[1] = data[1], data[0]  # Swap the DWI and ADC images to put them back in the right order

    for i in range(len(new_img_paths)):
        ants.image_write(data[i], new_img_paths[i])

    return new_img_paths

def zscore_normalisation(img_paths):
    new_img_paths = []
    for i in range(len(img_paths)-1): # Exclude the mask
        new_img_paths.append(img_paths[i].replace('registered.nii.gz', 'zscore.nii.gz'))
        img_np = ants.image_read(img_paths[i]).numpy()
        img_np = (img_np - img_np.mean()) / img_np.std()
        img = ants.from_numpy(img_np)
        ants.image_write(img, new_img_paths[i])
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
    for i in range(len(img_paths)):

        if 'dwi' in img_paths[i] or 'adc' in img_paths[i]:
            save_path = os.path.join(save_location, f"{subject_id}", f"ses-{session_id}", "dwi", f"{subject_id}_ses-{session_id}_{parameters[i]}.nii.gz")

        elif 'FLAIR' in img_paths[i]:
            save_path = os.path.join(save_location, f"{subject_id}", f"ses-{session_id}", "anat", f"{subject_id}_ses-{session_id}_FLAIR.nii.gz")

        elif 'msk' in img_paths[i]:
            save_path = os.path.join(save_location, "derivatives", f"{subject_id}", f"ses-{session_id}", f"{subject_id}_ses-{session_id}_msk.nii.gz")

        else:
            print(f"Warning: NIfTI {parameters[i]} file not found for subject {subject_id} and session {session_id}. Skipping.")
            return None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ants.image_write(ants.image_read(img_paths[i]), save_path)  # Save ANTs image to file



# MAIN
subject_ids = get_patient_ids(os.path.join(bids_dir_WS, 'participants.tsv'))
for subject_id in subject_ids[:5]:
    if choice_brain_extraction == "Y":

        # Perform the brain extraction only for the datasets for which it hasn't been done yet (you can comment this line if you don't want to perform the brain extraction for the dataset)
        temp_img_paths = brain_extraction(bids_dir_WS, parameters, temp_save_location, subject_id, session_id)

    else:
        bids_img_paths = retrieve_img_paths(bids_dir_WS, parameters, subject_id, session_id)
        temp_img_paths = retrieve_out_paths(bids_img_paths, temp_save_location, subject_id, session_id, parameters)

    # Perform the bias field correction
    temp_img_paths=bias_field_correction(temp_img_paths)

    # Perform the registration
    temp_img_paths=registration(temp_img_paths, template_address, parameters)

    # Perform intensity normalization
    temp_img_paths=zscore_normalisation(temp_img_paths)

    # Save the images
    save_images(subject_id, session_id, parameters, preprocessed_dir_WS, temp_img_paths)
    print(f"Subject {subject_id} preprocessed successfully.")