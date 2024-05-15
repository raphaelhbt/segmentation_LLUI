import os
import data_loading as dl
import get_ids as gi
import registration as reg
import intensity_normalisation as inorm
import saving  
import bf_correction as bfc
import b_extraction as be


# Path to the ISLES 2022 dataset and the preprocessed directory. Be careful it only works on the workstation !!!
bids_dir_WS = r"/home/user/Documents/raph/ISLES-2022"
preprocessed_dir_WS = r"/home/user/Documents/raph/Preprocessed_images"

# Path to the template image used for registration
template_address = r'/home/user/Documents/raph/u_net/template_registration/MNI152_T1_1mm_brain.nii.gz'

session_id = "0001"  # Only 0001 is available for the ISLES 2022 dataset

subject_ids = gi.get_patient_ids(os.path.join(bids_dir_WS, 'participants.tsv'))

parameters = ['adc', 'dwi']

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_location = "/home/user/Documents/raph/Preprocessed_images/"

# Perform the brain extraction or not ? Y/N
choice_brain_extraction = input("Do you want to perform the brain extraction ? (Y/N) : ")


# Preprocessing loop for images
for subject_id in subject_ids[:5]:

    if choice_brain_extraction == "Y":

        # Perform the brain extraction only for the datasets for which it hasn't been done yet (you can comment this line if you don't want to perform the brain extraction for the dataset)
        adc_path, dwi_path, flair_path = be.brain_extraction(bids_dir_WS, parameters, save_location, subject_id, session_id)

        adc, dwi, flair, mask = dl.load_data(preprocessed_dir_WS, subject_id, parameters, session_id)

    else:
        adc, dwi, flair, mask = dl.load_data(bids_dir_WS, subject_id, parameters, session_id)
 
    # Perform the bias field correction
    adc, dwi, flair = bfc.bias_field_correction(adc, dwi, flair)
    print(f"Adc tensor shape: {adc.shape}")
    
    # Perform the registration
    adc, dwi, flair, mask = reg.register_images(adc, dwi, flair, mask, template_address)
    print(f"Adc shape after registration: {adc.shape}")

    # Perform the intensity normalisation
    adc, dwi, flair = inorm.histogram_matching(adc, dwi, flair, template_address)

    saving.save_as_ants_images(subject_id, session_id, parameters, save_location, adc, dwi, flair, mask)

    # Free memory
    del adc, dwi, flair, mask
