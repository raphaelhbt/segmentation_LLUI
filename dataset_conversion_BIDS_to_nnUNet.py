import os
import shutil
import pandas as pd
import json

def get_patient_ids(file_path):
    """
    Get the last three characters of patient IDs from a TSV file.

    Parameters:
        file_path (str): The path to the TSV file containing patient IDs.

    Returns:
        list: A list of patient ID suffixes.
    """
    df = pd.read_csv(file_path, sep='\t')
    return df['participant_id'].str[-3:].tolist()

def create_directories(base_path, subdirectories):
    """
    Create specified subdirectories within a base path.

    Parameters:
        base_path (str): The base directory path.
        subdirectories (list): A list of subdirectories to create.
    """
    for subdirectory in subdirectories:
        os.makedirs(os.path.join(base_path, subdirectory), exist_ok=True)

def move_files(patient_id, source_dirs, target_dir, file_mappings):
    """
    Move files based on mappings from source to target directories.

    Parameters:
        patient_id (str): The patient ID.
        source_dirs (list): List of source directories.
        target_dir (str): Target directory for the files.
        file_mappings (list): List of tuples with source and target file suffixes.
    """
    patient_id_str = f'{int(patient_id):04d}'
    for source_dir, (file_suffix, target_suffix) in zip(source_dirs, file_mappings):
        source_file = os.path.join(source_dir, f'sub-strokecase{patient_id_str}_ses-0001_{file_suffix}.nii.gz')
        target_file = os.path.join(target_dir, f'ISLES_{patient_id_str}_{target_suffix}.nii.gz')
        shutil.move(source_file, target_file)

def convert_dataset_to_nnUNet(bids_dataset_path, output_path, list_of_patients):
    """
    Convert a BIDS dataset to nnUNet format.

    Parameters:
        bids_dataset_path (str): Path to the BIDS dataset.
        output_path (str): Path to the output directory.
        list_of_patients (list): List of patient IDs.
    """
    dataset_name = 'Dataset011'
    nnunet_base_path = os.path.join(output_path, dataset_name)
    image_train_dir = os.path.join(nnunet_base_path, 'imagesTr')
    image_test_dir = os.path.join(nnunet_base_path, 'imagesTs')
    label_train_dir = os.path.join(nnunet_base_path, 'labelsTr')
    label_test_dir = os.path.join(nnunet_base_path, 'labelsTs')

    # Copy dataset and create necessary subdirectories
    shutil.copytree(bids_dataset_path, nnunet_base_path)
    create_directories(nnunet_base_path, ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs'])

    # Define file mappings
    file_mappings = [('FLAIR', '000'), ('adc', '001'), ('dwi', '002')]

    # Process training patients
    for patient_id in list_of_patients[:-50]:
        patient_base_dir = os.path.join(nnunet_base_path, f'sub-strokecase{int(patient_id):04d}', 'ses-0001')
        move_files(patient_id, [os.path.join(patient_base_dir, d) for d in ['anat', 'dwi', 'dwi']], image_train_dir, file_mappings)
        shutil.move(os.path.join(nnunet_base_path, 'derivatives', f'sub-strokecase{int(patient_id):04d}', 'ses-0001', f'sub-strokecase{int(patient_id):04d}_ses-0001_msk.nii.gz'),
                    os.path.join(label_train_dir, f'ISLES_{int(patient_id):04d}.nii.gz'))

    # Process testing patients
    for patient_id in list_of_patients[-50:]:
        patient_base_dir = os.path.join(nnunet_base_path, f'sub-strokecase{int(patient_id):04d}', 'ses-0001')
        move_files(patient_id, [os.path.join(patient_base_dir, d) for d in ['anat', 'dwi', 'dwi']], image_test_dir, file_mappings)
        shutil.move(os.path.join(nnunet_base_path, 'derivatives', f'sub-strokecase{int(patient_id):04d}', 'ses-0001', f'sub-strokecase{int(patient_id):04d}_ses-0001_msk.nii.gz'),
                    os.path.join(label_test_dir, f'ISLES_{int(patient_id):04d}.nii.gz'))

    # Remove unnecessary folders
    for folder in os.listdir(nnunet_base_path):
        if folder not in ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']:
            shutil.rmtree(os.path.join(nnunet_base_path, folder))

    print("Dataset conversion completed successfully!")

def create_dataset_json(output_path):
    """
    Create a dataset.json file for nnU-Net.

    Parameters:
        output_path (str): Path to the output directory.
    """
    dataset = {
        "channel_names": { "0": "FLAIR", "1": "ADC", "2": "DWI" },
        "labels": { "background": 0, "SL": 1 },
        "numTraining": 200,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    with open(os.path.join(output_path, 'dataset.json'), 'w') as f:
        json.dump(dataset, f, indent=4)

# Define paths
bids_preprocessed_dataset_path = "/home/user/Documents/raph/preprocessed_datasets/ISLES2022"
output_path = "/home/user/Documents/raph/nnUNet/nnUNet_raw"
bids_dataset_path = "/home/user/Documents/raph/raw_datasets/ISLES-2022"

# Get patient IDs and convert dataset
list_of_patients = get_patient_ids(os.path.join(bids_dataset_path, 'participants.tsv'))
convert_dataset_to_nnUNet(bids_preprocessed_dataset_path, output_path, list_of_patients)

# Create the dataset.json file
create_dataset_json(os.path.join(output_path, 'Dataset011'))
