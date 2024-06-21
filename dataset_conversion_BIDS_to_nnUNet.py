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
    list_of_patients = df['participant_id'].tolist()
    for i in range(len(list_of_patients)):
        list_of_patients[i] = list_of_patients[i].replace('sub-', "")
    return list_of_patients

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
        source_file = os.path.join(source_dir, f'sub-{patient_id}_ses-0001_{file_suffix}.nii.gz')
        target_file = os.path.join(target_dir, f'SOOP_{patient_id_str}_{target_suffix}.nii.gz')
        shutil.move(source_file, target_file)

def convert_dataset_to_nnUNet(bids_dataset_path, output_path, list_of_patients):
    """
    Convert a BIDS dataset to nnUNet format.

    Parameters:
        bids_dataset_path (str): Path to the BIDS dataset.
        output_path (str): Path to the output directory.
        list_of_patients (list): List of patient IDs.
    """
    dataset_name = 'Dataset012'
    nnunet_base_path = os.path.join(output_path, dataset_name)
    image_train_dir = os.path.join(nnunet_base_path, 'imagesTr')
    image_test_dir = os.path.join(nnunet_base_path, 'imagesTs')
    label_train_dir = os.path.join(nnunet_base_path, 'labelsTr')
    label_test_dir = os.path.join(nnunet_base_path, 'labelsTs')

    # Copy dataset and create necessary subdirectories
    shutil.copytree(bids_dataset_path, nnunet_base_path)
    create_directories(nnunet_base_path, ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs'])

    # Define file mappings
    file_mappings = [('FLAIR', '0000'), ('T1w','0001'), ('ADC', '0002'), ('dwi', '0003')]

    # Process training patients
    for patient_id in list_of_patients[:-290]: # 1016 training patients but bad tsv file so we use 1016-36 because 36 patients are missing
        patient_base_dir = os.path.join(nnunet_base_path, f'sub-{int(patient_id)}', 'ses-0001')
        if not os.path.exists(os.path.join(nnunet_base_path, f'sub-{int(patient_id)}')):
            print(f'Patient {patient_id} does not exist in the dataset. Skipping...', os.path.join(patient_base_dir, f'sub-{int(patient_id)}'))
            continue
        move_files(patient_id, [os.path.join(patient_base_dir, d) for d in ['anat', 'anat', 'dwi', 'dwi']], image_train_dir, file_mappings)
        shutil.move(os.path.join(nnunet_base_path, 'derivatives', f'sub-{int(patient_id)}', 'ses-0001', f'sub-{(patient_id)}_ses-0001_msk.nii.gz'),
                    os.path.join(label_train_dir, f'SOOP_{int(patient_id):04d}.nii.gz'))
        print(f'Processed patient {patient_id}')

    # Process testing patients
    for patient_id in list_of_patients[-290:]:
        patient_base_dir = os.path.join(nnunet_base_path, f'sub-{int(patient_id)}', 'ses-0001')
        if not os.path.exists(os.path.join(nnunet_base_path, f'sub-{int(patient_id)}')):
            print(f'Patient {patient_id} does not exist in the dataset. Skipping...')
            continue
        move_files(patient_id, [os.path.join(patient_base_dir, d) for d in ['anat', 'anat', 'dwi', 'dwi']], image_test_dir, file_mappings)
        shutil.move(os.path.join(nnunet_base_path, 'derivatives', f'sub-{int(patient_id)}', 'ses-0001', f'sub-{int(patient_id)}_ses-0001_msk.nii.gz'),
                    os.path.join(label_test_dir, f'SOOP_{int(patient_id):04d}.nii.gz'))
        print(f'Processed patient {patient_id}')

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
        "channel_names": { "0": "FLAIR", "1": "T1w", "2": "ADC", "3": "DWI" },
        "labels": { "background": 0, "SL": 1 },
        "numTraining": 1016,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    with open(os.path.join(output_path, 'dataset.json'), 'w') as f:
        json.dump(dataset, f, indent=4)

# Define paths
bids_preprocessed_dataset_path = "/home/user/Documents/raph/preprocessed_datasets/SOOP"
output_path = "/home/user/Documents/raph/nnUNet/nnUNet_raw"
bids_dataset_path = "/home/user/Documents/raph/raw_datasets/SOOP"

# Get patient IDs and convert dataset
list_of_patients = get_patient_ids(os.path.join(bids_dataset_path,'ds004889', 'participants.tsv'))
convert_dataset_to_nnUNet(bids_preprocessed_dataset_path, output_path, list_of_patients)

# Create the dataset.json file
create_dataset_json(os.path.join(output_path, 'Dataset012'))
