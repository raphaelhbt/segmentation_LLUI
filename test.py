import pandas as pd
import os

def get_patient_ids2(file_path, preprocessed_path):
    """
    Get patient IDs from a TSV file.

    Parameters:
        file_path (str): The path to the TSV file containing patient IDs.

    Returns:
        list: A list of patient ID suffixes.
    """
    df = pd.read_csv(file_path, sep='\t')
    list_of_patients = df['participant_id'].tolist()
    j=0
    for i in list_of_patients:
        print(i)
        if not os.path.exists(os.path.join(preprocessed_path, i)):
            list_of_patients.pop(j)
            print('popped', i)
        list_of_patients[j] = list_of_patients[j].replace('sub-', "")
        j+=1
        print(f'{int(list_of_patients[j-1]):04d}')
    return list_of_patients


bids_dataset_path = "/home/user/Documents/raph/raw_datasets/SOOP"
preprocessed_path = "/home/user/Documents/raph/preprocessed_datasets/SOOP"

print(len(get_patient_ids2(os.path.join(bids_dataset_path, 'ds004889', "participants.tsv"), preprocessed_path)))


    

