import pandas as pd

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

