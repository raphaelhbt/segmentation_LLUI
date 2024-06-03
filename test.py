import os

preprocessed_dir_WS = r"/home/user/Documents/raph/preprocessed_datasets/ISLES2022"

def rename_adc_files(base_path):
    # Walk through all directories and files in the base_path
    for dirpath, dirnames, filenames in os.walk(base_path):
        # Rename files containing 'adc'
        for filename in filenames:
            if 'adc' in filename:
                new_filename = filename.replace('adc', 'ADC')
                old_file = os.path.join(dirpath, filename)
                new_file = os.path.join(dirpath, new_filename)
                os.rename(old_file, new_file)

# Example usage
rename_adc_files(preprocessed_dir_WS)

