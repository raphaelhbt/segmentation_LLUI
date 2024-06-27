import os

def replace_adc_with_ADC(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if 'adc' in file_name:
                new_name = file_name.replace('adc', 'ADC')
                old_file = os.path.join(root, file_name)
                new_file = os.path.join(root, new_name)
                os.rename(old_file, new_file)
                print(f'Renamed: {old_file} to {new_file}')

# Example usage
bids_dir = "/home/user/Documents/raph/preprocessed_datasets/ISLES2022"
replace_adc_with_ADC(bids_dir)
