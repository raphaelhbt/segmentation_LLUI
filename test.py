import ants

temp_save_location = r'/home/user/Documents/raph/temp/sub-strokecase0001_ses-0001_adc_registered.nii.gz'

path= r'/home/user/Documents/raph/temp/sub-strokecase0002_ses-0001_adc_zscore.nii.gz'
path2= r'/home/user/Documents/raph/temp/sub-strokecase0002_ses-0001_dwi_zscore.nii.gz'
path3= r'/home/user/Documents/raph/temp/sub-strokecase0002_ses-0001_FLAIR_zscore.nii.gz'

paths = [path, path2, path3]

for i in range(len(paths)):
    img = ants.image_read(paths[i]).numpy()
    print('mean', img.mean())
    print('std', img.std())

