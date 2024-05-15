import SimpleITK as sitk
import numpy as np
import ants

# Utilisation de la fonction avec le chemin de votre image NIfTI
img_before_registration = r"/home/user/Documents/raph/ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz"
img_after_registration = r"/home/user/Documents/raph/Preprocessed_images/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz"
template_address = r'/home/user/Documents/raph/u_net/template_registration/MNI152_T1_1mm_brain.nii.gz'


# To check if the histogram has been normalized to the template image
stats = sitk.StatisticsImageFilter()

stats.Execute(sitk.ReadImage(template_address))
print("\n Template Image")
print( "max =", stats.GetMaximum())
print( "min =", stats.GetMinimum())

# Convert ANTs images to numpy arrays and then to SimpleITK images
template_image_sitk = sitk.GetImageFromArray(np.array(ants.image_read(template_address).numpy(), dtype=np.float32))
img_before_registration_sitk = sitk.GetImageFromArray(np.array(ants.image_read(img_before_registration).numpy(), dtype=np.float32))
img_after_registration_sitk = sitk.GetImageFromArray(np.array(ants.image_read(img_after_registration).numpy(), dtype=np.float32))

stats.Execute(img_before_registration_sitk)
print("\n Raw Image")
print( "max =", stats.GetMaximum())
print( "min =", stats.GetMinimum())

stats.Execute(img_after_registration_sitk)
print("\n Matched Image")
print( "max =", stats.GetMaximum())
print( "min =", stats.GetMinimum())
