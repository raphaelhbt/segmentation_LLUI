import data_display as dd
import data_loader_all as dla
import os
import registration as reg
import get_ids as gi
import intensity_normalisation as ino

# Path to the ISLES 2022 dataset. Be careful it only works in Vitznau !!!
bids_dir_VZ = "//nslliappl01.lli.local/automated_upload/new_format/MRI_VR/database/ISLES-2022/ISLES-2022"

# Path to the ISLES 2022 dataset. Be careful it only works in Hertenstein !!!
bids_dir_HS = "//192.168.19.50/automated_upload/new_format/MRI_VR/database/ISLES-2022/ISLES-2022"

# Path to the template image used for registration
template_address = r'C:\Users\raphael.hembert\Documents\projet\code\my_code\u_net\template_registration\MNI152_T1_1mm_brain.nii.gz'

session_id = "0001"  # Only 0001 is available for the ISLES 2022 dataset

subject_ids = gi.get_patient_ids(os.path.join(bids_dir_HS, 'participants.tsv'))


#IMAGE AND MASK LOADING ALL MODALITIES
parameters = ['adc', 'dwi']
data, mask = dla.load_all_modalities_nifti_images_as_tensors(bids_dir_HS, subject_ids, parameters, session_id)


#REGISTRATION
registered_images, registered_masks = reg.register_images_and_masks2(data, mask, template_address)
dd.show_nifti_images_2D(registered_images, registered_masks, 1, session_id)
#len(registered_images), len(registered_images[1]), len(registered_masks) # Check if the images are loaded in the correct format. Should be: number of modalities, number of images per modality, number of masks


#HISTOGRAM MATCHING
matched_images = ino.histogram_matching(registered_images, template_address)

# To check if the histogram has been normalized to the template image
#stats = sitk.StatisticsImageFilter()

#stats.Execute(sitk.ReadImage(template_ad))
#print("\n Template Image")
#print( "max =", stats.GetMaximum())
#print( "min =", stats.GetMinimum())

#stats.Execute(sitk.GetImageFromArray(np.array(registered_images[0][0].numpy(), dtype=np.float32)))
#print("\n Raw Image")
#print( "max =", stats.GetMaximum())
#print( "min =", stats.GetMinimum())

#stats.Execute(matched_images[0][0])
#print("\n Matched Image")
#print( "max =", stats.GetMaximum())
#print( "min =", stats.GetMinimum())





