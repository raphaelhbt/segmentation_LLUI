import ants
import matplotlib.pyplot as plt
import data_display as dd
# Utilisation de la fonction avec le chemin de votre image NIfTI
img_before_registration = r"/home/user/Documents/raph/ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz"
img_after_registration = r"/home/user/Documents/raph/Preprocessed_images/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz"
template_address = r'/home/user/Documents/raph/u_net/template_registration/MNI152_T1_1mm_brain.nii.gz'
temp_save_location = r'/home/user/Documents/raph/temp/sub-strokecase0001_ses-0001_adc_corrected.nii.gz'

# Load the images
img = ants.image_read(img_before_registration)
img_corrected = ants.image_read(temp_save_location)


image= img - img_corrected

while True:
        # Prompt user to input slice index
        user_input = input(f"Choose slice index (0-{image.shape[-1] - 1}), 'q' to quit: ")

        # If user quits, break out of the loop
        if user_input.lower() == 'q':
            break
        
        try:
            # Convert user input to integer
            slice_index = int(user_input)
            
            # Display selected slice
            if 0 <= slice_index < image.shape[-1]:
                plt.imshow(image[:,:,slice_index], cmap="gray")
                plt.axis("off")
                plt.title(f"Slice {slice_index}")
                plt.show()
            else:
                print(f"Invalid input. Slice index must be between 0 and {image.shape[-1] - 1}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


