import ants

def bias_field_correction(adc, dwi, flair):
    # Create ANTs images from image adress
    dwi_img = ants.from_numpy(dwi)
    adc_img = ants.from_numpy(adc)
    flair_img = ants.from_numpy(flair)

    # Compute the bias field correction
    dwi_img_corrected = ants.n4_bias_field_correction(dwi_img)
    adc_img_corrected = ants.n4_bias_field_correction(adc_img)
    flair_img_corrected = ants.n4_bias_field_correction(flair_img)

    # Convert ANTs images back to numpy arrays
    return adc_img_corrected.numpy(), dwi_img_corrected.numpy(), flair_img_corrected.numpy()
