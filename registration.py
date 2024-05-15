import ants

def register_images(adc, dwi, flair, mask, template_address):

    template = ants.image_read(template_address)

    # Create ANTs images from numpy arrays
    dwi_img = ants.from_numpy(dwi)
    adc_img = ants.from_numpy(adc)
    flair_img = ants.from_numpy(flair)
    mask_img = ants.from_numpy(mask)

    # DWI
    registration_DWI = ants.registration(fixed = template, moving = dwi_img, type_of_transform = 'Rigid')
    dwi_img_warped = registration_DWI['warpedmovout']
    
    # ADC
    adc_img_warped = ants.apply_transforms(fixed = template, moving = adc_img, interpolator = 'lanczosWindowedSinc', transformlist = registration_DWI['fwdtransforms'])
        
    # MASK
    mask_warped = ants.apply_transforms(fixed = template, moving = mask_img, interpolator = 'nearestNeighbor', transformlist = registration_DWI['fwdtransforms'])
        
    # FLAIR (via DWI)
    registration_FLAIR2DWI = ants.registration(fixed = dwi_img, moving = flair_img, type_of_transform = 'Rigid')
    
    flair_img_warped = ants.apply_transforms(fixed = template, moving = flair_img, interpolator = 'lanczosWindowedSinc', transformlist = registration_DWI['fwdtransforms'] + registration_FLAIR2DWI['fwdtransforms'])


    return adc_img_warped.numpy(), dwi_img_warped.numpy(), flair_img_warped.numpy(), mask_warped.numpy()