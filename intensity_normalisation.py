import SimpleITK as sitk
import ants
import numpy as np

def histogram_matching(adc, dwi, flair, template_address):
    # Load the template image using ANTs
    template_ants = ants.image_read(template_address)

    # Convert the ANTs image to a SimpleITK image
    template_sitk = sitk.GetImageFromArray(template_ants.numpy())

    #test
    #stats = sitk.StatisticsImageFilter()
    #stats.Execute(sitk.ReadImage(template_address))
    #print("\n Template Image")
    #print( "max =", stats.GetMaximum())
    #print( "min =", stats.GetMinimum())


    # Create SimpleITK images from numpy arrays
    dwi_img = sitk.GetImageFromArray(dwi)
    adc_img = sitk.GetImageFromArray(adc)
    flair_img = sitk.GetImageFromArray(flair)

    #test
    #stats.Execute(dwi_img)
    #print("\n Raw Image")
    #print( "max =", stats.GetMaximum())
    #print( "min =", stats.GetMinimum())

    # IMAGE HISTOGRAM MATCHING
    dwi_img_matched = sitk.HistogramMatching(dwi_img, template_sitk)
    adc_img_matched = sitk.HistogramMatching(adc_img, template_sitk)
    flair_img_matched = sitk.HistogramMatching(flair_img, template_sitk)

    #test
    #stats.Execute(dwi_img_matched)
    #print("\n Matched Image")
    #print( "max =", stats.GetMaximum())
    #print( "min =", stats.GetMinimum())

    return sitk.GetArrayFromImage(adc_img_matched), sitk.GetArrayFromImage(dwi_img_matched), sitk.GetArrayFromImage(flair_img_matched)





