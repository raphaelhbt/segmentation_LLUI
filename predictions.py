import torch
import pandas as pd
import os
import ants
from torch.utils.data import DataLoader
from monai.networks.nets import DynUNet
from pathlib import Path
import torchio as tio
from torch.utils.data import random_split
from tqdm import tqdm
import torch.nn as nn
import numpy as np

# Parameters
NB_FORWARD = 200
dropout=0.5
BATCH_SIZE = 2
weights_path = '/home/user/Documents/raph/code/saved_models/best_model_UNet_StrokeLesion_nnUNet_aug_new_loss_20_patches_new_label_sampler_adam_dropout_200_epochs.pth'

bids_dir = Path('/home/user/Documents/raph/preprocessed_datasets/ISLES2022')
parameters = ['FLAIR', 'ADC', 'dwi', 'msk']
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Model instantiation ------------------------------------------------------------------------------------------
# Define network parameters
spatial_dims = 3
in_channels = 3
out_channels = 1
kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
up_sample_kernel_size = strides[1:]
filters = [32, 64, 128, 256, 320]
dropout = 0.5
# default params
# - norm_name: instance
# - act_name: leaky relu (negative_slope 0.01)
    
# Initialize the DynUNet
model = DynUNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    strides=strides,
    upsample_kernel_size=up_sample_kernel_size,
    filters=filters,
    dropout = dropout
)
# CORRECT THE DROPOUT
# Function to remove a specified dropout layer
def remove_dropout_layers(model, layer_indices):
    # Flatten all model layers
    layers = [module for module in model.modules()]
    
    # Find all dropout layers
    dropout_layers = [layer for layer in layers if isinstance(layer, nn.Dropout)]
    # Check if the specified layer indices are valid
    for index in layer_indices:
        if index < 1 or index > len(dropout_layers):
            raise ValueError(f"Invalid layer index: {index}. Must be between 1 and {len(dropout_layers)}")
    
    # Remove the specified dropout layers
    for layer_index in layer_indices:
        # Get the dropout layer to remove
        dropout_layer = dropout_layers[layer_index - 1]
        
        # Remove the dropout layer from the model
        for name, module in model.named_modules():
            if module == dropout_layer:
                parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
                for key, value in parent_module._modules.items():
                    if value == dropout_layer:
                        del parent_module._modules[key]
                        break

list_dropout_layers_to_remove = [1,3,5,7,9,11,12,14,15,17,18,20,21,22,23]
remove_dropout_layers(model, list_dropout_layers_to_remove)

# Loading the weights
state_dict = torch.load(weights_path)

# Loading the weights into the model
model.load_state_dict(state_dict)
model.to(DEVICE)
# ------------------------------------------------------------------------------------------------------------




# Dataset and DataLoader --------------------------------------------------------------------------------------
bids_dir = Path('/home/user/Documents/raph/preprocessed_datasets/ISLES2022')

# List all directories in the parent directory that start with 'sub-'
sub_folders = [p.name for p in bids_dir.iterdir() if p.is_dir() and p.name.startswith('sub-')]

# loop over all subjects to get relevant images/labels         
subjects = []
for subject_id in sub_folders:

    # get all images and labelmap of subject
    FLAIR_path = os.path.join(bids_dir, subject_id, "ses-0001", "anat", f"{subject_id}_ses-0001_FLAIR.nii.gz")
    dwi_path = os.path.join(bids_dir, subject_id, "ses-0001", "dwi", f"{subject_id}_ses-0001_dwi.nii.gz")
    adc_path = os.path.join(bids_dir, subject_id, "ses-0001", "dwi", f"{subject_id}_ses-0001_ADC.nii.gz")
    mask_path = os.path.join(bids_dir, "derivatives", subject_id, "ses-0001", f"{subject_id}_ses-0001_msk.nii.gz")

    # create a subject
    subject = tio.Subject(
        flair=tio.ScalarImage(FLAIR_path),
        dwi=tio.ScalarImage(dwi_path),
        adc=tio.ScalarImage(adc_path),
        label=tio.LabelMap(mask_path),
    )

    # add subject to list
    subjects.append(subject)

# create the SubjectsDataset
dataset = tio.SubjectsDataset(subjects, transform=None)

# split data into training and validation set
test_size = 50
train_percent = 0.8
train_size = int(train_percent * (len(dataset) - 50)) #160
val_size = len(dataset) - train_size - test_size #40

train_indices, val_indices = random_split(range(len(dataset) - 50), [train_size, val_size])


def filter_global_list(global_list, portion_list1, portion_list2):
    # Combine portion_list1 and portion_list2 into a set for efficient lookup
    portions_set = set(portion_list1 + portion_list2)
    
    # Filter out elements from global_list that are present in portions_set
    filtered_list = [item for idx, item in enumerate(global_list) if idx not in portions_set]
    
    return filtered_list

test_subjects= filter_global_list(subjects, train_indices, val_indices)

test_dataset = tio.SubjectsDataset(test_subjects, transform=None)
# ------------------------------------------------------------------------------------------------------------

# Function to conmpute uncertainty estimation ----------------------------------------------------------------

# Dropout activation and deactivation
def enable_dropout(model):
	""" Function to enable the dropout layers during test-time """
	for m in model.modules():
		if m.__class__.__name__.startswith('Dropout'):
			m.train()

# Function to compute the volume of the '1' class in a 3D mask
def compute_volume(mask):
    """
    Computes the volume of the '1' class in a 3D mask.
    
    Parameters:
    mask (numpy.ndarray): A 3D NumPy array with values 0 and 1.
    
    Returns:
    int: The volume of the '1' class in the mask.
    """
    mask_np = mask.cpu().numpy()
    volume = np.sum(mask_np) + 1 # Add 1 to avoid division by zero
    return volume

savepath = Path('uncertainty_predictions')

# Function to save the images
def saving_images(subject, predictions, savedir, kind, mask_volume=None):
    # kind = 'mean' or 'std'
    # Create the directory if it does not exist
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # Extract the patient number from the subject path:
    patient_number = subject['flair'].path.parts[-1]

    #####################
    if mask_volume is not None:
        # Compute the volume of the '1' class in the mask
        mean_value = torch.mean(predictions)
        print('mean', mean_value)
        ratio = mean_value / mask_volume
        # Create the output path
        output_savepath = os.path.join(savedir, f'{patient_number}_{kind}_mean_{mean_value:.5f}_volume_{mask_volume:.2f}_ratio_{ratio:.10f}.nii.gz')
    #####################
    else:
        # Create the output path
        output_savepath = os.path.join(savedir, f'{patient_number}_{kind}.nii.gz')

    # Extract the affine matrix from the subject's flair image
    affine_matrix = subject['flair'].affine
    # Save the image
    uncertainty_image = tio.ScalarImage(tensor = predictions, affine=affine_matrix)
    uncertainty_image.save(output_savepath)
    
# ------------------------------------------------------------------------------------------------------------

# Test loop ----------------------------------------------------------------------------------------------
# Set the model to evaluation mode
model.eval()

# predict with dropout
enable_dropout(model)

# batch size used in validation
batch_size_test = 4

# patch size and overlap for gridSampler used in validation
patch_size_test= 128
patch_overlap_test = 64

# Iterate over the test dataset
with torch.no_grad():
    # loop over all subject in validation set
    for subject in tqdm(test_dataset):
        # define GridSampler for current subject
        grid_sampler = tio.inference.GridSampler(
            subject = subject,  
            patch_size = patch_size_test,
            patch_overlap = patch_overlap_test,
            padding_mode = 'constant'
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size = batch_size_test, shuffle = True)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode = 'hann')

        # to store n_forward predictions on the same batch
        dropout_predictions = torch.empty((0, 1, 182, 218, 182))

        # loop over patches to get mean model predictions
        for f_pass in range(NB_FORWARD):
            for patches_batch in patch_loader:
                patch_inputs = torch.cat([patches_batch['flair'][tio.DATA], patches_batch['dwi'][tio.DATA], patches_batch['adc'][tio.DATA]], dim=1).to(DEVICE)
                patch_locations = patches_batch[tio.LOCATION]
                with torch.no_grad():
                    patch_prediction = model(patch_inputs)

                # aggregate the mean over patches
                patch_prediction_logits = torch.sigmoid(patch_prediction)
                aggregator.add_batch(patch_prediction_logits, patch_locations)

            # concatenate prediction to the other made on the same batch
            dropout_predictions = torch.cat((dropout_predictions, aggregator.get_output_tensor().cpu().unsqueeze(dim=0)),dim=0) # Output shape is (n_forward, batch_size, 128, 128, 128)

        #####################
        # Compute the volume of the '1' class in the mask
        mask = dropout_predictions.mean(dim=0)
        # Binarize the mask to 0 or 1 where the threshold is 0.5
        mask[mask >= 0.5] = 1
        mask_volume = compute_volume(mask)
        print('volume', mask_volume)
        #####################

        # save the mean and std of the predictions
        saving_images(subject, dropout_predictions.mean(dim=0), savepath, 'mean')
        saving_images(subject, dropout_predictions.std(dim=0), savepath, 'std', mask_volume)

        print('one done')