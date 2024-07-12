import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
import ants
import numpy as np
import UNet_modelv2 as unet2
from torchvision import transforms
import random
from torch.utils.tensorboard import SummaryWriter  
import patchify as pt
import gc
from time import time
from torch.optim.lr_scheduler import PolynomialLR
import torchio as tio

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)  # Set seed for random module

# Set parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = unet2.UNet3D(in_channels=3, out_channels=1).to(DEVICE)
model.apply(unet2.InitWeights_He(neg_slope=1e-2))
ORIGINAL_SIZE = [182, 218, 182]
NUM_EPOCHS = 100
INITIAL_LEARNING_RATE = 1e-2
BATCH_SIZE = 2
PATCH_SIZE = [128, 128, 128]
STEP_SIZE = 64
optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE, momentum=0.99, nesterov=True)
power = 0.9

# Initialise the saving directory
SAVE_EVERY = 5  # Save the model every 5 epochs
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize TensorBoard writer
log_dir = 'runs/UNet3D_experiment_1'
writer = SummaryWriter(log_dir)

def RandomRotateScale(list_of_images):
    """
    Randomly rotate and scale the images with a probability of 0.16 for each transformation and 0.08 for the combined transformation.
    The angle is sampled from a uniform distribution between -30 and 30 degrees.
    The scale is sampled from a uniform distribution between 0.7 and 1.4.

    Args:
    list_of_images: list of torchio images

    Returns:
    transformed_img: list of transformed torchio images
    """
    prob_both = 0.08
    prob_scale = 0.16
    prob_rotate = 0.16

    do_both = random.random() < prob_both
    do_scale = random.random() < prob_scale
    do_rotate = random.random() < prob_rotate

    both = tio.RandomAffine(
            scales=(0.7, 1.4),
            degrees=(-30, 30, -30, 30, -30, 30),
            isotropic=True,
            default_pad_value=0
        )
    scale = tio.RandomAffine(
            scales=(0.7, 1.4),
            isotropic=True,
            default_pad_value=0
        )
    rotate = tio.RandomAffine(
            degrees=(-30, 30, -30, 30, -30, 30),
            isotropic=True,
            default_pad_value=0
        )
    if do_both:
        for img in list_of_images:
            img = both(img)
    elif do_scale:
        for img in list_of_images:
            img = scale(img)
    elif do_rotate:
        for img in list_of_images:
            img = rotate(img)
    return list_of_images

def RandomGaussianNoise(list_of_images):
    transform = tio.RandomNoise(
                mean=0, 
                std=(0, 0.1), 
                p=0.15
            )
    
    transformed = []
    for i in range(len(list_of_images) - 1): # Don't apply noise to the mask
        transformed.append(transform(list_of_images[i]))
    transformed.append(list_of_images[-1]) # Append the mask
    return transformed

def RandomGaussianBlur(list_of_images):
    """
    Add random Gaussian blur to the images with a probability of 0.2. If this augmentation
    is triggered in a sample, blurring is applied with a probability of 0.5 for each of the
    associated modalities. The kernel width is sampled from a uniform distribution between 0.5 and 1.5.

    Args:
    list_of_images: list of torchio images

    Returns:
    transformed_sample: list of transformed torchio images
    """
    sample_prob=0.2
    modality_prob=0.5
    
    do_blur = random.random() < sample_prob
    do_modality = random.random() < modality_prob

    kernel_width = random.uniform(0.5, 1.5)
    transform = tio.RandomBlur(
                    std=(kernel_width, kernel_width),
                )
    if do_blur:
        transformed_sample = []
        if do_modality:
            for i in range(len(list_of_images) - 1): # Don't apply blur to the mask
                transformed_sample.append(transform(list_of_images[i]))
            transformed_sample.append(list_of_images[-1]) # Append the mask
            return transformed_sample
    else:
        return list_of_images

def RandomBrightness(list_of_images):
    """
    Randomly adjust the brightness of the images with a probability of 0.15.
    The factor is sampled from a uniform distribution between 0.7 and 1.3.

    Args:
            list_of_images: list of torchio images
    
    Returns:
    list_of_images: list of torchio images
    """
    prob = 0.15
    factor = random.uniform(0.7, 1.3)
    if random.random() < prob:
        for i in range(len(list_of_images) - 1): # Don't apply brightness to the mask
            img = list_of_images[i].numpy().squeeze(0)
            list_of_images[i] = (img * factor) 
            list_of_images[i] = tio.ScalarImage(tensor=torch.tensor(list_of_images[i]).unsqueeze(0))
    return list_of_images

def RandomContrast(list_of_images):
    """
    Randomly adjust the contrast of the images with a probability of 0.15.
    The factor is sampled from a uniform distribution between 0.7 and 1.3.
    Following multiplication, the values are clipped to their original value range.

    Args:
    list_of_images: list of torchio images

    Returns:
    list_of_images: list of torchio images
    """
    prob = 0.15
    factor = random.uniform(0.7, 1.3)

    if random.random() < prob:
        for i in range(len(list_of_images) - 1): # Don't apply contrast to the mask
            img = list_of_images[i].numpy().squeeze(0)
            list_of_images[i] = (img * factor)
            list_of_images[i] = np.clip(list_of_images[i], img.min(), img.max())
            list_of_images[i] = tio.ScalarImage(tensor=torch.tensor(list_of_images[i]).unsqueeze(0))
    return list_of_images
 
def RandomLowResolution(list_of_images):
    """
    Randomly downsample the images with a probability of 0.25. If this augmentation is triggered in a sample,
    downsampling is applied with a probability of 0.5 for each of the associated modalities.
    The downsampling factor is sampled from a uniform distribution between 1 and 2. The interpolation method is nearest neighbor for downsampling.

    Args:
    list_of_images: list of torchio images

    Returns:
    transformed_sample: list of transformed torchio images
    """
    sample_prob = 0.25
    modality_prob = 0.5
    do_low_res = random.random() < sample_prob

    if do_low_res:
        transformed_sample = []
        factor = random.uniform(1, 2)
        for i in range(len(list_of_images)):
            do_modality = random.random() < modality_prob
            if do_modality:
                original_image = list_of_images[i]
                original_shape = original_image.shape

                # Downsample
                resample_transform = tio.Resample(
                    target=(factor, factor, factor),
                    image_interpolation='nearest',  # nearest neighbor for downsampling
                )
                downsampled = resample_transform(original_image)

                # Ensure the final shape matches the original using CropOrPad
                final_image = tio.CropOrPad(target_shape=original_shape[1:])(downsampled)
                transformed_sample.append(final_image)
            else:
                transformed_sample.append(list_of_images[i])
        return transformed_sample
    else:
        return list_of_images

def RandomGamma(list_of_images):
    """
    Randomly adjust the gamma of the images with a probability of 0.15. If this augmentation is triggered in a sample,
    gamma adjustment is applied with a probability of 0.15 for each of the associated modalities.
    The gamma factor is sampled from a uniform distribution between 0.7 and 1.5.
    The patch intensities are first normalized to [0,1] before applying the gamma transformation. With a probability of 0.15,
    the gamma transformation is applied to the inverted intensities of the image before being applied again.

    Args:
    list_of_images: list of torchio images

    Returns:
    list_of_images: list of torchio images
    """
    prob= 0.15
    prob_prior_transform = 0.15

    do_it = random.random() < prob
    do_it_prior_transform = random.random() < prob_prior_transform

    if do_it:
        transform = tio.RandomGamma(log_gamma=(0.7, 1.5))
        for i in range(len(list_of_images) - 1): # Don't apply gamma to the mask
            mask = list_of_images[i].numpy().squeeze(0) != 0

            # Normalize image to [0,1]
            img_min, img_max = list_of_images[i].numpy().min(), list_of_images[i].numpy().max()
            normalized_img = (list_of_images[i] - img_min) / (img_max - img_min + 1e-6)

            if do_it_prior_transform:
                normalized_img = 1 - transform(1 - normalized_img)
            
            normalized_img = transform(normalized_img)

            # Scale back to original value range 
            scaled_img = normalized_img * (img_max - img_min) + img_min
            clipped_img = torch.clamp(torch.tensor(scaled_img), img_min, img_max)
            list_of_images[i] = clipped_img * torch.tensor(mask)
    return list_of_images

def RandomMirror(list_of_images):
    """
    Randomly mirror the images with a probability of 0.15. If this augmentation is triggered in a sample,
    mirroring is applied with a probability of 0.5 for each of the associated modalities.

    Args:
    list_of_images: list of torchio images

    Returns:
    list_of_images: list of torchio images
    """
    # Initialize the transformation
    transform = tio.RandomFlip(axes=(0, 1, 2), p= 0.5) 
    
    # Apply the same transformation to all images
    transformed_images = []
    for i in range(len(list_of_images)):
        torch.manual_seed(0)  
        transformed_tensor = transform(list_of_images[i])  
        transformed_images.append(transformed_tensor)
        
    return transformed_images
   
# BIDS Dataset Loader
class BidsDataset(Dataset):
    """
    BIDS Dataset Loader
    
    Args:
    bids_dir: str, path to the BIDS directory
    transform: torchvision.transforms.Compose object, containing the transformations to apply
    patch_size: list, size of the patches to extract
    step_size: int, step size for patch extraction
    
    Returns:
    concatenated_data_new: torch tensor, containing the concatenated modalities of size [batch_size, 3, patch_size[0], patch_size[1], patch_size[2]], 3 for the number of modalities
    torch.tensor(mask_patches.squeeze(), dtype=torch.float32): torch tensor, containing the mask patches, of size [batch_size, patch_size[0], patch_size[1], patch_size[2]]
    np.array(coord_patch_list): numpy array, containing the coordinates of the extracted patches, of size [batch_size, 3]
    """
    def __init__(self, bids_dir, transform=None, patch_size=None, step_size=None):
        self.bids_dir = bids_dir
        self.patch_size = patch_size
        self.transform = transform
        self.subjects = self.get_subjects()
        self.step_size = step_size

    def get_subjects(self):
        subjects = []
        for subject in os.listdir(self.bids_dir):
            subject_path = os.path.join(self.bids_dir, subject)
            if os.path.isdir(subject_path) and subject != "derivatives":
                subjects.append(subject)
        return subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        FLAIR_path = os.path.join(self.bids_dir, subject, "ses-0001", "anat", f"{subject}_ses-0001_FLAIR.nii.gz")
        adc_path = os.path.join(self.bids_dir, subject, "ses-0001", "dwi", f"{subject}_ses-0001_ADC.nii.gz")
        dwi_path = os.path.join(self.bids_dir, subject, "ses-0001", "dwi", f"{subject}_ses-0001_dwi.nii.gz")
        mask_path = os.path.join(self.bids_dir, "derivatives", subject, "ses-0001", f"{subject}_ses-0001_msk.nii.gz")

        # Check if files exist
        if not all(os.path.exists(path) for path in [FLAIR_path, adc_path, dwi_path, mask_path]):
            raise FileNotFoundError("One or more files do not exist")

        # Load images
        FLAIR_img = ants.image_read(FLAIR_path, dimension=3, reorient=True).numpy()
        adc_img = ants.image_read(adc_path, dimension=3, reorient=True).numpy()
        dwi_img = ants.image_read(dwi_path, dimension=3, reorient=True).numpy()
        mask_img = ants.image_read(mask_path, dimension=3, reorient=True).numpy()

        # Apply transform if provided
        if self.transform:
            data = [adc_img, FLAIR_img, dwi_img, mask_img]
        
            for i in range(len(data)):
                data[i] = tio.ScalarImage(tensor=torch.tensor(data[i]).unsqueeze(0))

            data = self.transform(data)

            for i in range(len(data)):
                data[i] = data[i].numpy().squeeze(0)
            adc_img, FLAIR_img, dwi_img, mask_img = data

        # Normalize images after applying transforms
        FLAIR_img = (FLAIR_img - FLAIR_img.min()) / (FLAIR_img.max() - FLAIR_img.min()+1e-6)
        adc_img = (adc_img - adc_img.min()) / (adc_img.max() - adc_img.min()+1e-6)
        dwi_img = (dwi_img - dwi_img.min()) / (dwi_img.max() - dwi_img.min()+1e-6)
        mask_img = (mask_img - mask_img.min()) / (mask_img.max() - mask_img.min()+1e-6)

        # Extract patches
        FLAIR_patches, adc_patches, dwi_patches, mask_patches, coord_patch_list = self.extract_patches(FLAIR_img, adc_img, dwi_img, mask_img, PATCH_SIZE, STEP_SIZE)

        # Convert to torch tensors
        FLAIR_patches = torch.tensor(FLAIR_patches, dtype=torch.float32)
        adc_patches = torch.tensor(adc_patches, dtype=torch.float32)
        dwi_patches = torch.tensor(dwi_patches, dtype=torch.float32)
        
        # Concatenate the modalities
        concatenated_data_list = []
        for j in range(len(FLAIR_patches)):
            concatenated_data = torch.stack((FLAIR_patches[j], adc_patches[j], dwi_patches[j]), dim=0)
            concatenated_data = concatenated_data.type(torch.float32)
            concatenated_data_list.append(concatenated_data)

        concatenated_data_new = torch.stack(concatenated_data_list, dim=0)

        return (
            concatenated_data_new,
            torch.tensor(mask_patches.squeeze(), dtype=torch.float32),
            np.array(coord_patch_list)
        )

    def extract_patches(self, FLAIR, adc, dwi, mask, patch_size, step):
        """
        Extract patches from the images and the mask
        
        Args:
        FLAIR: numpy array, containing the FLAIR image
        adc: numpy array, containing the ADC image
        dwi: numpy array, containing the DWI image
        mask: numpy array, containing the mask
        
        Returns:
        FLAIR_patches: numpy array, containing the FLAIR patches
        adc_patches: numpy array, containing the ADC patches
        dwi_patches: numpy array, containing the DWI patches
        mask_patches: numpy array, containing the mask patches
        np.array(coords_list): numpy array, containing the coordinates of the extracted patches
        """
        pad_sizes = [(int((max_size - img_size) / 2), int((max_size - img_size) / 2)) for max_size, img_size in zip((256, 256, 256), FLAIR.shape)]
        FLAIR_padded = np.pad(FLAIR, pad_sizes, mode='constant') # Pad the images to 256x256x256 in order to have a constant size in the patch extraction
        adc_padded = np.pad(adc, pad_sizes, mode='constant')
        dwi_padded = np.pad(dwi, pad_sizes, mode='constant')
        mask_padded = np.pad(mask, pad_sizes, mode='constant')

        FLAIR_patches = pt.patchify(FLAIR_padded, self.patch_size, step=64) # Extract patches with a step size of 64, the returing list is of size [3, 3, 3, 128, 128, 128]
        adc_patches = pt.patchify(adc_padded, self.patch_size, step=64)
        dwi_patches = pt.patchify(dwi_padded, self.patch_size, step=64)
        mask_patches = pt.patchify(mask_padded, self.patch_size, step=64)

        FLAIR_patches = FLAIR_patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2]) # Reshape the list to have a size of [27, 128, 128, 128]
        adc_patches = adc_patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
        dwi_patches = dwi_patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
        mask_patches = mask_patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])

        coords_list = [(i * step, j * step, k * step) for i in range(3) for j in range(3) for k in range(3)] # Get the coordinates of the extracted patches    
        return FLAIR_patches, adc_patches, dwi_patches, mask_patches, np.array(coords_list)

def reconstruct_segmented_image(predictions, image_shape, coords, original_shape):
    """
    Reconstruct the segmented image from the patches

    Args:
    predictions: numpy array, containing the list of predictions patches to reconstruct 
    image_shape: list, containing the shape of the image
    coords: numpy array, containing the coordinates of the patches
    original_shape: list, containing the original shape of the image

    Returns:
    all_segmented_images: numpy array, containing the reconstructed segmented images
    """
    num_images, num_patches, pd, ph, pw = predictions.shape
    d, h, w = image_shape
    all_segmented_images = []
    
    for image_index in range(num_images):
        segmented_image = np.zeros(image_shape)
        count_map = np.zeros(image_shape)
        
        for patch_index in range(num_patches):
            patch = predictions[image_index, patch_index]
            z, y, x = coords[image_index, patch_index]
            segmented_image[z:z+pd, y:y+ph, x:x+pw] += patch
            count_map[z:z+pd, y:y+ph, x:x+pw] += 1
        
        # Adjust for overlap by dividing by the count map to get the average
        segmented_image /= count_map
        d2, h2, w2 = int((d-original_shape[0])/2), int((h-original_shape[1])/2), int((w-original_shape[2])/2)
        segmented_image = segmented_image[d2:d-d2, h2:h-h2, w2:w-w2]
        all_segmented_images.append(segmented_image)
    
    return np.array(all_segmented_images)

class BCEDiceLoss(nn.Module):
    """
    Compute the BCE Dice Loss

    Args:
    nn.Module: PyTorch module

    Returns:
    loss: float, containing the loss computed as bce + dice_loss
    dice_score: float, containing the Dice score
    bce: float, containing the BCE loss
    """
    def __init__(self, epsilon=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.epsilon = epsilon
 
    def forward(self, predictions, targets, is_validation=False):
        #Convert to float
        predictions = predictions.float()
        targets = targets.float()
        # Compute BCE Loss
        bce = self.bce_loss(predictions, targets)

        # Compute Dice Loss
        if is_validation:
            predictions = (predictions > 0.5).float() # Binarize the predictions for the validation set
        predictions_flat = predictions.view(predictions.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (predictions_flat * targets_flat).sum(1)
        sum_pred_target = predictions_flat.sum(1) + targets_flat.sum(1)
        dice = (2. * intersection + self.epsilon) / (sum_pred_target + self.epsilon)
        dice_loss = 1 - dice.mean()
 
        # Combine BCE and Dice Loss
        loss = bce + dice_loss
        dice_score = dice.mean()
 
        return loss, dice_score, bce
    
transform = transforms.Compose([
    lambda data: RandomRotateScale(data),
    lambda data: RandomGaussianNoise(data),
    lambda data: RandomGaussianBlur(data),
    lambda data: RandomBrightness(data),
    lambda data: RandomContrast(data),
    lambda data: RandomLowResolution(data),
    lambda data: RandomGamma(data),
    lambda data: RandomMirror(data)
    ])

# Initialize loss function
criterion = BCEDiceLoss()

# Initialize Dataset and DataLoader
bids_dir = "/home/user/Documents/raph/preprocessed_datasets/ISLES2022"
dataset = BidsDataset(bids_dir, transform=None, patch_size=PATCH_SIZE, step_size=STEP_SIZE)

test_size = 50
train_size = 10 #int(0.8 * len(dataset) - test_size) #160
val_size = len(dataset) - train_size - test_size #50

train_indices, val_indices = random_split(range(len(dataset)- test_size), [train_size, val_size])

train_dataset = BidsDataset(bids_dir, transform=None, patch_size=PATCH_SIZE, step_size=STEP_SIZE)
val_dataset = BidsDataset(bids_dir, transform=None, patch_size=PATCH_SIZE, step_size=STEP_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, sampler=SubsetRandomSampler(val_indices))
scheduler = PolynomialLR(optimizer, total_iters=NUM_EPOCHS, power=power)

# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_BCE = 0
    epoch_dice = 0
    epoch_loss = 0
    start_time = time()
    for batch_idx, (concatenated_data_all_patches, target, coord_patch_list) in enumerate(train_loader):
        target = target.to(DEVICE)
        #print('Batch:', batch_idx)
        for j in range(len(concatenated_data_all_patches[0])):
            concatenated_data = concatenated_data_all_patches[:,j].to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision        
            scores = model(concatenated_data)

            # Flatten predictions and targets
            predictions_flat = scores.squeeze()
            targets_flat = target[:, j].squeeze()

            # Compute Loss and Dice score
            global_loss, dice, BCE = criterion(predictions_flat, targets_flat)
            
            # Backward pass 
            global_loss.backward()

            # Optimizer step
            optimizer.step()
            
            # Accumulate loss and Dice score for the batch
            epoch_BCE += BCE.item()
            epoch_dice += dice.item()
            epoch_loss += global_loss.item()

            # Free up memory
            del concatenated_data, scores, predictions_flat, targets_flat
            torch.cuda.empty_cache()
            gc.collect()

        # if batch_idx % 10 == 0:
        #     print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], BCE: {BCE:.4f}, Dice: {dice:.4f}, Loss: {global_loss:.4f}")
    
    # Log average epoch loss and dice score
    avg_epoch_BCE = epoch_BCE / (len(train_loader)*len(concatenated_data_all_patches[0]))
    avg_epoch_dice = epoch_dice / (len(train_loader)*len(concatenated_data_all_patches[0]))
    avg_epoch_loss = epoch_loss / (len(train_loader)*len(concatenated_data_all_patches[0]))
    writer.add_scalars('Metrics/with_binarized_valid', {'Train_BCE': avg_epoch_BCE}, epoch + 1)
    writer.add_scalars('Metrics/with_binarized_valid', {'Train_Dice': avg_epoch_dice}, epoch + 1)
    writer.add_scalars('Metrics/with_binarized_valid', {'Train_Loss': avg_epoch_loss}, epoch + 1)

    end_time = time()
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Time taken: {(end_time - start_time):.2f} seconds.')
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average BCE: {avg_epoch_BCE:.4f}, Average Dice: {avg_epoch_dice:.4f}, Average Loss: {avg_epoch_loss:.4f}")
    
    del global_loss, BCE, dice, epoch_BCE, epoch_dice, epoch_loss, avg_epoch_BCE, avg_epoch_dice, avg_epoch_loss, concatenated_data_all_patches, target, coord_patch_list, start_time, end_time
    torch.cuda.empty_cache()
    gc.collect()

    # Save the model every 5 epochs
    # if (epoch + 1) % SAVE_EVERY == 0:
    #     save_path = os.path.join(MODEL_DIR, f"model_epoch_with_initialisation_full_dataset_no_dropout{epoch+1}.pth")
    #     torch.save(model.state_dict(), save_path)
    #     print(f"Model saved to {save_path}")
    
    # print("Validation started...")

    # Validation
    model.eval()
    dice_scores = []
    val_BCE = 0.0
    val_dice = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (concatenated_data_all_patches, target, coord_patch_list) in enumerate(train_loader):
            target = target.to(DEVICE)
            predictions_flat = []

            for j in range(len(concatenated_data_all_patches[0])):
                concatenated_data=concatenated_data_all_patches[:,j].to(DEVICE)

                # Forward pass
                scores = model(concatenated_data)

                scores = scores.squeeze()
                predictions_flat.append(scores.cpu())
                del scores, concatenated_data
                torch.cuda.empty_cache()
                gc.collect()

            predictions_flat = torch.stack(predictions_flat).numpy()
            predictions_flat = np.transpose(predictions_flat, (1, 0, 2, 3, 4))

            # Reconstruct the segmented image from patches
            reconstructed_image = reconstruct_segmented_image(
                predictions_flat,
                [256, 256, 256],
                coord_patch_list,
                ORIGINAL_SIZE
            )
            
            reconstructed_groundtruth = reconstruct_segmented_image(
                target.cpu().numpy(),
                [256, 256, 256],
                coord_patch_list,
                ORIGINAL_SIZE
            )
            
            # Compute the loss on the reconstructed images
            reconstructed_image_tensor = torch.tensor(reconstructed_image).to(DEVICE)
            reconstructed_gt_tensor = torch.tensor(reconstructed_groundtruth).to(DEVICE)
            loss, dice, BCE = criterion(reconstructed_image_tensor.float(), reconstructed_gt_tensor.float(), is_validation=True)

            # Compute the loss
            val_BCE += BCE
            val_dice += dice
            val_loss += loss
            # if batch_idx % 10 == 0:
            #     print(f"Batch [{batch_idx + 1}/{len(val_loader)}], Validation BCE: {BCE:.4f}, Dice Score: {dice:.4f}")

            # Free up memory
            del target, predictions_flat, reconstructed_image, reconstructed_groundtruth, reconstructed_image_tensor, reconstructed_gt_tensor, loss, dice, BCE, concatenated_data_all_patches, coord_patch_list
            torch.cuda.empty_cache()
            gc.collect()
    
    #Step the learning rate scheduler
    scheduler.step()

    # Calculate the average Dice score and validation loss
    avg_val_BCE = val_BCE / len(train_loader) #len(val_loader)
    avg_dice_score = val_dice / len(train_loader) #len(val_loader)
    avg_val_loss = val_loss / len(train_loader) #len(val_loader) 

    print(f"Average Validation Loss: {avg_val_BCE:.4f}, Average Dice Score: {avg_dice_score:.4f}")

    #Print the learning rate
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Learning Rate: {scheduler.get_last_lr()[0]:.5f}")

    # Log the average Dice score and Loss for the validation set
    writer.add_scalars('Metrics/with_binarized_valid', {'Validation_Dice': avg_dice_score}, epoch + 1)
    writer.add_scalars('Metrics/with_binarized_valid', {'Validation_BCE': avg_val_BCE}, epoch + 1)
    writer.add_scalars('Metrics/with_binarized_valid', {'Validation_Loss': avg_val_loss}, epoch + 1)
    
    # Clear cache and collect garbage
    del val_BCE, val_dice, val_loss, avg_val_BCE, avg_dice_score, avg_val_loss
    torch.cuda.empty_cache()
    gc.collect()

# Close the TensorBoard writer
writer.close()