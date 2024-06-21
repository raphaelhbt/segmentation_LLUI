import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import ants
import numpy as np
import UNet_model as unet
from scipy.ndimage import gaussian_filter, zoom
from skimage.transform import rotate, resize
from torchvision import transforms
import random
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter  
import shutil
import patchify as pt
from scipy.ndimage import rotate, zoom
from skimage.transform import resize
import random
import numpy as np
import gc
from time import time
from torch.optim.lr_scheduler import PolynomialLR


# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)  # Set seed for random module

# Set parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet.UNet3D(in_channels=3, out_channels=1).to(DEVICE)  # Adjust in_channels for 3 input modalities
ORIGINAL_SIZE = [182, 218, 182]
NUM_EPOCHS = 100
INITIAL_LEARNING_RATE = 1e-2
BATCH_SIZE = 2
PATCH_SIZE = [128, 128, 128]
STEP_SIZE = 64
optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE, momentum=0.99, nesterov=True)
threshold = 0.5
power = 0.9
# Initialize TensorBoard writer
log_dir = 'runs/UNet3D_experiment_1'
writer = SummaryWriter(log_dir)

class RandomRotateScale:
    '''Rotate with prob=0.16, scale with prob 0.16 and both with prob 0.08. Rotation following each axis with each angle=random.uniform(-30, 30). 
    Scale = np.random.uniform(0.7, 1.4), if<1 then scaling and then padding to original size, else scaling and then cropping.'''
    def __init__(self, rotate_prob=0.16, scale_prob=0.16, both_prob=0.08):
        self.rotate_prob = rotate_prob
        self.scale_prob = scale_prob
        self.both_prob = both_prob

    def rotate_3d(self, img, angle_x, angle_y, angle_z):
        # Rotate around x-axis
        for i in range(img.shape[1]):
            img[:, i, :] = rotate(img[:, i, :], angle_x, reshape=False, mode='reflect')
        # Rotate around y-axis
        for i in range(img.shape[0]):
            img[i, :, :] = rotate(img[i, :, :], angle_y, reshape=False, mode='reflect')
        # Rotate around z-axis
        for i in range(img.shape[2]):
            img[:, :, i] = rotate(img[:, :, i], angle_z, reshape=False, mode='reflect')
        return img

    def scale(self, image):
        # Generate a random scale parameter
        scale = np.random.uniform(0.7, 1.4)

        # Scale the image
        scaled_image = zoom(image, scale)

        # Initialize an empty array with the original image size
        output_image = np.zeros_like(image)

        # If the scaled image is larger than the original image, crop it
        if scale > 1:
            start_x = (scaled_image.shape[0] - image.shape[0]) // 2
            start_y = (scaled_image.shape[1] - image.shape[1]) // 2
            start_z = (scaled_image.shape[2] - image.shape[2]) // 2

            output_image = scaled_image[
                start_x:start_x + image.shape[0],
                start_y:start_y + image.shape[1],
                start_z:start_z + image.shape[2]
            ]
        # If the scaled image is smaller than the original image, pad it
        else:
            pad_x = (image.shape[0] - scaled_image.shape[0]) // 2
            pad_y = (image.shape[1] - scaled_image.shape[1]) // 2
            pad_z = (image.shape[2] - scaled_image.shape[2]) // 2

            output_image[
                pad_x:pad_x + scaled_image.shape[0],
                pad_y:pad_y + scaled_image.shape[1],
                pad_z:pad_z + scaled_image.shape[2]
            ] = scaled_image
        gc.collect()  
        return output_image

    def __call__(self, sample):
        if not isinstance(sample, list):
            raise TypeError("Sample must be a list or iterable.")
        
        do_rotate = random.random() < self.rotate_prob
        do_scale = random.random() < self.scale_prob
        do_both = random.random() < self.both_prob

        if do_rotate or do_both:
            angle_x = random.uniform(-30, 30)
            angle_y = random.uniform(-30, 30)
            angle_z = random.uniform(-30, 30)

            for i in range(len(sample)):
                sample[i] = self.rotate_3d(sample[i], angle_x, angle_y, angle_z)

        if do_scale or do_both:
            for i in range(len(sample)):
                img = sample[i]
                # Resize image
                sample[i] = self.scale(img)

        return sample

class RandomGaussianNoise:
    '''prob=0.15, generates a gausian blur with variance = random.uniform(0, 0.1) and mean = 0'''
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            for i in range(len(sample)):
                variance = random.uniform(0, 0.1)
                noise = np.random.normal(0, variance, sample[i].shape)
                sample[i] = sample[i] + noise
        return sample

class RandomGaussianBlur:
    '''prob of 0.2. If it happens for one modality then prob of 0.5 that it also happens to the other modalities of the same patient.  
    sigma = random.uniform(0.5, 1.5) and sample[j] = gaussian_filter(sample[j], sigma).'''
    def __init__(self, sample_prob=0.2, modality_prob=0.5):
        self.sample_prob = sample_prob
        self.modality_prob = modality_prob

    def __call__(self, sample):
        do_blur = random.random() < self.sample_prob

        for j in range(len(sample)):
            if do_blur:
                sigma = random.uniform(0.5, 1.5)
                sample[j] = gaussian_filter(sample[j], sigma)
                for i in range(len(sample)):
                    if random.random() < self.modality_prob and i != j:
                        sigma = random.uniform(0.5, 1.5)
                        sample[i] = gaussian_filter(sample[i], sigma)
                break  # Exit the loop after the iteration of j when do_blur is True
                
        return sample

class RandomBrightness:
    '''prob=0.15, factor = random.uniform(0.7, 1.3) and then values multiplied by factor.'''
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            factor = random.uniform(0.7, 1.3)
            for i in range(len(sample)):
                sample[i] = sample[i] * factor
        return sample

class RandomContrast:
    '''prob=0.15, factor = random.uniform(0.65, 1.5) then multiplication between factor and original values, finally, values are clipped between min and max of the original image.'''
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        new_sample = sample
        if random.random() < self.prob:
            factor = random.uniform(0.65, 1.5)
            new_sample = []
            for i in range(len(sample)):
                new_sample.append(sample[i] * factor)
                new_sample[i] = np.clip(new_sample[i], sample[i].min(), sample[i].max())
        return new_sample

class RandomLowResolution:
    '''prob=0.25, if happens for one modality then prob of 0.5 that happens to the others of the patient. factor = random.uniform(1, 2)    
    downsampled = zoom(sample[j], 1 / factor, order=0) and then resizing to original size'''
    def __init__(self, sample_prob=0.25, modality_prob=0.5):
        self.sample_prob = sample_prob
        self.modality_prob = modality_prob

    def __call__(self, sample):
        for j in range(len(sample)):
            if random.random() < self.sample_prob:
                factor = random.uniform(1, 2)
                downsampled = zoom(sample[j], 1 / factor, order=0)
                sample[j] = resize(downsampled, sample[j].shape, order=3, mode='reflect', anti_aliasing=True)
                 
                for i in range(len(sample)):
                    if random.random() < self.modality_prob and i != j:
                        factor = random.uniform(1, 2)
                        downsampled = zoom(sample[i], 1 / factor, order=0)
                        sample[i] = resize(downsampled, sample[i].shape, order=3, mode='reflect', anti_aliasing=True)
                break  # Exit the loop after the iteration of j when do_blur is True
                
        return sample

class RandomGamma:
    '''prob=0.15, intensity is normalized, non linear intensity transformation i_new=i_old**g with g=random.uniform(0.7, 1.5) and then voxel intensity is scaled back to their original value range. 
    Also prob2=0.15 and if it is triggered then voxel intensities being inverted prior to transformation with sample[i] = 1 - ((1 - sample[i]) ** g)'''
    def __init__(self, prob=0.15, prob2=0.15):
        self.prob = prob
        self.prob2 = prob2

    def __call__(self, sample):
        if random.random() < self.prob:
            g = random.uniform(0.7, 1.5)
            for i in range(len(sample)):
                min_val, max_val = np.min(sample[i]), np.max(sample[i])
                sample[i] = (sample[i] - min_val) / (max_val - min_val + 1e-6)
                if random.random() < self.prob2:
                    sample[i] = 1 - (1 - sample[i]) ** g
                sample[i] = sample[i] ** g
                sample[i] = (sample[i] * (max_val - min_val)) + min_val
        return sample

class RandomMirror:
    '''prob=0.5, if triggered, all patches are mirrored'''
    def __init__(self):
        self.prob = 0.5

    def __call__(self, sample):
        if random.random() < self.prob:
            axes = [0, 1, 2]
            for i in range(len(sample)):
                for axis in axes:
                    if random.random() < self.prob:
                        sample[i] = np.flip(sample[i], axis=axis).copy()
        return sample
    
# BIDS Dataset Loader
class BidsDataset(Dataset):
    '''BIDS Dataset Loader for 3D MRI images. The dataset should be organized in the BIDS format.
    Returns 4 torch tensors: FLAIR, ADC, DWI, and the mask. The mask is a binary mask where 1 represents the lesion and 0 the background.
    Also returns a list of coordinates for each patch.
    The_size of the output images is a list of 5 integers: [batch_size, number of patches, patch_depth, patch_height, patch_width]. So here it is [2, 27, 128, 128, 128]'''
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
        FLAIR_img = ants.image_read(FLAIR_path).numpy()
        adc_img = ants.image_read(adc_path).numpy()
        dwi_img = ants.image_read(dwi_path).numpy()
        mask_img = ants.image_read(mask_path).numpy()

        # Apply transform if provided
        if self.transform:
            data = [FLAIR_img, adc_img, dwi_img, mask_img]
            data = self.transform(data)
            FLAIR_img, adc_img, dwi_img, mask_img = data

        # Normalize images again after applying transforms
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

        concatenated_data_list = []
        # Concatenate the modalities
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
        '''Extract patches of size 128x128x128 with a step size of 64. The patches are extracted from the images padded to 256x256x256, padding is equal on each side of the image.'''
        pad_sizes = [(int((max_size - img_size) / 2), int((max_size - img_size) / 2)) for max_size, img_size in zip((256, 256, 256), FLAIR.shape)]
        FLAIR_padded = np.pad(FLAIR, pad_sizes, mode='constant')
        adc_padded = np.pad(adc, pad_sizes, mode='constant')
        dwi_padded = np.pad(dwi, pad_sizes, mode='constant')
        mask_padded = np.pad(mask, pad_sizes, mode='constant')

        FLAIR_patches = pt.patchify(FLAIR_padded, self.patch_size, step=64)
        adc_patches = pt.patchify(adc_padded, self.patch_size, step=64)
        dwi_patches = pt.patchify(dwi_padded, self.patch_size, step=64)
        mask_patches = pt.patchify(mask_padded, self.patch_size, step=64)

        FLAIR_patches = FLAIR_patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
        adc_patches = adc_patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
        dwi_patches = dwi_patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
        mask_patches = mask_patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])

        coords_list = [(i * step, j * step, k * step) for i in range(3) for j in range(3) for k in range(3)]        
        return FLAIR_patches, adc_patches, dwi_patches, mask_patches, np.array(coords_list)

def reconstruct_segmented_image(predictions, image_shape, coords, original_shape):
    '''Reconstruct the segmented image from patches with help of the coordinates at which the patches were taken. The image_shape is the shape of the original image'''
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
        
        # Adjust for overlap
        segmented_image /= count_map
        d2, h2, w2 = int((d-original_shape[0])/2), int((h-original_shape[1])/2), int((w-original_shape[2])/2)
        segmented_image = segmented_image[d2:d-d2, h2:h-h2, w2:w-w2]
        all_segmented_images.append(segmented_image)
    
    return np.array(all_segmented_images)

class BCEDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.epsilon = epsilon
 
    def forward(self, predictions, targets):
        #Convert to float
        predictions = predictions.float()
        targets = targets.float()
        # Compute BCE Loss
        bce = self.bce_loss(predictions, targets)

        # Compute Dice Loss
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
    RandomRotateScale(),
    RandomGaussianNoise(),
    RandomGaussianBlur(),
    RandomBrightness(),
    RandomContrast(),
    RandomLowResolution(),
    RandomGamma(), 
    RandomMirror(),
])

# Initialize loss function
criterion = BCEDiceLoss()

# Initialize Dataset and DataLoader
bids_dir = "/home/user/Documents/raph/preprocessed_datasets/ISLES2022"
dataset = BidsDataset(bids_dir, transform=transform, patch_size=PATCH_SIZE, step_size=STEP_SIZE)
 
train_size = int(0.8 * len(dataset)) #200
val_size = len(dataset) - train_size #50
train_indices, val_indices = random_split(range(len(dataset)), [train_size, val_size])
 
train_dataset = BidsDataset(bids_dir, transform=transform, patch_size=PATCH_SIZE, step_size=STEP_SIZE)
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
        targets = target.to(DEVICE)
        #print('Batch:', batch_idx)
        for j in range(len(concatenated_data_all_patches[0])):
            concatenated_data = concatenated_data_all_patches[:,j].to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            scores = model(concatenated_data)
            
            # Flatten predictions and targets
            predictions_flat = scores.squeeze()
            targets_flat = targets[:, j].squeeze()

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

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], BCE: {BCE:.4f}, Dice: {dice:.4f}, Loss: {global_loss:.4f}")
    
    # Log average epoch loss and dice score
    avg_epoch_BCE = epoch_BCE / (len(train_loader)*len(concatenated_data_all_patches[0]))
    avg_epoch_dice = epoch_dice / (len(train_loader)*len(concatenated_data_all_patches[0]))
    avg_epoch_loss = epoch_loss / (len(train_loader)*len(concatenated_data_all_patches[0]))
    writer.add_scalars('Metrics', {'Train_BCE': avg_epoch_BCE}, epoch)
    writer.add_scalars('Metrics', {'Train_Dice': avg_epoch_dice}, epoch)
    writer.add_scalars('Metrics', {'Train_Loss': avg_epoch_loss}, epoch)

    end_time = time()
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Time taken: {(end_time - start_time):.2f} seconds.')
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average BCE: {avg_epoch_BCE:.4f}, Average Dice: {avg_epoch_dice:.4f}, Average Loss: {avg_epoch_loss:.4f}")
    
    del global_loss, BCE, dice, epoch_BCE, epoch_dice, epoch_loss
    torch.cuda.empty_cache()
    gc.collect()

    print("Validation started...")

    # Validation
    model.eval()
    dice_scores = []
    val_BCE = 0.0
    val_dice = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (concatenated_data_all_patches, target, coord_patch_list) in enumerate(val_loader):
            targets = target.to(DEVICE)
            predictions_flat = []

            for j in range(len(concatenated_data_all_patches[0])):
                concatenated_data=concatenated_data_all_patches[:,j].to(DEVICE)

                # Forward pass
                scores = model(concatenated_data)
                scores = scores.squeeze()
                predictions_flat.append(scores.cpu())
                del scores, concatenated_data

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
            loss, dice, BCE = criterion(reconstructed_image_tensor.float(), reconstructed_gt_tensor.float())

            # Compute the loss
            val_BCE += BCE
            val_dice += dice
            val_loss += loss
            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(val_loader)}], Validation BCE: {BCE:.4f}, Dice Score: {dice:.4f}")

            # Free up memory
            del target, predictions_flat, reconstructed_image, reconstructed_groundtruth, reconstructed_image_tensor, reconstructed_gt_tensor
            torch.cuda.empty_cache()
            gc.collect()
    
    #Step the learning rate scheduler
    scheduler.step()

    # Calculate the average Dice score and validation loss
    avg_val_BCE = val_BCE / len(val_loader)
    avg_dice_score = val_dice / len(val_loader)
    avg_val_loss = val_loss / len(val_loader) 

    print(f"Average Validation Loss: {avg_val_BCE:.4f}, Average Dice Score: {avg_dice_score:.4f}")

    # Print the learning rate
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Learning Rate: {scheduler.get_last_lr()[0]:.5f}")

    # Log the average Dice score and Loss for the validation set
    writer.add_scalars('Metrics', {'Validation_Dice': avg_dice_score}, epoch)
    writer.add_scalars('Metrics', {'Validation_BCE': avg_val_BCE}, epoch)
    writer.add_scalars('Metrics', {'Validation_Loss': avg_val_loss}, epoch)
    
    # Clear cache and collect garbage
    del val_BCE, val_dice, val_loss, avg_val_BCE, avg_dice_score, avg_val_loss
    torch.cuda.empty_cache()
    gc.collect()

# Close the TensorBoard writer
writer.close()