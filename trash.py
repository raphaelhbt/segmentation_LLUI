import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)  # Set seed for random module

# Set parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet.UNet3D(in_channels=3, out_channels=1).to(DEVICE)  # Adjust in_channels for 3 input modalities
ORIGINAL_SIZE = [182, 218, 182]
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
PATCH_SIZE = [2, 2, 128, 128, 128]
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
threshold = 0.5

# Initialize TensorBoard writer
log_dir = 'runs/UNet3D_experiment_1'
writer = SummaryWriter(log_dir)

class RandomRotateScale:
    def __init__(self, patch_size, rotate_prob=0.16, scale_prob=0.16, both_prob=0.08):
        self.patch_size = patch_size
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
                print('sample rotate', sample[i].shape, i)

        if do_scale or do_both:
            scale = random.uniform(0.7, 1.4)
            new_size = [sample[0].shape[0]] + [int(dim * scale) for dim in sample[0].shape[1:]]

            for i in range(len(sample)):
                resized_sample = resize(sample[i], new_size, order=2, anti_aliasing=True)
                if scale > 1:  # Crop to original size
                    start_indices = [0] + [(new_dim - orig_dim) // 2 for new_dim, orig_dim in zip(new_size[1:], self.patch_size[1:])]
                    sample[i] = resized_sample[start_indices[0]:start_indices[0]+self.patch_size[0],
                                            start_indices[1]:start_indices[1]+self.patch_size[1],
                                            start_indices[2]:start_indices[2]+self.patch_size[2]]
                else:  # Pad to original size
                    pad_width = [(0, 0)] + [(orig_dim - new_dim) // 2 for new_dim, orig_dim in zip(new_size[1:], self.patch_size[1:])]
                    sample[i] = np.pad(resized_sample, pad_width, mode='constant')
                print('sample scale', sample[i].shape, i)
        return sample

class RandomGaussianNoise:
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
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            factor = random.uniform(0.7, 1.3)
            for i in range(len(sample)):
                sample[i] = sample[i] * factor
        return sample

class RandomContrast:
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
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            min_val, max_val = np.min(sample), np.max(sample)
            for i in range(len(sample)):
                sample[i] = (sample[i] - min_val) / (max_val - min_val)
                g = random.uniform(0.7, 1.5)
                if random.random() < self.prob:
                    sample[i] = 1 - ((1 - sample[i]) ** g)
                else:
                    sample[i] = sample[i] ** g
                sample[i] = (sample[i] * (max_val - min_val)) + min_val
        return sample

class RandomMirror:
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
    def __init__(self, bids_dir, transform=None, patch_size=None):
        self.bids_dir = bids_dir
        self.patch_size = patch_size
        self.transform = transform
        self.subjects = self.get_subjects()

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

        # Normalize images
        FLAIR_img = (FLAIR_img - FLAIR_img.min()) / (FLAIR_img.max() - FLAIR_img.min())
        adc_img = (adc_img - adc_img.min()) / (adc_img.max() - adc_img.min())
        dwi_img = (dwi_img - dwi_img.min()) / (dwi_img.max() - dwi_img.min())

        # Apply transform if provided
        if self.transform:
            data = [FLAIR_img, adc_img, dwi_img, mask_img]
            data = self.transform(data)
            FLAIR_img, adc_img, dwi_img, mask_img = data
        
        # Extract patches
        FLAIR_patches, adc_patches, dwi_patches, mask_patches, coord_patch_list = self.extract_patches(FLAIR_img, adc_img, dwi_img, mask_img)
        # Return the patches as torch tensors
        return (
            torch.tensor(FLAIR_patches, dtype=torch.float32),
            torch.tensor(adc_patches, dtype=torch.float32),
            torch.tensor(dwi_patches, dtype=torch.float32),
            torch.tensor(mask_patches.squeeze(), dtype=torch.float32),
            np.array(coord_patch_list)
        )

    def extract_patches(self, FLAIR, adc, dwi, mask):
        pad_sizes = [(int((max_size - img_size) / 2), int((max_size - img_size) / 2)) for max_size, img_size in zip((256, 256, 256), FLAIR.shape)]
        FLAIR_padded = np.pad(FLAIR, pad_sizes, mode='constant')
        adc_padded = np.pad(adc, pad_sizes, mode='constant')
        dwi_padded = np.pad(dwi, pad_sizes, mode='constant')
        mask_padded = np.pad(mask, pad_sizes, mode='constant')

        FLAIR_patches = pt.patchify(FLAIR_padded, (128, 128, 128), step=64)
        adc_patches = pt.patchify(adc_padded, (128, 128, 128), step=64)
        dwi_patches = pt.patchify(dwi_padded, (128, 128, 128), step=64)
        mask_patches = pt.patchify(mask_padded, (128, 128, 128), step=64)

        FlAIR_patch_list = []
        adc_patch_list = []
        dwi_patch_list = []
        mask_patch_list = []
        coords_list = []

        for i in range(FLAIR_patches.shape[0]):
            for j in range(FLAIR_patches.shape[1]):
                for k in range(FLAIR_patches.shape[2]):
                    FlAIR_patch_list.append(FLAIR_patches[i, j, k, :, :, :])
                    adc_patch_list.append(adc_patches[i, j, k, :, :, :])
                    dwi_patch_list.append(dwi_patches[i, j, k, :, :, :])
                    mask_patch_list.append(mask_patches[i, j, k, :, :, :])
                    coords_list.append((i * 64, j * 64, k * 64))  # Step size is 64
        return np.array(FlAIR_patch_list), np.array(adc_patch_list), np.array(dwi_patch_list), np.array(mask_patch_list), np.array(coords_list)


def reconstruct_segmented_image(predictions, image_shape, patch_size, coords, original_shape):
    d, h, w = image_shape
    pd, ph, pw = patch_size
    segmented_image = np.zeros(image_shape)
    count_map = np.zeros(image_shape)
    
    for (patch, (z, y, x)) in zip(predictions, coords):
        segmented_image[z:z+pd, y:y+ph, x:x+pw] += patch
        count_map[z:z+pd, y:y+ph, x:x+pw] += 1
    
    # Adjust for overlap
    segmented_image /= count_map
    print(segmented_image.shape)
    d2, h2, w2 = int((d-original_shape[0])/2), int((h-original_shape[1])/2), int((w-original_shape[2])/2)
    return segmented_image[d2:d-d2, h2:h-h2, w2:w-w2]


transform = transforms.Compose([
    #RandomRotateScale(patch_size=PATCH_SIZE),
    #RandomGaussianNoise(),
    #RandomGaussianBlur(),
    #RandomBrightness(),
    #RandomContrast(),
    #RandomLowResolution(),
    #RandomGamma(), 
    #RandomMirror(), 
])

# Initialize Dataset and DataLoader
bids_dir = "/home/user/Documents/raph/preprocessed_datasets/ISLES2022"
dataset = BidsDataset(bids_dir, transform=transform, patch_size=PATCH_SIZE)

train_size = 10 #int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Mixed precision training scaler
scaler = torch.cuda.amp.GradScaler()

def plot_and_save_image(image_tensor):
    # Convert the PyTorch tensor to a numpy array
    image_array = image_tensor.cpu().detach().numpy().astype(np.float32)
    # Create an ANTs image from the numpy array
    ants.from_numpy(image_array).plot()



# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    optimizer.zero_grad()  
    for batch_idx, (FLAIR, adc, dwi, target, coord_patch_list) in enumerate(train_loader):
        targets = target.to(DEVICE)
        print('batch number', batch_idx)
        for j in range(len(FLAIR[0])):
            concatenated_data = torch.stack((FLAIR[:, j], adc[:, j], dwi[:, j]), dim=1).to(DEVICE)
            concatenated_data = concatenated_data.type(torch.float32)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                scores = model(concatenated_data)

            # Flatten predictions and targets
            predictions_flat = scores.view(scores.size(0), -1, scores.size(2), scores.size(3))
            #print('predictions_flat', predictions_flat.shape)
            targets_flat = targets[:, j].view(targets.size(0), -1, targets.size(2), targets.size(3))
            #print('targets_flat', targets_flat.shape)
            #plot_and_save_image(predictions_flat[0])
            #plot_and_save_image(targets_flat[0])

            # Threshold predictions
            binary_predictions = (predictions_flat > threshold).float()

            # Compute Dice Loss
            intersection = torch.sum(binary_predictions * targets_flat, dim=1)
            union = torch.sum(binary_predictions, dim=1) + torch.sum(targets_flat, dim=1)
            dice = (2.0 * intersection + 1.0) / (union + 1.0)

            # Compute the loss
            loss = criterion(predictions_flat.float(), targets_flat.float())

            # Accumulate loss and Dice score for the batch
            epoch_loss += loss.item()
            epoch_dice += dice.mean().item()

            # Compute global loss
            global_loss = loss + (1-dice.mean())

            # Backward pass with mixed precision
            scaler.scale(global_loss).backward()
                
            del concatenated_data, scores, predictions_flat, targets_flat
            torch.cuda.empty_cache()

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Dice: {dice.mean().item():.4f}")

    # Log average epoch loss and dice score
    avg_epoch_loss = epoch_loss / (len(train_loader.dataset) * len(FLAIR[0]))
    avg_epoch_dice = epoch_dice / (len(train_loader.dataset) * len(FLAIR[0]))  
    writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
    writer.add_scalar('Dice/train', avg_epoch_dice, epoch)  

    torch.save(model.state_dict(), f"unet_epoch_{epoch + 1}.pth")

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Loss: {avg_epoch_loss:.4f}, Average Dice: {avg_epoch_dice:.4f}")

# Close the TensorBoard writer
writer.close()

print("Training complete!")


# Evaluate the model
model.eval()
val_loss = 0
val_dice = 0  
with torch.no_grad():
    for FLAIR, adc, dwi, targets in val_loader:
        data = torch.cat((FLAIR, adc, dwi), dim=1).to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast():
            scores = model(data)
            scores=(scores > threshold).int()
            dice = dice_score(scores, targets)  

        val_loss += loss.item()
        val_dice += dice.item() 

avg_val_dice = val_dice / len(val_loader)  
print(f"Validation DICE after epoch {epoch + 1}: {avg_val_dice:.4f}")  

# Log validation metrics to TensorBoard
writer.add_scalar('Dice/val', avg_val_dice, epoch)  

# Close the TensorBoard writer
writer.close()

print("Training complete!")