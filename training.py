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

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)  # Set seed for random module

# Set parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet.UNet3D(in_channels=3, out_channels=2).to(DEVICE)  # Adjust in_channels for 3 input modalities

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
PATCH_SIZE = [128, 128, 128]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize TensorBoard writer
log_dir = 'runs/UNet3D_experiment_1'
writer = SummaryWriter(log_dir)


class RandomRotateScale:
    def __init__(self, patch_size, rotate_prob=0.16, scale_prob=0.16, both_prob=0.08):
        self.patch_size = patch_size
        self.rotate_prob = rotate_prob
        self.scale_prob = scale_prob
        self.both_prob = both_prob

    def __call__(self, sample):
        new_sample = []
        do_rotate = random.random() < self.rotate_prob
        do_scale = random.random() < self.scale_prob
        do_both = random.random() < self.both_prob

        if do_rotate or do_both:
            angle_x = random.uniform(-30, 30)
            angle_y = random.uniform(-30, 30)
            angle_z = random.uniform(-30, 30)

            for i in range(len(sample)):
                rotated_img = rotate(sample[i], angle_x, resize=False, preserve_range=True)
                rotated_img = rotate(rotated_img, angle_y, resize=False, preserve_range=True)
                rotated_img = rotate(rotated_img, angle_z, resize=False, preserve_range=True)
                new_sample.append(rotated_img)

        if do_scale or do_both:
            scale = random.uniform(0.7, 1.4)

            for i in range(len(new_sample)):
                # Zoom and Resize to match patch size
                resized_img = resize(zoom(new_sample[i], scale), self.patch_size, anti_aliasing=True)
                new_sample[i] = resized_img

        # Convert list of arrays to a single numpy array
        new_sample = np.array(new_sample)

        return new_sample if new_sample.size > 0 else sample  # If no transformation is applied, return the original sample
    
class RandomGaussianNoise:
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            variance = random.uniform(0, 0.1)
            noise = np.random.normal(0, variance, sample.shape)
            sample = sample + noise
        return sample

class RandomGaussianBlur:
    def __init__(self, sample_prob = 0.2, modality_prob = 0.5):
        self.sample_prob = sample_prob
        self.modality_prob = modality_prob

    def __call__(self, sample):
        do_blur = random.random() < self.sample_prob

        if do_blur:
            sigma = random.uniform(0.5, 1.5)
            sample = gaussian_filter(sample, sigma)
            for i in range(len(sample)):
                if random.random() < self.modality_prob:
                    sigma = random.uniform(0.5, 1.5)
                    sample[i] = gaussian_filter(sample[i], sigma)

        return sample
class RandomBrightness:
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            factor = random.uniform(0.7, 1.3)
            sample = sample * factor
        return sample

class RandomContrast:
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        new_sample = sample
        if random.random() < self.prob:
            factor = random.uniform(0.65, 1.5)
            new_sample = sample * factor
            new_sample = np.clip(new_sample, sample.min(), sample.max())
        return new_sample


class RandomLowResolution:
    def __init__(self, sample_prob=0.25, modality_prob=0.5):
        self.sample_prob = sample_prob
        self.modality_prob = modality_prob

    def __call__(self, sample):
        if random.random() < self.sample_prob:
            factor = random.uniform(1, 2)
            downsampled = zoom(sample, 1 / factor, order=0)
            upsampled = resize(downsampled, sample.shape, order=3, mode='reflect', anti_aliasing=True)
            sample = upsampled

        return sample

class RandomGamma:
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            # Scale intensities to [0, 1]
            min_val, max_val = np.min(sample), np.max(sample)
            sample = (sample - min_val) / (max_val - min_val)

            # Apply nonlinear intensity transformation
            g = random.uniform(0.7, 1.5)
            if random.random() < self.prob:
                # Invert intensities prior to transformation
                sample = 1 - ((1 - sample) ** g)
            else:
                sample = sample ** g

            # Scale intensities back to original range
            sample = (sample * (max_val - min_val)) + min_val

        return sample

class RandomMirror:
    def __init__(self):
        self.prob = 0.5

    def __call__(self, sample):
        if random.random() < self.prob:
            axes = [0, 1, 2]  # Define the possible axes to flip
            for axis in axes:
                if random.random() < self.prob:  # 50% chance to flip along each axis
                    sample = np.flip(sample, axis=axis).copy()  # Ensure to create a copy after flipping

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

        # Extract patches
        FLAIR_patches, adc_patches, dwi_patches, mask_patches = self.extract_patches(FLAIR_img, adc_img, dwi_img, mask_img)

        # Apply transform if provided
        if self.transform:
            FLAIR_patches = self.transform(FLAIR_patches)
            adc_patches = self.transform(adc_patches)
            dwi_patches = self.transform(dwi_patches)
            mask_patches = self.transform(mask_patches)

        # Return the patches as torch tensors
        return (
            torch.tensor(FLAIR_patches, dtype=torch.float32),
            torch.tensor(adc_patches, dtype=torch.float32),
            torch.tensor(dwi_patches, dtype=torch.float32),
            torch.tensor(mask_patches.squeeze(), dtype=torch.long)
        )

    def extract_patches(self, FLAIR, adc, dwi, mask):
        FLAIR_patches, adc_patches, dwi_patches, mask_patches = [], [], [], []

        for i in range(0, FLAIR.shape[0] - self.patch_size[0] + 1, self.patch_size[0]):
            for j in range(0, FLAIR.shape[1] - self.patch_size[1] + 1, self.patch_size[1]):
                for k in range(0, FLAIR.shape[2] - self.patch_size[2] + 1, self.patch_size[2]):
                    FLAIR_patch = FLAIR[i:i+self.patch_size[0], j:j+self.patch_size[1], k:k+self.patch_size[2]]
                    adc_patch = adc[i:i+self.patch_size[0], j:j+self.patch_size[1], k:k+self.patch_size[2]]
                    dwi_patch = dwi[i:i+self.patch_size[0], j:j+self.patch_size[1], k:k+self.patch_size[2]]
                    mask_patch = mask[i:i+self.patch_size[0], j:j+self.patch_size[1], k:k+self.patch_size[2]]

                    if FLAIR_patch.shape == tuple(self.patch_size) and adc_patch.shape == tuple(self.patch_size) and dwi_patch.shape == tuple(self.patch_size) and mask_patch.shape == tuple(self.patch_size):
                        FLAIR_patches.append(FLAIR_patch)
                        adc_patches.append(adc_patch)
                        dwi_patches.append(dwi_patch)
                        mask_patches.append(mask_patch)
                        #print('i:', i, 'j:', j, 'k:', k)

        return np.array(FLAIR_patches), np.array(adc_patches), np.array(dwi_patches), np.array(mask_patches)

# Initialize Dataset and DataLoader
transform = transforms.Compose([
    RandomRotateScale(patch_size=PATCH_SIZE),
    RandomGaussianNoise(),
    RandomGaussianBlur(),
    RandomBrightness(),
    RandomContrast(),
    RandomLowResolution(),
    RandomGamma(), 
    RandomMirror(), 
])

bids_dir = "/home/user/Documents/raph/preprocessed_datasets/ISLES2022"
dataset = BidsDataset(bids_dir, transform=transform, patch_size=PATCH_SIZE)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Mixed precision training scaler
scaler = torch.cuda.amp.GradScaler()

# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (FLAIR, adc, dwi, targets) in enumerate(train_loader):
        # Concatenate modalities along the channel dimension
        data = torch.cat((FLAIR, adc, dwi), dim=1).to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = criterion(scores, targets)
        
        # Print the shapes to debug
        if batch_idx == 0:  # Only print for the first batch to reduce clutter
            print(f"Batch {batch_idx} - data shape: {data.shape}, scores shape: {scores.shape}, targets shape: {targets.shape}")
        
        epoch_loss += loss.item()

        # Backward pass with mixed precision
        optimizer.zero_grad(set_to_none=True)  # Reduce memory usage by setting gradients to None instead of zero
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


        # Log training loss to TensorBoard
        if epoch > 0:
            avg_epoch_loss = epoch_loss / len(train_loader)

            writer.add_scalar('Loss/train', avg_epoch_loss, epoch)

        # Save the model checkpoint
        torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth")

    # Evaluate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for FLAIR, adc, dwi, targets in val_loader:
            data = torch.cat((FLAIR, adc, dwi), dim=1).to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                scores = model(data)
                loss = criterion(scores, targets)
            
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f}")

    # Log validation loss to TensorBoard
    writer.add_scalar('Loss/val', avg_val_loss, epoch)

# Close the TensorBoard writer
writer.close()

print("Training complete!")

