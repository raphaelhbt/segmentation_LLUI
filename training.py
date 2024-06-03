import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import ants
import numpy as np
import UNet_model as unet

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Set parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet.UNet3D(in_channels=3, out_channels=4).to(DEVICE)  # Adjust in_channels for 3 input modalities

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
PATCH_SIZE = [128, 128, 128]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# BIDS Dataset Loader
class BidsDataset(Dataset):
    def __init__(self, bids_dir, transform=None, patch_size=None):
        self.bids_dir = bids_dir
        self.transform = transform
        self.patch_size = patch_size
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

        # Load images
        FLAIR_img = ants.image_read(FLAIR_path).numpy()
        adc_img = ants.image_read(adc_path).numpy()
        dwi_img = ants.image_read(dwi_path).numpy()
        mask_img = ants.image_read(mask_path).numpy()

        # Extract patches
        FLAIR_patches, adc_patches, dwi_patches, mask_patches = self.extract_patches(FLAIR_img, adc_img, dwi_img, mask_img)

        # Apply transform if provided
        if self.transform:
            FLAIR_patches = self.transform(FLAIR_patches)
            adc_patches = self.transform(adc_patches)
            dwi_patches = self.transform(dwi_patches)
            mask_patches = self.transform(mask_patches)

        return torch.tensor(FLAIR_patches, dtype=torch.float32), torch.tensor(adc_patches, dtype=torch.float32), torch.tensor(dwi_patches, dtype=torch.float32), torch.tensor(mask_patches.squeeze(), dtype=torch.long)

    def extract_patches(self, FLAIR, adc, dwi, mask):
        FLAIR_patches = []
        adc_patches = []
        dwi_patches = []
        mask_patches = []

        for i in range(0, FLAIR.shape[0], self.patch_size[0]):
            for j in range(0, FLAIR.shape[1], self.patch_size[1]):
                for k in range(0, FLAIR.shape[2], self.patch_size[2]):
                    FLAIR_patch = FLAIR[i:i+self.patch_size[0], j:j+self.patch_size[1], k:k+self.patch_size[2]]
                    adc_patch = adc[i:i+self.patch_size[0], j:j+self.patch_size[1], k:k+self.patch_size[2]]
                    dwi_patch = dwi[i:i+self.patch_size[0], j:j+self.patch_size[1], k:k+self.patch_size[2]]
                    mask_patch = mask[i:i+self.patch_size[0], j:j+self.patch_size[1], k:k+self.patch_size[2]]

                    if FLAIR_patch.shape == tuple(self.patch_size) and adc_patch.shape == tuple(self.patch_size) and dwi_patch.shape == tuple(self.patch_size) and mask_patch.shape == tuple(self.patch_size):
                        FLAIR_patches.append(FLAIR_patch)
                        adc_patches.append(adc_patch)
                        dwi_patches.append(dwi_patch)
                        mask_patches.append(mask_patch)

        return np.array(FLAIR_patches), np.array(adc_patches), np.array(dwi_patches), np.array(mask_patches)

# Initialize Dataset and DataLoader
bids_dir = "/home/user/Documents/raph/preprocessed_datasets/ISLES2022"
dataset = BidsDataset(bids_dir, patch_size=PATCH_SIZE)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (FLAIR, adc, dwi, targets) in enumerate(train_loader):
        # Concatenate modalities along the channel dimension
        data = torch.cat((FLAIR, adc, dwi), dim=1).to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {epoch_loss/len(train_loader):.4f}")

    # Save the model checkpoint
    torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth")

    # Evaluate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for FLAIR, adc, dwi, targets in val_loader:
            data = torch.cat((FLAIR, adc, dwi), dim=1).to(DEVICE)
            targets = targets.to(DEVICE)
            scores = model(data)
            loss = criterion(scores, targets)
            val_loss += loss.item()
    
    print(f"Validation Loss after epoch {epoch+1}: {val_loss/len(val_loader):.4f}")

print("Training complete!")
