import os
import torch
import monai
from monai.networks.layers import Norm
import torchio as tio
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from monai.networks.nets import DynUNet
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import PolynomialLR
import random

# Set all random seed for reproducibility
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# NETWORK ARCHITECTURE
#--------------------------------------------------------------------------------------
# Define network parameters
spatial_dims = 3
in_channels = 3
out_channels = 1
kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
up_sample_kernel_size = strides[1:]
filters = [32, 64, 128, 256, 320]
# default params
# - norm_name: instance
# - act_name: leaky relu (negative_slope 0.01)
# - dropout: None
    
# Initialize the DynUNet
model = DynUNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    strides=strides,
    upsample_kernel_size=up_sample_kernel_size,
    filters=filters,
    #dropout = dropout
)

# Print model summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#summary(model, (in_channels, 128, 128, 128))

#--------------------------------------------------------------------------------------


# SETTINGS
#--------------------------------------------------------------------------------------

model_savepath = Path("saved_models")
model_savepath_file = model_savepath / "best_model_UNet_StrokeLesion_Half_data_aug.pth"

epochs = 100
val_interval = 1 #10 # at which every number of epochs validation should be computed

criterion_train = monai.losses.DiceCELoss(sigmoid=True)
criterion_val = monai.losses.DiceCELoss(sigmoid=False) # no sigmoid in validation as the sigmoid is already done before patch aggregation
dice_metric = monai.metrics.DiceMetric() # dice score for inference

lr_init = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0.99, nesterov=True)
scheduler = PolynomialLR(optimizer, total_iters=epochs, power=0.9)
#optimizer = torch.optim.Adam(model.parameters(), 1e-3)

#--------------------------------------------------------------------------------------




# DATALOADER
#--------------------------------------------------------------------------------------

# Define the paths to your BIDS data (PREPROCESSED DATA!!!)
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

# Define data augmentations 
value1=random.uniform(0.5, 1.5)
value2=random.uniform(0.5, 1.5)
std_values=sorted([value1, value2])

def RandomBlur(x):
    prob_modality = 0.5
    do_blur = random.random() < prob_modality
    if do_blur:
        std = random.uniform(std_values[0], std_values[1])
        x = tio.RandomBlur(std=std)(x)
    return x

def RandomBrightness(x):
    factor = random.uniform(0.7, 1.3)
    x = x * factor
    return x

def RandomContrast(x):
    factor = random.uniform(0.65, 1.5)
    original_min = x.data.min().item()  # Get the original minimum intensity
    original_max = x.data.max().item()  # Get the original maximum intensity
    x.data *= factor
    x.data = x.data.clip(original_min, original_max)  # Clip the voxel intensities to the original range
    return x

def RandomLowResolution(x):
    modality_prob = 0.5
    do_low_res = random.random() < modality_prob
    factor = random.uniform(1, 2)
    if do_low_res:
        x = tio.Resample((factor, factor, factor),
                         image_interpolation='nearest')(x)
        x = tio.CropOrPad(x.shape, padding_mode='constant')(x)
    return x

def RandomGamma(x):
    prob_prior_transform = 0.15
    do_prior_transform = random.random() < prob_prior_transform
    transform= tio.RandomGamma(log_gamma=(0.7, 1.5))

    mask = x.data != 0

    #Normalise image to [0, 1]
    x.data = (x.data - x.data.min()) / (x.data.max() - x.data.min() + 1e-6)

    if do_prior_transform:
        x = 1 - transform(1 - x)
    
    x = transform(x)

    #Scale back to original range
    x.data = x.data * (x.data.max() - x.data.min()) + x.data.min()
    x.data = x.data * mask
    return x

transforms = tio.Compose([
    tio.OneOf({
        tio.RandomAffine(
            scales=(0.7, 1.4),
            degrees=30,  # This will be interpreted as (-30, 30) for each axis
            isotropic=True,
            default_pad_value=0,
            p=0.08
        ),
        tio.RandomAffine(
            scales=(0.7, 1.4),  # only scaling from U(0.7,1.4)
            isotropic=True,
            default_pad_value=0,
            p=0.16
        ),
        tio.RandomAffine(
            degrees=30,  # This will be interpreted as (-30, 30) for each axis
            isotropic=True,
            default_pad_value=0,
            p=0.16
        ),
    }, p=1.0),
    tio.RandomNoise(     # Add Gaussian noise with random parameters
                mean=0, 
                std=(0, 0.1), 
                p=0.15
            ),
    tio.Lambda(RandomBlur,
                p=0.2,
                ),
    tio.Lambda(lambda x: RandomBrightness, 
               types_to_apply=[tio.INTENSITY],
               p=0.15),
    tio.Lambda(lambda x: RandomContrast, 
               types_to_apply=[tio.INTENSITY],
               p=0.15),
    tio.Lambda(lambda x: RandomLowResolution, 
               p=0.2),
    tio.Lambda(lambda x: RandomGamma,
                types_to_apply=[tio.INTENSITY],
                p=0.15),
    tio.RandomFlip(
                axes=(0, 1, 2), 
                p=0.5
            ),
])

# create the SubjectsDataset
dataset = tio.SubjectsDataset(subjects, transform=None)

# split data into training and validation set
test_size = 50
train_percent = 0.8
train_size = int(train_percent * (len(dataset) - 50)) #160
val_size = len(dataset) - train_size - test_size #40

train_indices, val_indices = random_split(range(len(dataset) - 50), [train_size, val_size])

train_subjects = [subjects[i] for i in train_indices]
val_subjects = [subjects[i] for i in val_indices]

train_dataset = tio.SubjectsDataset(train_subjects, transform=transforms)
val_dataset = tio.SubjectsDataset(val_subjects, transform=None)

# PATCHED TRAINING SET

batch_size_train = 2
patch_size_train = 128
samples_per_volume = 10
max_queue_length = 50

# define sampler to perform foreground oversampling
sampler = tio.data.LabelSampler(patch_size = patch_size_train,
    label_name = 'label',
    label_probabilities = {0: 0.67, 1: 0.33}) # 33% oversamppling of foreground

num_workers = 2 
patches_training_set = tio.Queue(              
    subjects_dataset = train_dataset,
    max_length = max_queue_length,
    samples_per_volume = samples_per_volume,
    sampler = sampler,
    num_workers = num_workers,
    shuffle_subjects = True,
    shuffle_patches = True,
)

training_loader_patches = DataLoader(patches_training_set, batch_size=batch_size_train, shuffle=True)

# TRAINING LOOP
#--------------------------------------------------------------------------------------

# batch size used in validation
batch_size_val = 2

# patch size and overlap for gridSampler used in validation
patch_size_val = 128
patch_overlap_val = 64

# init
best_metric = -1
best_metric_epoch = -1
metric_values = []

writer = SummaryWriter('runs/UNet3D_new_script')

# loop over epochs
for epoch in range(epochs):
    # print("-" * 10)
    # print(f"epoch {epoch + 1}/{epochs}")
    
    # epoch_loss_train = []

    # model.train()
    
    # # loop over batches
    # for batch_idx, batch in enumerate(tqdm(training_loader_patches)):

    #     inputs = torch.cat([batch['flair'][tio.DATA], batch['dwi'][tio.DATA], batch['adc'][tio.DATA]], dim=1).to(device)
    #     labels = batch['label'][tio.DATA].to(device)
    
    #     optimizer.zero_grad()
    #     outputs = model(inputs)
    #     loss = criterion_train(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
        
    #     epoch_loss_train.append(loss.item())

    # average_epoch_loss = sum(epoch_loss_train) / len(epoch_loss_train)    

    # print(f"epoch {epoch + 1} average loss: {average_epoch_loss:.4f}")

    # metrics = {
    #         "Train_Loss": average_epoch_loss,
    #     }
    # writer.add_scalars("New_script/Half_data_aug", metrics, epoch + 1)

    # # update learning rate
    # scheduler.step()
    # print(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {scheduler.get_last_lr()[0]:.5f}")  
    
# VALIDATION
    if (epoch + 1) % val_interval == 0:

        epoch_loss_val = []
        
        model.eval()
        
        # loop over all subject in validation set
        for subject in tqdm(val_dataset):
            # define GridSampler for current subject
            grid_sampler = tio.inference.GridSampler(
                subject = subject,  
                patch_size = patch_size_val,
                patch_overlap = patch_overlap_val,
                padding_mode = 'constant'
            )
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size = batch_size_val)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode = 'hann')
            
            # loop over patches to get model predictions
            for patches_batch in patch_loader:
                patch_inputs = torch.cat([patches_batch['flair'][tio.DATA], patches_batch['dwi'][tio.DATA], patches_batch['adc'][tio.DATA]], dim=1).to(device)
                patch_locations = patches_batch[tio.LOCATION]
                with torch.no_grad():
                    patch_prediction = model(patch_inputs)
                patch_prediction_logits = torch.sigmoid(patch_prediction)
                aggregator.add_batch(patch_prediction_logits, patch_locations)

            # aggregate over patches
            aggregated_logits = aggregator.get_output_tensor().to(device)
            
            # compute validation dice and loss
            ground_truth_seg = subject['label'][tio.DATA].to(device)
            dice_metric((aggregated_logits > 0.5).float(), ground_truth_seg) # keeps track of all dices until called .reset()
            loss_val = criterion_val(aggregated_logits, ground_truth_seg) # no sigmoid needed here
            epoch_loss_val.append(loss_val.item())         

        # get validation loss over epoch
        average_epoch_val_loss = sum(epoch_loss_val) / len(epoch_loss_val)     

        # get validation dice score over epoch        
        average_epoch_dice_val = dice_metric.aggregate().item() # Aggregate the Dice metric for the entire validation set

        # Write the metrics to tensorboard
        metrics2 = {
            "Val_Loss": average_epoch_val_loss,
            "Val_Dice": average_epoch_dice_val
        }
        writer.add_scalars("New_script/Half_data_aug", metrics2, epoch + 1)
      
        # check if best model so far
        metric_values.append(average_epoch_dice_val)
        if average_epoch_dice_val > best_metric:
            best_metric = average_epoch_dice_val
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), model_savepath_file)
            print("saved new best metric model")
        print(
            "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                epoch + 1, average_epoch_dice_val, best_metric, best_metric_epoch
            )
        )
        dice_metric.reset() # reset for next validation round

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
            
