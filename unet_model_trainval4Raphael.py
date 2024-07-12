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
# - dropout: None # change to 0.2 for example (=dropout rate in every layer)
# dropout = 0.2

class DynUNetWithSigmoid(torch.nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, strides, upsample_kernel_size, filters, dropout=None):
        super(DynUNetWithSigmoid, self).__init__()
        # Initialize the DynUNet
        self.dynunet = DynUNet(spatial_dims=spatial_dims, 
                               in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size, 
                               strides=strides, 
                               upsample_kernel_size=upsample_kernel_size,
                               filters=filters,
                               dropout = dropout
                            )
        
    def forward(self, x):
        # Pass input through DynUNet
        x = self.dynunet(x)
        # Apply sigmoid activation function
        x = torch.sigmoid(x)
        return x
    
# Initialize the DynUNet
model = DynUNetWithSigmoid(
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
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
#summary(model, (in_channels, 128, 128, 128))

#--------------------------------------------------------------------------------------


# SETTINGS
#--------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_savepath = Path("saved_models")
model_savepath_file = model_savepath / "best_model_UNet_StrokeLesion.pth"

epochs = 100
val_interval = 10 # at which every number of epochs validation should be computed

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
print(sub_folders)

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

# defina data augmentations ## --- PLEASE CHANGE AUGMENTATIONS, THIS IS JUST A DUMMY PLACEHOLDER
transforms = tio.Compose([
    tio.RandomMotion(p=0.2),
    tio.RandomBiasField(p=0.3),
    tio.RandomNoise(p=0.5),
    tio.RandomFlip(),
    tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    }),
])

# create the SubjectsDataset
dataset = tio.SubjectsDataset(subjects, transform=transforms)


# split data into training and validation set
train_percent = 0.8
train_size = int(train_percent * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# add transformations to the training dataset
train_dataset.dataset.transform = transforms
# ---COMMENT: if we do for example also z-scoring and so in augmentation, we also have to do this for the validation set




# PATCHED TRAINING SET

batch_size_train = 2
patch_size_train = 128
samples_per_volume = 25 # adjust if necessary
max_queue_length = 25 #????? dont know how many probably depends on memory

# define sampler to perform foreground oversampling
sampler = tio.data.LabelSampler(patch_size = patch_size_train,
    label_name = 'label',
    label_probabilities = {0: 0.67, 1: 0.33}) # 33% oversamppling of foreground ## is this even possible to have patches with 0 in centre when only at border????

num_workers = 2 #os.cpu_count() - 2 # dont know how many, probably best to number of CPU minus 2
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


#*** if we wanted to do the validation metrics computation also do with random patches over image instead of specified grid, we could use the following:
#
# patches_validation_set = tio.Queue(
    # subjects_dataset=val_dataset,
    # max_length=max_queue_length,
    # samples_per_volume=samples_per_volume,
    # sampler=sampler,
    # num_workers=num_workers,
    # shuffle_subjects=False,
    # shuffle_patches=False,
# )

# validation_loader_patches = DataLoader(
    # patches_validation_set, batch_size=validation_batch_size)


#--------------------------------------------------------------------------------------



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

writer = SummaryWriter()

# loop over epochs
for epoch in range(epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epochs}")
    
    epoch_loss_train = []

    model.train()
    
    # loop over batches
    for batch_idx, batch in enumerate(tqdm(training_loader_patches)):

        inputs = torch.cat([batch['flair'][tio.DATA], batch['dwi'][tio.DATA], batch['adc'][tio.DATA]], dim=1).to(device)
        labels = batch['label'][tio.DATA].to(device)
        
        #*** CHATGPT proposed the follwoing, but I think we do not need it, but please check!
        # Assuming labels are in shape (batch_size, 1, D, H, W), need to flatten
        # labels = torch.squeeze(labels).float()  # Assuming labels are originally (batch_size, 1, D, H, W)
    
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion_train(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss_train.append(loss.item())

    average_epoch_loss = sum(epoch_loss_train) / len(epoch_loss_train)    
    writer.add_scalar("train_loss", average_epoch_loss, epoch + 1)
    print(f"epoch {epoch + 1} average loss: {average_epoch_loss:.4f}")


# VALIDATION
    if (epoch + 1) % val_interval == 0:

        epoch_loss_val = []
        
        model.eval()
        
        # loop over all subject in validation set
        for subject in tqdm(val_dataset):
            # define GridSampler for current subject
            grid_sampler = tio.inference.GridSampler(
                subject = subject,  ### in torch.tio tutorial: subject = random.choice(validation_set) = meaning only one subject at a time????? (dont know, but for now I did a loop over all subjects)
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
                patch_prediction_logits = torch.sigmoid(patch_prediction) # only needed when sigmoid is not in Unet as last layer (as it is the default in MONAI UNet)
                aggregator.add_batch(patch_prediction_logits, patch_locations)

            # aggregate over patches
            aggregated_logits = aggregator.get_output_tensor()
            
            # compute validation dice and loss
            ground_truth_seg = subject['label'][tio.DATA].to(device)
            dice_metric((aggregated_logits > 0.5).float(), ground_truth_seg) # keeps track of all dices until called .reset()
            loss_val = criterion_val(aggregated_logits, ground_truth_seg) # no sigmoid needed here
            epoch_loss_val.append(loss_val.item())         

        # get validation loss over epoch
        average_epoch_loss = sum(epoch_loss_val) / len(epoch_loss_train)     
        writer.add_scalar("val_loss", average_epoch_loss, epoch + 1)

        # get validation dice score over epoch        
        average_epoch_dice_val = dice_metric.aggregate().item() # Aggregate the Dice metric for the entire validation set
        writer.add_scalar("val_dice", average_epoch_dice_val, epoch + 1)

        # check if best model so far
        metric_values.append(average_epoch_dice_val)
        if average_epoch_dice_val > best_metric:
            best_metric = average_epoch_dice_val
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), model_savepath_file)
            print("saved new best metric model")
        print(
            "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                epoch + 1, dice_metric, best_metric, best_metric_epoch
            )
        )
        dice_metric.reset() # reset for next validation round

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()




            
# # ONLY NEEDED WHEN TEST SET (NOT VALIDATION)
# #**********************************************************
# # add sigmoid output image (prediction in logits) to subject in torch.tio dataset
# logits_image = tio.ScalarImage(tensor = aggregated_logits, affine = subject.label.affine)
# logits_image.save(output_savepath_sigmoid)
# # add binary segmentation image to subject in torch.tio dataset
# binary_segmentation = (aggregated_logits > 0.5).float()
# segmentation_image = tio.ScalarImage(tensor = binary_segmentation, affine = subject.label.affine)
# logits_image.save(output_savepath_binary)
# #**********************************************************
            
