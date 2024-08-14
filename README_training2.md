# UNet3D Training for Stroke Lesion Segmentation

This project trains a 3D U-Net model for stroke lesion segmentation using the MONAI and TorchIO libraries. The data used is the ISLES2022 dataset, preprocessed into the BIDS format using the preprocessing.py file.

Best result: Combination of parameter number 9 (see in [Parameters Tried]) which reached a mean dice of 0.7420, with a median of 0.79 and a standard deviation of 0.15.
 
## Table of Contents

- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Validation](#validation)
- [Model Saving & Results](#model-saving-results)
- [Parameters Tried](#params-tried)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.x
- PyTorch 1.7+
- TorchIO 0.18+
- TorchSummary
- TensorBoard
- TQDM

Install the required packages using:

```bash
pip3 install torch torchio torchsummary tensorboard tqdm
```

## Data Preparation

The data should be in BIDS format and preprocessed. The directory structure should be as follows:

bids_dir/
│
├── sub-01/
│   ├── ses-0001/
│   │   ├── anat/
│   │   │   └── sub-01_ses-0001_FLAIR.nii.gz
│   │   ├── dwi/
│   │   │   └── sub-01_ses-0001_dwi.nii.gz
│   │   │   └── sub-01_ses-0001_ADC.nii.gz
│   │   └── ...
│   └── ...
└── derivatives/
    ├── sub-01/
    │   ├── ses-0001/
    │   │   └── sub-01_ses-0001_msk.nii.gz
    └── ...

## Training
The model is trained using a DynUNet with 3D convolutions from MONAI. It consists in 33,090,657 trainable parameters for an estimated total size 22639.73 MB.
Most of the training parameters have been chosen according to nnU-Net's supplementary material in an attempt to obtain similar results to them as they currently
are one of the best (if not the best) segmentation algorithm as it outperforms most of the existing algorithms on a wide range of segmentation tasks.

The training is done on patches that are sampled from the original images randomly using tio.data.LabelSampler which samples randomly, but with centre voxels as labels according to the prespecified probabilities for each label. To do exactly as nnU-Net, another option to LabelSampler would be to extract patches accross a whole volume, this can be done by using the GridSampler. We already tried 
to extract patches accross a whole volume manually in the training.py file but couldn't manage to reach good results. They are then shuffled into a tio.Queue that is loaded in the training loop for each epoch. 

However, there is an issue with TorchIO's LabelSampler which is that patches cannot be extracted from labels that are too close from the side of the image and which will result in a patch being partially out
of the image due to the patch size. To counter this we adapted the LabelSampler and created the LabelSampler_Pad4LabelPatches class directly into TorchIO's library. The steps to adapt the TorchIO library can be found in the Parameters tried part of this READ_ME file.

**nnU-Net's Training Parameters**
- Epochs: 100
- Training size: 160
- Initial Learning Rate: 0.01, using the same scheduler as nnU-Net which follows the next formula: next_lr = base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
	with power = 0.9
- Batch Size: 2
- Patch Size: 128x128x128

Parameters added by the use of the TorchIO library:
- Samples per Volume: 20 (For Queue)
- Foreground Oversampling: 33% (For LabelSampler)

Changed parameters:
- optimizer: Adam with lr=1e-3 can be used
- batch size = 4 was used for all trainings

**Data Augmentation**
The training dataset is augmented using various data augmentation techniques provided by TorchIO. The following augmentations (same as nnU-Net) are applied to the training data:
- RandomAffine: Scaling and rotation are applied together for improved speed of computation. 
	This approach reduces the amount of required data interpolations to one. Scaling and 
	rotation are applied with a probability of 0.2 each (resulting in probabilities of
	0.16 for only scaling, 0.16 for only rotation and 0.08 for both being triggered). If 
	processing isotropic 3D patches, the angles of rotation (in degrees) alpha_x, alpha_y 
	and alpha_z are each drawn from U (30, 30). If a patch is anisotropic or 2D, the angle 
	of rotation is sampled from Uniform (180, 180). If the 2D patch size is anisotropic, 
	the angle is sampled from Uniform (15, 15). Scaling is implemented via multiplying 
	coordinates with a scaling factor in the voxel grid. Thus, scale factors smaller than
	 one result in a "zoom out" effect while values larger one result in a "zoom in" effect. 
	 The scaling factor is sampled from Uniform (0.7, 1.4) for all patch
	types.
- RandomNoise: Zero centered additive Gaussian noise is added to each voxel in the sample
	independently. This augmentation is applied with a probability of 0.15. The variance of 
	the noise is drawn from Uniform (0, 0.1) (note that the voxel intensities in all samples 
	are close to zero mean and unit variance due to intensity normalization).
- RandomBlur: Blurring is applied with a probability of 0.2 per sample. If this augmentation
	is triggered in a sample, blurring is applied with a probability of 0.5 for each of the
	associated modalities (resulting in a combined probability of only 0.1 for samples with a
	single modality). The width (in voxels) of the Gaussian kernel  is sampled from Uniform (0.5, 1.5)
	independently for each modality.
- RandomBrightness: Voxel intensities are multiplied by x ⇠ Uniform (0.7, 1.3) with a probability of 0.15.
- RandomContrast: Voxel intensities are multiplied by x ⇠ Uniform (0.65, 1.5) with a probability of 0.15.
	Following multiplication, the values are clipped to their original value range.
- RandomLowResolution: This augmentation is applied with a probability of 0.25 per sample and 0.5 per 
	associated modality. Triggered modalities are downsampled by a factor of x ⇠ Uniform (1, 2) using 
	nearest neighbor interpolation and then sampled back up to their original size with cubic interpolation. 
	For 2D patches or anisotropic 3D patches, this augmentation is applied only in 2D leaving the out 
	of plane axis (if applicable) in its original state.
- RandomGamma: This augmentation is applied with a probability of 0.15. The
	patch intensities are scaled to a factor of [0, 1] of their respective value range. Then, a
	nonlinear intensity transformation is applied per voxel: i_new = i_old**g with g ⇠ U (0.7, 1.5).
	The voxel intensities are subsequently scaled back to their original value range. With a
	probability of 0.15, this augmentation is applied with the voxel intensities being inverted
	prior to transformation: (1 - i_new) = (1 - i_old)**g.
- RandomFlip: All patches are mirrored with a probability of 0.5 along all axes.

**nnU-Net's Optimizer**
The optimizer used is the Stochastic Gradient Descent with a Nesterov momentum of 0.99, as nnU-Net does. SGD might be slower than Adam but generalizes better.

**Loss**
The loss used in this code is the combination of Dice Loss and BCE Loss. This loss allows
to combine Dice loss and Binary Cross entropy loss into one metric.

## Validation
Validation is performed every 5 epochs using a combination of Dice Loss and BCE Loss. The validation set (of size 40) is a subset of the dataset not used in training.
The patches are created using GridSampler, presented before that allows to extract patches across a whole volume. They are then aggregated using the GridAggregator
in order to rebuild the volume from patches. We deal with the overlapping areas by using Hann window function which allows to weight the overlapping predictions.
For validation the Dice score is computed on the binarized predictions.

**Validation Parameters**
- Batch Size: 2
- Validation Size: 40
- Patch Size: 128x128x128
- Patch Overlap: 64

Changed parameters:
- Batch size = 4 was used for all validations

## Model Saving & Results
The model with the best Dice score on the validation set is saved to ```saved_models/*.pth```

The results of the training and validation are displayed in TensorBoard. Those results are saved in ```runs/UNet3D_new_script```

## Parameters tried 
Here is the detailed list of parameters we used in each run with the tensorboard name that corresponds and the corresponding saved model and results.
Note: A visualisation of the results on validation set with the best model is possible using the training_plotting.py file. You will also get median dice and its standard deviation.
      The file should be ready to use. You just need to change the name of the model that you want to use!

The parameters in between * * are the parameters that change from an attempt to the next one.

**1st attempt:**
model name: best_model_UNet_StrokeLesion_Full_data_aug_new_loss.pth
tensorboard graph name: Data_aug_new_loss

Parameters:
- Use of adam with lr = 1e-3 as optimizer instead of SGD to have a first look
- 15 images per patch
- TorchIO's label sampler
- Data augmentation (still findable in the code as transforms, we were not sure about it because we had some troubles with nnU-Net's data aug in the training.py file so we made a new one 
using only 3 of nnU-Net's original transforms (because they were already implemented in the TorchIO's library as we wanted them to work) some more transforms were also added but the parameters 
have not been rigorously chosen and were in fact just adjusted to have a not to brutal data aug. So in total there are 3 data augmentation functions such as nnU-Net, 3 more alike but with different probabilities and 4 new.

Results (on validation set with best model):
Dice mean = 0.6814, median = 0.77, standard deviation = 0.10


**2nd attempt:**
model name: best_model_UNet_StrokeLesion_Full_data_aug_new_loss_20_patches.pth
tensorboard graph name: Data_aug_new_loss_20_patches

Parameters:
- Use of adam with lr = 1e-3
- *20 images per patch*
- TorchIO's label sampler
- Same data augmentation

Results (on validation set with best model):
Dice mean = 0.6787, median = 0.77, standard deviation = 0.23

**3rd attempt:**
model name: best_model_UNet_StrokeLesion_Full_data_aug_new_loss_20_patches_uniform_sampler.pth
tensorboard graph name: Data_aug_new_loss_20_patches_uniform_sampler

Parameters:
- Use of adam with lr = 1e-3
- 20 images per patch
- *TorchIO's uniform sampler*
- Same data augmentation

Results (on validation set with best model):
Dice mean = 0.7182, median = 0.76, standard deviation = 0.21


**4th attempt:**
model name: best_model_UNet_StrokeLesion_nnUNet_data_aug_new_loss_20_patches_uniform_sampler.pth
tensorboard graph name: nnUNet_aug_new_loss_20_patches_uniform_sampler

Parameters:
- Use of adam with lr = 1e-3
- 20 images per patch
- *TorchIO's label sampler* (Yes, I made a mistake in the model and tb's names but it really is the label sampler that has been used there)
- *nnU-Net's data augmentation* (transforms_nnUNet variable in the code)

Results (on validation set with best model):
Dice mean = 0.7104, median = 0.80, standard deviation = 0.14

**5th attempt:**
model name: best_model_UNet_StrokeLesion_nnUNet_aug_new_loss_20_patches_label_sampler_SGD.pth
tensorboard graph name: nnUNet_aug_new_loss_20_patches_label_sampler_SGD

Parameters:
- *Use of SGD with lr scheduler*
- 20 images per patch
- TorchIO's label sampler
- nnU-Net's data augmentation

Results (on validation set with best model):
Dice mean = 0.6494, median = 0.74, standard deviation = 0.24


**6th attempt:**
model name: best_model_UNet_StrokeLesion_nnUNet_aug_new_loss_20_patches_label_sampler_adam_dropout.pth
tensorboard graph name: nnUNet_aug_new_loss_20_patches_label_sampler_adam_dropout

Parameters:
- Use of adam with lr = 1e-3
- 20 images per patch
- TorchIO's label sampler
- nnU-Net's data augmentation
- *20 % Dropout*

Results (on validation set with best model):
Dice mean = 0.66, median = 0.79, standard deviation = 0.17

Note: This dropout is way to aggressive. You can see that the loss barely decreases. It's because when looking at DynUnet in more details you will see that they in fact have
      23 layers of dropout coded in their model. --> To do again with less dropout layers.


**7th attempt:**
model name: best_model_UNet_StrokeLesion_nnUNet_aug_new_loss_20_patches_label_sampler_SGD_300_epochs.pth
tensorboard graph name: nnUNet_aug_new_loss_20_patches_label_sampler_SGD_300_epochs

Parameters:
- Use of adam with lr = 1e-3
- 20 images per patch
- TorchIO's label sampler
- nnU-Net's data augmentation
- No Dropout
- 300 epochs

Results (on validation set with best model):
Dice mean = 0.692, median = 0.76, standard deviation = 0.19

Note: This training was interrupted after 201 epoch because the computer crashed but we can se that the model is still training. --> To do again.


**8th attempt:**
model name: best_model_UNet_StrokeLesion_nnUNet_aug_new_loss_20_patches_label_sampler_SGD_300_epochs.pth
tensorboard graph name: nnUNet_aug_new_loss_20_patches_label_sampler_SGD_300_epochs

Parameters:
- Use of adam with lr = 1e-3
- 20 images per patch
- new label sampler that can make patches on the side of the images and pad them if needed! To be able to use it in another computer, you need to import 3 files
and modify 2:
	- the 3 files to import are label_Pad4LabelPatches.py, sampler_Pad4LabelPatches.py and weighted_Pad4LabelPatches.py which are located in this computer in this 
	  folder /home/user/.local/lib/python3.10/site-packages/torchio/data/sampler. You need to put the 3 files in your torchio/data/sampler folder on your computer.
	  To know where your TorchIO library folder is you can do: pip3 show torchio. 
	- The first file to modify is the __init__.py file which is in torchio/data/sampler. You need to initialize the new classes and functions.
	  On this computer it is located at: /home/user/.local/lib/python3.10/site-packages/torchio/data/sampler.
	- The final file to modify is also an __init__.py file but in the parent folder : torchio/data. 
	  Located at: /home/user/.local/lib/python3.10/site-packages/torchio/data.
- nnU-Net's data augmentation
- No Dropout
- 100 epochs

Results (on validation set with best model):
Dice mean = 0.7420, median = 0.79, standard deviation = 0.16

Note: This training was interrupted after 201 epoch because the computer crashed but we can se that the model is still training. --> To do again.


**9th attempt:**
model name: best_model_UNet_StrokeLesion_nnUNet_aug_new_loss_20_patches_new_label_sampler_adam_dropout_200_epochs.pth
tensorboard graph name: nnUNet_aug_new_loss_20_patches_new_label_sampler_adam_dropout_200_epochs

Parameters:
- Use of adam with lr = 1e-3
- 20 images per patch
- new label sampler 
- nnU-Net's data augmentation
- 20 % Dropout applied only once at each downsampling/upsampling step but the last upsampling step. Also applied once in the bottleneck.
- 200 epochs

Results (on validation set with best model):
Dice mean = 0.7434, median = 0.79, standard deviation = 0.15


**10th attempt:**
model name: best_model_UNet_StrokeLesion_nnUNet_aug_new_loss_20_patches_label_sampler_SGD_500_epochs.pth
tensorboard graph name: nnUNet_aug_new_loss_20_patches_label_sampler_SGD_500_epochs

Parameters:
- Use of SGD with lr scheduler 
- 20 images per patch
- new label sampler 
- nnU-Net's data augmentation
- No Dropout
- 500 epochs

Results (on validation set with best model):
Dice mean = 0.7375, median = 0.79, standard deviation = 0.15

## Usage
To train the model, go to the associated repository in the terminal and then type:
```bash
python3 training2.py
```

## Acknowledgments
This project uses the following libraries:
- **TorchIO**
- **TensorBoard**
- **PyTorch**

Special thanks to the authors and contributors of these libraries for providing excellent tools for medical image analysis and deep learning.

