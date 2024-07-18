# UNet3D Training for Stroke Lesion Segmentation

This project trains a 3D U-Net model for stroke lesion segmentation using the MONAI and TorchIO libraries. The data used is the ISLES2022 dataset, preprocessed into the BIDS format using the preprocessing.py file.

## Table of Contents

- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Validation](#validation)
- [Model Saving & Results](#model-saving-results)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.x
- PyTorch
- MONAI
- TorchIO
- TorchSummary
- TensorBoard

Install the required packages using:

```bash
pip3 install torch monai torchio torchsummary tensorboard
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

The training is done on patches that are sampled from the original images randomly using tio.data.LabelSampler which extracts random patches with labeled voxels at their center.
To do exactly as nnU-Net, another option to LabelSampler would be the GridSampler which extract patches accross a whole volume.
They are then shuffled into a tio.Queue that is loaded in the training loop for each epoch.

**Training Parameters**
- Epochs: 100
- Training size: 160
- Initial Learning Rate: 0.01, using the same scheduler as nnU-Net which follows the next formula: next_lr = base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
	with power = 0.9
- Batch Size: 2
- Patch Size: 128x128x128
- Samples per Volume: 10 (For Queue)
- Foreground Oversampling: 33% (For LabelSampler)

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

**Optimizer**
The optimizer used is the Stochastic Gradient Descent with a Nesterov momentum of 0.99, as nnU-Net does. SGD might be slower than Adam but generalizes better.

**Loss**
The loss used in this code is the DiceCELoss from Monai with a sigmoid as activation function since we do not have any in our implementation of U-Net. The DiceCELoss allows
to combine Dice loss and Cross entropy loss into one metric.

## Validation
Validation is performed every 10 epochs using a DiceCELoss without sigmoid activation. The validation set is a subset of the dataset not used in training.
The patches are created using GridSampler, presented before that allows to extract patches across a whole volume. They are then aggregated using the GridAggregator
in order to rebuild the volume from patches. We deal with the overlapping areas by using Hann window function which allows to weight the overlapping predictions.
For validation the Dice score is computed on the binarized predictions.

**Validation Parameters**
- Batch Size: 2
- Validation Size: 40
- Patch Size: 128x128x128
- Patch Overlap: 64

## Model Saving & Results
The model with the best Dice score on the validation set is saved to ```saved_models/best_model_UNet_StrokeLesion.pth```

The results of the training and validation are displayed in TensorBoard. Those results are saved in ```runs/UNet3D_new_script```

## Usage
To train the model, go to the associated repository in the terminal and then type:
```bash
python3 training2.py
```

## Acknowledgments
This project uses the following libraries:
- **MONAI**
- **TorchIO**
- **TensorBoard**
- **PyTorch**

Special thanks to the authors and contributors of these libraries for providing excellent tools for medical image analysis and deep learning.

