import numpy as np
import monai
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import ants

a=np.zeros(1,128,128,128)
b=np.zeros(2,128,128,128)
# path1= '/home/user/Documents/raph/preprocessed_datasets/ISLES2022/derivatives/sub-227/ses-0001/sub-227_ses-0001_msk.nii.gz'
# path2= '/home/user/Documents/raph/code/aggregated_logits_sub-sub-227.nii.gz'

# # Load the image using MONAI
# image = ants.image_read(path1)
# image2 = ants.image_read(path2)
# print(image.shape, image2.shape)
# # Plotting the binary image
# #plt.imshow(binary_image, cmap='gray')
# #plt.show()
# # Convert the numpy array to a PyTorch tensor
# binary_image = torch.tensor(image.numpy())
# binary_image2 = torch.tensor(image2.numpy())

# dice_metric = monai.metrics.DiceMetric(include_background=False) # dice score for inference

# binary_image3 = np.zeros((9, 9))
# binary_image3[3:6, 3:6] = 1
# binary_image3 = torch.tensor(binary_image3).unsqueeze(0).float()


# binary_image4 = np.zeros((9, 9))
# binary_image4[2:5, 2:5] = 1
# binary_image4 = torch.tensor(binary_image4).unsqueeze(0).float()
# class BCEDiceLoss(nn.Module):

#     """
#     Compute the BCE Dice Loss

#     Args:
#     nn.Module: PyTorch module

#     Returns:
#     loss: float, containing the loss computed as bce + dice_loss
#     dice_score: float, containing the Dice score
#     bce: float, containing the BCE loss
#     """
#     def __init__(self, epsilon=1e-6):
#         super(BCEDiceLoss, self).__init__()
#         self.epsilon = epsilon
 
#     def forward(self, predictions, targets):
#         #Convert to float
#         predictions = predictions.float()
#         targets = targets.float()

#         predictions_flat = predictions.reshape(predictions.size(0), -1)
#         targets_flat = targets.reshape(targets.size(0), -1)
#         intersection = (predictions_flat * targets_flat).sum(1)
#         sum_pred_target = predictions_flat.sum(1) + targets_flat.sum(1)
#         dice = (2. * intersection + self.epsilon) / (sum_pred_target + self.epsilon)
 
#         # Combine BCE and Dice Loss
#         dice_score = dice.mean()
 
#         return dice_score

# binary_image2 = binary_image2 > 0.5
# binary_image = binary_image.unsqueeze(0).float()
# binary_image2 = binary_image2.unsqueeze(0).float()
# #Compute the BCE Dice Loss

# criterion = BCEDiceLoss()
# Dice = criterion(binary_image3, binary_image4)

# # Compute the Dice score
# dice_metric(binary_image3, binary_image4)
# dice_score = dice_metric.aggregate().item()

# print('Manual dice', Dice,'Dice score:', dice_score)