import os
import ants
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate the Dice score
def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

# Paths to the directories containing the images
prediction_dir = "/home/user/Documents/raph/temporary/results_predictions_nnUNet"
mask_dir = "/home/user/Documents/raph/nnUNet/nnUNet_raw/Dataset011/labelsTs"

# Loop over the files ISLES_2xx.nii.gz
dice_scores = []
for i in range(1, 51):
    file_name = f"ISLES_2{i:02d}.nii.gz"
    mask_file_path = os.path.join(mask_dir, file_name)
    
    if os.path.exists(mask_file_path):
        mask_img = ants.image_read(mask_file_path).numpy()
        
        # Suppose that the ground truth is available in the same format
        gt_file_path = os.path.join(prediction_dir, file_name)
        
        if os.path.exists(gt_file_path):
            gt_img = ants.image_read(gt_file_path).numpy()
            
            dice_score = dice(mask_img, gt_img)
            dice_scores.append(dice_score)
            
            print(f"Score DICE pour {file_name}: {dice_score:.4f}")
        else:
            print(f"Fichier de prédiction manquant pour {file_name}")
    else:
        print(f"Fichier {file_name} introuvable")

# Calculate and display the average Dice score
if dice_scores:
    average_dice = sum(dice_scores) / len(dice_scores)
    print(f"Score DICE moyen: {average_dice:.4f}")
else:
    print("Aucun score DICE calculé")

# Plot the boxplot using seaborn
if dice_scores:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dice_scores)
    plt.title("DICE Boxplot")
    plt.ylabel("Score DICE")
    plt.xlabel("Images")
    plt.show()

