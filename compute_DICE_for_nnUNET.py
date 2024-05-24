import os
import ants
import numpy as np

# Fonction pour calculer le score DICE
def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
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

# Chemin vers les dossiers contenant les images
prediction_dir = "/home/user/Documents/raph/temporary/results_predictions_nnUNet"
mask_dir = "/home/user/Documents/raph/nnUNet/nnUNet_raw/Dataset011/labelsTs"

# Boucle sur les fichiers ISLES_2xx.nii.gz
dice_scores = []
for i in range(1, 51):
    file_name = f"ISLES_2{i:02d}.nii.gz"
    mask_file_path = os.path.join(mask_dir, file_name)
    
    if os.path.exists(mask_file_path):
        mask_img = ants.image_read(mask_file_path).numpy()
        
        # Suppose que la vérité terrain (ground truth) est disponible sous le même format
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

# Calculer et afficher la moyenne du score DICE
if dice_scores:
    average_dice = sum(dice_scores) / len(dice_scores)
    print(f"Score DICE moyen: {average_dice:.4f}")
else:
    print("Aucun score DICE calculé")
