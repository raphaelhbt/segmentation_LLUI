# segmentation_LLUI

This project is divided in 3 parts : preprocessing, 3D U-Net training and uncertainty estimation of the predictions. The results of 3D U-Net are compared to nnU-Net's on our preprocessed data.
The data used is the ISLES2022 dataset, preprocessed into the BIDS format using the preprocessing.py file.

Usage of the files:

**compute_DICE_for_nnUNET.py** : Just useful when you use nnU-Net to then check the DICE of the outputs.

**dataset_conversion_BIDS_to_nnUNet.py** : To convert a BIDS dataset to nnU-Net's dataset type (they require a specific architecture in input).

**predictions.py** : file to compute uncertainty estimation with MCD on the test dataset.

**preprocessing.py** : file to preprocess the ISLES-2022 dataset.

**preprocessingSOOP.py** : file to preprocess the SOOP dataset.

**test.ipynb** : just a dummy file used to test new things before implementing them in the real code.

**training_plotting.ipynb** : File used to:
- Overlay the model's predictions with the groundtruth on the FLAIR image of the patient. The cursor allows to go through the different 2D plans of the 3D volume. There is also a boxplot of the Dices + Median and std printed.
- Compute the uncertainty rates and boxplot them, with mean, median and std.
- Display the correlation between lesion volume and uncertainty rate.
- Boxplot the Dice scores obtained with MCD, with mean, median and std.
- Display the correlation between DICE and uncertainty rate.
- Display a 3D graph of correlation between Lesion volume, Dice score and uncertainty rate.

**training.py** : first training loop with whole patch extraction and almost everything hard coded by hand. It works but there is an issue in the code that prevents us from having good results.

**training2.py** : Working training loop. See the other readme file to know more about it.
