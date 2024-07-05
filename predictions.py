import UNet_modelv2_dropout as unet2_dropout
import torch
import pandas as pd
import os
import ants
from torch.utils.data import DataLoader

# Parameters
NB_FORWARD = 1000
dropout=0.5
BATCH_SIZE = 2
weights_path = '/home/user/Documents/raph/code/saved_models/model_epoch_65.pth'

bids_dir = "/home/user/Documents/raph/preprocessed_datasets/ISLES2022"
parameters = ['FLAIR', 'ADC', 'dwi', 'msk']
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instantiation
model = unet2_dropout.UNet3D(in_channels=3, out_channels=1, dropout_rate=dropout).to(DEVICE)

# Loading the weights
state_dict = torch.load(weights_path)

# Loading the weights into the model
model.load_state_dict(state_dict)

# Dropout activation and deactivation
def enable_dropout(model):
	""" Function to enable the dropout layers during test-time """
	for m in model.modules():
		if m.__class__.__name__.startswith('Dropout'):
			m.train()
			
def predict(net, inputs):
	# to store n_forward predictions on the same batch
	dropout_predictions = torch.empty((0,inputs.size(0),inputs.size(1),inputs.size(2),inputs.size(3)))

	# bayesian inference
	for f_pass in range(NB_FORWARD):
		with torch.no_grad():
			# predict with dropout
			enable_dropout(net)
			mask_pred = net(inputs)

            # concatenate prediction to the other made on the same batch
			dropout_predictions = torch.cat((dropout_predictions,mask_pred.cpu().softmax(dim=1).unsqueeze(dim=0)),dim=0) # Output shape is (n_forward, batch_size, 128, 128, 128)
			
	# Take the mean across the 0th dimension to reduce shape to (batch_size, 128, 128, 128)
	dropout_predictions_mean = dropout_predictions.mean(dim=0)

	return dropout_predictions_mean

#Helpers
def get_patient_ids(file_path):
    
    """
    Get all patient IDs from a TSV file.

    Parameters:
        file_path (str): The path to the TSV file containing patient IDs.

    Returns:
        list: A list containing all patient IDs extracted from the TSV file.
    """
    
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(file_path, sep='\t')
    
    # Extract the 'participant_id' column into a list
    patient_ids = df['participant_id'].tolist()
    
    return patient_ids

def retrieve_img_paths(bids_dir, parameters, subject_id, session_id):
    """
    Retrieve the paths of the input images from the BIDS directory.

    Arguments:
        bids_dir (str): The path to the BIDS directory.
        parameters (list): A list of the parameters to be extracted.
        subject_id (str): The ID of the subject.
        session_id (str): The ID of the session.

    Returns:
        list: A list containing the paths of the input images.
    """
    img_paths = []
    for parameter in parameters:
        if parameter == 'adc' or parameter == 'dwi':
            # Create the path to the NIfTI file (DWI & ADC)
            file_path = os.path.join(bids_dir, f"{subject_id}", f"ses-{session_id}", "dwi", f"{subject_id}_ses-{session_id}_{parameter}.nii.gz")
        elif parameter == 'FLAIR':
            # Create the path to the NIfTI file (FLAIR)
            file_path = os.path.join(bids_dir, f"{subject_id}", f"ses-{session_id}", "anat", f"{subject_id}_ses-{session_id}_{parameter}.nii.gz")
        elif parameter == 'msk':
            # Create the path to the NIfTI file (MASK)
            file_path = os.path.join(bids_dir, "derivatives", f"{subject_id}", f"ses-{session_id}", f"{subject_id}_ses-{session_id}_{parameter}.nii.gz")
        img_paths.append(file_path)

    for i, path in enumerate(img_paths):
        if not os.path.exists(path):
            print(f"Warning: NIfTI file for parameter {parameters[i]} not found for subject {subject_id} and session {session_id}. Skipping.")
            return None

    return img_paths
		
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, bids_dir, patient_ids, session_id='0001'):
        self.bids_dir = bids_dir
        self.patient_ids = ['sub-' + id.split('case')[-1][1:] for id in patient_ids[-50:]]  # Use only the last 50 patient IDs
        self.session_id = session_id
        self.img_paths = self._retrieve_img_paths()

    def _retrieve_img_paths(self):
        img_paths = []
        for patient_id in self.patient_ids:
            paths = retrieve_img_paths(self.bids_dir, parameters, patient_id, self.session_id)
            if paths is not None:
                img_paths.append(paths)
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        paths = self.img_paths[idx]
        FLAIR_path, adc_path, dwi_path, msk_path = paths

        # Load images
        FLAIR_img = torch.tensor(ants.image_read(FLAIR_path).numpy(), dtype=torch.float32)
        adc_img = torch.tensor(ants.image_read(adc_path).numpy(), dtype=torch.float32)
        dwi_img = torch.tensor(ants.image_read(dwi_path).numpy(), dtype=torch.float32)
        msk_img = torch.tensor(ants.image_read(msk_path).numpy().squeeze(), dtype=torch.float32)  # Mask
  
        concatenated_data = torch.stack((FLAIR_img, adc_img, dwi_img), dim=0)
        concatenated_data = concatenated_data.type(torch.float32)
        return concatenated_data, msk_img
    
# Load the test dataset
test_dataset = TestDataset(bids_dir, get_patient_ids(os.path.join(bids_dir, "participants.tsv")))

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Iterate over the test dataset
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):

        ## IMAGES ARE CORRECTLY LOADED BUT HAVE TO TRY THE REST OF THE CODE
        # Move the inputs and targets to the device
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Perform the forward pass
        outputs= predict(model, inputs)

        # Save the outputs
        ants_image = ants.image_write(ants.from_numpy(outputs[0].numpy()), 'output.nii.gz')

