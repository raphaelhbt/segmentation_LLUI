from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch
import monai
from monai.networks.nets import DynUNet

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
dropout = 0.2
 
# Initialize the DynUNet
model = DynUNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    strides=strides,
    upsample_kernel_size=up_sample_kernel_size,
    filters=filters,
    dropout = dropout
)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
#summary(model, (in_channels, 128, 128, 128))

# Création d'un writer pour TensorBoard
#writer = SummaryWriter('runs/unet3d_experiment_1')
 
# Création d'un tenseur de données d'entrée factice
#images = torch.randn(1, 3, 128, 128, 128)
 
# Ajout du modèle au writer de TensorBoard
#writer.add_graph(model, images)
#writer.close()