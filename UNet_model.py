import torch
import torch.nn as nn
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.instnorm1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.instnorm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.instnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.instnorm2(x)
        x = self.relu2(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, in_channels)
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.down(x)
        return skip, x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        self.first_conv = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)

        self.enc1 = DownBlock(32, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 320)
        self.enc5 = DownBlock(320, 320)
        
        self.bottleneck_conv = ConvBlock(320, 320)
        
        self.up5 = UpBlock(320, 320)
        self.up4 = UpBlock(320, 256)
        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up1 = UpBlock(64, 32)
        
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        # Initial Convolution
        x = self.first_conv(x)
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        skip5, x = self.enc5(x)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Decoder
        x = self.up5(x, skip5)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        
        # Final Convolution
        x = self.final_conv(x)
        return x

# Création du modèle
#net = UNet3D(in_channels=1, out_channels=2)
 
# Création d'un writer pour TensorBoard
#writer = SummaryWriter('runs/unet3d_experiment_1')
 
# Création d'un tenseur de données d'entrée factice
#images = torch.randn(1, 1, 128, 128, 128)
 
# Ajout du modèle au writer de TensorBoard
#writer.add_graph(net, images)
#writer.close()