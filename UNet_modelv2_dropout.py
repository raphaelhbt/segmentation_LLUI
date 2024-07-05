import torch
import torch.nn as nn
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_rate= 0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.instnorm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.instnorm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, dropout_rate=0.0): # Because for the first block we don't want to downsample
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        skip = x
        return skip, x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2 , out_channels, stride=1, dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, dropout_rate=dropout_rate)


    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(UNet3D, self).__init__()
        
        self.enc1 = DownBlock(in_channels, 32, stride=1, dropout_rate=dropout_rate)
        self.enc2 = DownBlock(32, 64, dropout_rate=dropout_rate)
        self.enc3 = DownBlock(64, 128, dropout_rate=dropout_rate)
        self.enc4 = DownBlock(128, 256, dropout_rate=dropout_rate)
        self.enc5 = DownBlock(256, 320, dropout_rate=dropout_rate)
        
        self.bottleneck_conv_part1 = ConvBlock(320, 320, stride=2, dropout_rate=dropout_rate)
        self.bottleneck_conv_part2 = ConvBlock(320, 320, stride=1, dropout_rate=dropout_rate)
        
        self.up5 = UpBlock(320, 320, dropout_rate=dropout_rate)
        self.up4 = UpBlock(320, 256, dropout_rate=dropout_rate)
        self.up3 = UpBlock(256, 128, dropout_rate=dropout_rate)
        self.up2 = UpBlock(128, 64, dropout_rate=dropout_rate)
        self.up1 = UpBlock(64, 32, dropout_rate=dropout_rate)
        
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        skip5, x = self.enc5(x)

        # Bottleneck
        x = self.bottleneck_conv_part1(x)
        x = self.bottleneck_conv_part2(x)

        # Decoder
        x = self.up5(x, skip5)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        
        # Final Convolution
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

class InitWeights_He(object): # He initialization
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = UNet3D(3, 1, 0.2).to(device)
#summary(model, (3, 128, 128, 128))

# Création du modèle
#net = UNet3D(in_channels=1, out_channels=2)
 
# Création d'un writer pour TensorBoard
#writer = SummaryWriter('runs/unet3d_experiment_1')
 
# Création d'un tenseur de données d'entrée factice
#images = torch.randn(1, 1, 128, 128, 128)
 
# Ajout du modèle au writer de TensorBoard
#writer.add_graph(net, images)
#writer.close()