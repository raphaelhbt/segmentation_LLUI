import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
##################################################################################################################################################

class UNet3D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.encoder1 = DoubleConv3D(in_channels, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv3D(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv3D(128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv3D(256, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder5 = DoubleConv3D(512, 1024)

        self.upconv4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv3D(1024, 512)
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(128, 64)

        self.out = nn.Conv3d(64, out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.out(dec1)
