import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),       
       nn.ReLU(inplace=True),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),
       nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
               
        self.dconv_down1 = double_conv(n_in, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        self.dconv_down5 = double_conv(512, 1024)
       
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp4  = nn.ConvTranspose2d(1024,512,2,stride=2,padding=0)
        self.xUp3  = nn.ConvTranspose2d(512,256,2,stride=2,padding=0)
        self.xUp2  = nn.ConvTranspose2d(256,128,2,stride=2,padding=0)
        self.xUp1  = nn.ConvTranspose2d(128,64,2,stride=2,padding=0)
        
        self.dconv_up4 = double_conv(512 + 512, 512)
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.dconv_up1 = double_conv(64 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        inp = x
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        
        x = self.maxpool(conv4)  
        x = self.dconv_down5(x)
        x = self.xUp4(x)        
        
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.xUp3(x)        
        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.xUp2(x)        
        
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)
        x = self.xUp1(x)        
        
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)
        update = self.conv_last(x)

        return inp + self.stepsize * update
    
