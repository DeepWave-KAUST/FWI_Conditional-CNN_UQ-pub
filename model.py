import torch
import torch.nn as nn
import torchsummary
import torch.nn.functional as F
import math
import torchvision.transforms.functional as TF

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

## Context Unet, two inpt, one output
class CONTEXT_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1,  n_feat=256, n_cfeat=10, feature=[32, 64, 128, 256, 512],
    ):
        super(CONTEXT_UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.feature = feature

        # Down part of UNET
        self.downs1 = DoubleConv(in_channels, feature[0])
        self.downs2 = DoubleConv(feature[0],  feature[1])
        self.downs3 = DoubleConv(feature[1],  feature[2])
        self.downs4 = DoubleConv(feature[2],  feature[3])
        self.downs5 = DoubleConv(feature[3],  feature[4])

        self.contextembed1 = EmbedFC(n_cfeat, feature[4]*2)
        self.contextembed2 = EmbedFC(n_cfeat, feature[3]*2)
        self.contextembed3 = EmbedFC(n_cfeat, feature[2]*2)
        self.contextembed4 = EmbedFC(n_cfeat, feature[1]*2)
        self.contextembed5 = EmbedFC(n_cfeat, feature[0]*2)

        
        # Up part of UNET
        self.ups1 = nn.ConvTranspose2d(feature[4]*2, feature[4], kernel_size=2, stride=2)
        self.ups2 = DoubleConv(feature[4]*2, feature[4]) 
        self.ups3 = nn.ConvTranspose2d(feature[3]*2, feature[3], kernel_size=2, stride=2)
        self.ups4 = DoubleConv(feature[3]*2, feature[3])  
        self.ups5 = nn.ConvTranspose2d(feature[2]*2, feature[2], kernel_size=2, stride=2)
        self.ups6 = DoubleConv(feature[2]*2, feature[2]) 
        self.ups7 = nn.ConvTranspose2d(feature[1]*2, feature[1], kernel_size=2, stride=2)
        self.ups8 = DoubleConv(feature[1]*2, feature[1])         
        self.ups9 = nn.ConvTranspose2d(feature[0]*2, feature[0], kernel_size=2, stride=2)
        self.ups10 = DoubleConv(feature[0]*2, feature[0]) 

        self.bottleneck = DoubleConv(feature[-1], feature[-1]*2)
        self.final_conv = nn.Conv2d(feature[0], out_channels, kernel_size=1)

    def forward(self, x, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label,
        #  context_mask says which samples to block the context on

        # Down part of UNET
        x1 = self.downs1(x) # skip1 = x1
        x2 = self.pool(x1)
        x3 = self.downs2(x2) # skip2 = x3
        x4 = self.pool(x3)
        x5 = self.downs3(x4) # skip3 = x5
        x6 = self.pool(x5)        
        x7 = self.downs4(x6) # skip4 = x7  
        x8 = self.pool(x7) 
        x9 = self.downs5(x8) # skip5 = x9 
        x10 = self.pool(x9) 

        skip1 = x1
        skip2 = x3
        skip3 = x5
        skip4 = x7
        skip5 = x9

        x11 = self.bottleneck(x10)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1,self.feature[4]*2,1,1)
        cemb2 = self.contextembed2(c).view(-1,self.feature[3]*2,1,1)
        cemb3 = self.contextembed3(c).view(-1,self.feature[2]*2,1,1)
        cemb4 = self.contextembed4(c).view(-1,self.feature[1]*2,1,1)    
        cemb5 = self.contextembed5(c).view(-1,self.feature[0]*2,1,1)     

        # Up part of UNET       
        y1 = self.ups1(x11*cemb1)
        if y1.shape != skip5.shape:
            y1 = TF.resize(y1, size=skip5.shape[2:])
        y2 = torch.cat((skip5, y1), dim=1)
        y3 = self.ups2(y2)

        y4 = self.ups3(y3*cemb2)
        if y4.shape != skip4.shape:
            y4 = TF.resize(y4, size=skip4.shape[2:])
        y5 = torch.cat((skip4, y4), dim=1)
        y6 = self.ups4(y5)

        y7 = self.ups5(y6*cemb3)
        if y7.shape != skip3.shape:
            y7 = TF.resize(y7, size=skip3.shape[2:])
        y8 = torch.cat((skip3, y7), dim=1)
        y9 = self.ups6(y8)

        y10 = self.ups7(y9*cemb4)
        if y10.shape != skip2.shape:
            y10 = TF.resize(y10, size=skip2.shape[2:])
        y11 = torch.cat((skip2, y10), dim=1)
        y12 = self.ups8(y11)

        y13 = self.ups9(y12*cemb5)
        if y13.shape != skip1.shape:
            y13 = TF.resize(y13, size=skip1.shape[2:])
        y14 = torch.cat((skip1, y13), dim=1)
        y15 = self.ups10(y14)


        return self.final_conv(y15)