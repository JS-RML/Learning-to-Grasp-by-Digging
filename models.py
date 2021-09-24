from torch.nn import Module, Conv2d
import torch.nn.functional as F
import resnet
#import numpy as np
import torch

class Dignet(Module):
    def __init__(self, num_input_channels=3, num_output_channels=3*144):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv2 = Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv3 = Conv2d(32, num_output_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        # x is input 1, input2 is input 2       
        x = self.resnet18.features(x)       
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.conv3(x)
        x = x.view(-1,3, 144, 240, 240)
#        print(x.shape)
        return x
    
    
