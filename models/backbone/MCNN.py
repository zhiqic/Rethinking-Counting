import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import *
from gau_conv import *

class MCNN(nn.Module):
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        self.branch1 = nn.Sequential(GauConv2d( 3, 16, 16,bn=bn),
                                     nn.MaxPool2d(2),
                                     GauConv2d(16, 32, 16,bn=bn),
                                     nn.MaxPool2d(2),
                                     GauConv2d(32, 16, 2, bn=bn),
                                     GauConv2d(16,  8, 2, bn=bn))
        
        self.branch2 = nn.Sequential(GauConv2d( 3, 20, 16, bn=bn),
                                     nn.MaxPool2d(2),
                                     GauConv2d(20, 40, 16, bn=bn),
                                     nn.MaxPool2d(2),
                                     GauConv2d(40, 20, 4,  bn=bn),
                                     GauConv2d(20, 10, 4,  bn=bn))
        
        self.branch3 = nn.Sequential(GauConv2d( 3, 24, 16, bn=bn),
                                     nn.MaxPool2d(2),
                                     GauConv2d(24, 48, 16, bn=bn),
                                     nn.MaxPool2d(2),
                                     GauConv2d(48, 24, 8, bn=bn),
                                     GauConv2d(24, 12, 8, bn=bn))
        
        self.fuse = nn.Sequential(GauConv2d( 30, 1, 16, bn=bn))

        initialize_weights(self.modules())   
        
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        x = F.upsample(x,scale_factor=4)
        return x
