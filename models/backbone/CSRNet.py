import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from gau_conv import *

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_back_layers(self.backend_feat,in_channels = 512, Gconv = True)
        self.output_layer = GauConv2d(64, 1, 8)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x,scale_factor=8)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            Conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [Conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [Conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_back_layers(cfg, in_channels = 3,batch_norm=False, Gconv = False):
    layers = []
    for v in cfg:
        GauConv2d = GauConv2d(in_channels, v, 2)
        if batch_norm:
            layers += [GauConv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [GauConv2d, nn.ReLU(inplace=True)]
        in_channels = v
    for v in cfg:
        GauConv2d = GauConv2d(in_channels, v, 4)
        if batch_norm:
            layers += [GauConv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [GauConv2d, nn.ReLU(inplace=True)]
        in_channels = v
    for v in cfg:
        GauConv2d = GauConv2d(in_channels, v, 8)
        if batch_norm:
            layers += [GauConv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [GauConv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers) 