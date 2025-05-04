# baseline_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicUNet(nn.Module):
    """Standard U-Net implementation without enhancements"""
    def __init__(self, n_classes=1):
        super(BasicUNet, self).__init__()
        # Encoder
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.enc5 = self._conv_block(512, 1024)
        
        # Decoder
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = self._conv_block(1024, 512)
        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # For saving activations
        self.activations = {}
        self.register_hooks()
        
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def register_hooks(self):
        """Register hooks to save activations"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        self.enc1[0].register_forward_hook(get_activation('enc1_conv1'))
        self.enc2[0].register_forward_hook(get_activation('enc2_conv1'))
        self.enc3[0].register_forward_hook(get_activation('enc3_conv1'))
        self.enc4[0].register_forward_hook(get_activation('enc4_conv1'))
        self.enc5[0].register_forward_hook(get_activation('enc5_conv1'))
        
        self.dec5[0].register_forward_hook(get_activation('dec5_conv1'))
        self.dec4[0].register_forward_hook(get_activation('dec4_conv1'))
        self.dec3[0].register_forward_hook(get_activation('dec3_conv1'))
        self.dec2[0].register_forward_hook(get_activation('dec2_conv1'))
    
    def forward(self, x):
        # Clear activations
        self.activations = {}
        
        # Encoder
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, kernel_size=2, stride=2)
        
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, kernel_size=2, stride=2)
        
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, kernel_size=2, stride=2)
        
        e4 = self.enc4(p3)
        p4 = F.max_pool2d(e4, kernel_size=2, stride=2)
        
        e5 = self.enc5(p4)
        
        # Decoder
        d5 = self.up5(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        
        return out