import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        if psi.size()[2:] != x.size()[2:]:
            psi = F.interpolate(psi, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        return x * psi

class StochasticFeaturePyramid(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.2):
        super(StochasticFeaturePyramid, self).__init__()
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        identity = x
        out1 = x
        
        out2 = self.conv1(x)
        out2 = self.bn1(out2)
        out2 = F.relu(out2)
        out2 = self.dropout(out2)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        
        if self.training:
            
            return F.relu(out1 + out2 * 0.8)
        else:
            
            return F.relu(out1 + out2)

class DynamicRegularization(nn.Module):
    def __init__(self, in_channels, reduction=16, min_channels=8):
        super(DynamicRegularization, self).__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        channels = max(min_channels, in_channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.l1_strength = nn.Parameter(torch.tensor(0.001))
        
    def forward(self, x):
        
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.size(1)}")
        
        channel_att = self.avg_pool(x)
        channel_att = self.channel_gate(channel_att)
        
        x = x * channel_att
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_gate(spatial)
        
        x = x * (0.8 + 0.2 * spatial_att)  
    
        if self.training:
            l1_strength = torch.clamp(torch.sigmoid(self.l1_strength), 0.0001, 0.01)
            l1_reg = torch.abs(x).mean() * l1_strength
            self.l1_reg_value = l1_reg
        else:
            self.l1_reg_value = 0
                
        return x

class BoundaryRefinementModule(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryRefinementModule, self).__init__()
        
        self.edge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        edge_features = torch.abs(self.edge_conv(x))
        
        combined = torch.cat([x, edge_features], dim=1)
        
        refined = self.refine(combined)
        
        return x + refined * 0.1  

class UNetResNet50(nn.Module):
    def __init__(self, n_classes=1):
        super(UNetResNet50, self).__init__()
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        for param in list(self.resnet.parameters())[:20]:  
            param.requires_grad = False
        
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.pool1 = self.resnet.maxpool
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        self.sfp1 = StochasticFeaturePyramid(64, dropout_rate=0.2)
        self.sfp2 = StochasticFeaturePyramid(256, dropout_rate=0.2)
        self.sfp3 = StochasticFeaturePyramid(512, dropout_rate=0.3)
        self.sfp4 = StochasticFeaturePyramid(1024, dropout_rate=0.3)
        self.sfp5 = StochasticFeaturePyramid(2048, dropout_rate=0.3)
        
        self.center = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        self.att5 = AttentionGate(F_g=256, F_l=2048, F_int=256)
        self.att4 = AttentionGate(F_g=128, F_l=1024, F_int=128)
        self.att3 = AttentionGate(F_g=64, F_l=512, F_int=64)
        self.att2 = AttentionGate(F_g=64, F_l=256, F_int=64)
        
        self.decoder5 = nn.Sequential(
            nn.Conv2d(256 + 2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128 + 1024, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64 + 512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64 + 256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.reg5 = DynamicRegularization(256)
        self.reg4 = DynamicRegularization(128)
        self.reg3 = DynamicRegularization(64)
        self.reg2 = DynamicRegularization(64)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        self.boundary_refinement = BoundaryRefinementModule(32)
        
        self.output = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def get_l1_regularization_term(self):
        reg_term = 0
        for module in self.modules():
            if isinstance(module, DynamicRegularization) and hasattr(module, 'l1_reg_value'):
                reg_term += module.l1_reg_value
        return reg_term
        
    def forward(self, x):
        input_size = x.size()[2:]
        
        e1 = self.encoder1(x)
        e1 = self.sfp1(e1)
        
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        e2 = self.sfp2(e2)
        
        e3 = self.encoder3(e2)
        e3 = self.sfp3(e3)
        
        e4 = self.encoder4(e3)
        e4 = self.sfp4(e4)
        
        e5 = self.encoder5(e4)
        e5 = self.sfp5(e5)
        
        c = self.center(e5)
        
        d5 = self.upconv5(c)
        if d5.size()[2:] != e5.size()[2:]:
            d5 = F.interpolate(d5, size=e5.size()[2:], mode='bilinear', align_corners=False)
        a5 = self.att5(d5, e5)
        d5 = torch.cat([d5, a5], dim=1)
        d5 = self.decoder5(d5)
        d5 = self.reg5(d5)
        
        d4 = self.upconv4(d5)
        if d4.size()[2:] != e4.size()[2:]:
            d4 = F.interpolate(d4, size=e4.size()[2:], mode='bilinear', align_corners=False)
        a4 = self.att4(d4, e4)
        d4 = torch.cat([d4, a4], dim=1)
        d4 = self.decoder4(d4)
        d4 = self.reg4(d4)
        
        d3 = self.upconv3(d4)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = F.interpolate(d3, size=e3.size()[2:], mode='bilinear', align_corners=False)
        a3 = self.att3(d3, e3)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.decoder3(d3)
        d3 = self.reg3(d3)
        
        d2 = self.upconv2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = F.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=False)
        a2 = self.att2(d2, e2)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.decoder2(d2)
        d2 = self.reg2(d2)
        
        x = self.final_conv(d2)
        x = self.boundary_refinement(x)
        logits = self.output(x)
        
        if logits.size()[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
            
        return logits