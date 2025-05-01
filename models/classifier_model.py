"""
Classifier Model for Fruit Ripeness Detection System
"""
import torch
import torch.nn as nn
from torchvision import models

class FruitClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(FruitClassifier, self).__init__()
        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        
        # Modify the classifier
        in_features = self.model.classifier[1].in_features
        
        # Replace classifier with custom one
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        # Freeze all base layers
        for param in self.model.features.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self, layers_to_unfreeze=5):
        # Unfreeze the last few layers
        for i, layer in enumerate(self.model.features):
            if i >= len(self.model.features) - layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True