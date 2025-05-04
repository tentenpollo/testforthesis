import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw

def feature_map_visualization(activation, title, max_images=8):
    """Generate a visualization of feature maps"""
    if activation is None or not isinstance(activation, torch.Tensor):
        # Create a placeholder image with error message
        placeholder = Image.new('RGB', (400, 100), color=(240, 240, 240))
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 40), f"No activation data for {title}", fill=(0, 0, 0))
        return placeholder
    
def load_pretrained_unet(device=None):
    """Load pretrained U-Net model from PyTorch Hub"""
    try:
        # Try loading from PyTorch Hub
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, 
                               pretrained=True)
        print("✅ Successfully loaded pretrained U-Net from PyTorch Hub")
        
        if device is not None:
            model = model.to(device)
        
        return model
    except Exception as e:
        print(f"⚠️ Error loading from PyTorch Hub: {str(e)}")
        print("⚠️ Using fallback options...")
        
        # Try loading from cache
        try:
            hub_dir = torch.hub.get_dir()
            model_dir = os.path.join(hub_dir, 'mateuszbuda_brain-segmentation-pytorch')
            if os.path.exists(model_dir):
                model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                    in_channels=3, out_channels=1, init_features=32, 
                                    pretrained=True, force_reload=False)
                print("✅ Loaded pretrained U-Net from local cache")
                
                if device is not None:
                    model = model.to(device)
                
                return model
        except Exception as e2:
            print(f"⚠️ Error loading from cache: {str(e2)}")
            raise RuntimeError("Failed to load pretrained U-Net model") from e2

def register_hooks(model):
    """Register hooks to capture activations from the pretrained U-Net model"""
    # Initialize activations dictionary
    model.activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            model.activations[name] = output.detach()
        return hook
    
    # Register hooks for each layer we want to track
    # First identify the model's key components
    try:
        # Register encoder hooks
        model.encoder1.register_forward_hook(get_activation('enc1_conv1'))
        model.encoder2.register_forward_hook(get_activation('enc2_conv1'))
        model.encoder3.register_forward_hook(get_activation('enc3_conv1'))
        model.encoder4.register_forward_hook(get_activation('enc4_conv1'))
        
        # Register bottleneck hook
        model.bottleneck.register_forward_hook(get_activation('bottleneck'))
        
        # Register decoder hooks
        model.decoder4.register_forward_hook(get_activation('dec4_conv1'))
        model.decoder3.register_forward_hook(get_activation('dec3_conv1'))
        model.decoder2.register_forward_hook(get_activation('dec2_conv1'))
        model.decoder1.register_forward_hook(get_activation('dec1_conv1'))
        
        print("✅ Successfully registered hooks on pretrained U-Net")
    except Exception as e:
        print(f"⚠️ Error registering hooks: {str(e)}")
        print("⚠️ Some visualizations may not be available")
    
    return model

def adapt_input_channels(x, target_channels=3):
    """Adapt input tensor to the required number of channels"""
    current_channels = x.size(1)
    
    if current_channels == target_channels:
        return x
    
    # If input has 1 channel, repeat it 3 times
    if current_channels == 1 and target_channels == 3:
        return x.repeat(1, 3, 1, 1)
    
    # If input has more than 3 channels, take the first 3
    if current_channels > target_channels:
        return x[:, :target_channels, :, :]
    
    # Otherwise, repeat the last channel until we have 3
    result = torch.cat([x, x[:, -1:, :, :].repeat(1, target_channels - current_channels, 1, 1)], dim=1)
    return result