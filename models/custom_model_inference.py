import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from huggingface_hub import hf_hub_download

class CustomModelInference:
    """
    Class to handle custom PyTorch model inference for ripeness classification
    Supports various model architectures and class structures
    """
    
    def __init__(self, device=None):
        """Initialize the inference handler"""
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}
        self.model_configs = {}
        
        # Define model configurations for different fruits
        self._init_model_configs()
    
    def _init_model_configs(self):
        """Initialize configurations for different fruit classification models"""
        # Strawberry model configuration
        self.model_configs["strawberry"] = {
            "repo_id": "TentenPolllo/strawberryripe",
            "filename": "best_strawberry_model_kfold.pth",
            "input_size": (224, 224),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "model_arch": "convnext_tiny",
            "num_classes": 3,  # Unripe, Ripe, Overripe
            "class_names": ["Unripe", "Ripe", "Overripe"]
        }
        
        self.model_configs["pineapple"] = {
            "repo_id": "TentenPolllo/pineappleripeness",  # Placeholder - update when available
            "filename": "best_pineapple_model.pth",       # Placeholder - update when available
            "input_size": (224, 224),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "model_arch": "convnext_tiny",
            "num_classes": 3,  # Placeholder - update when available
            "class_names": ["Unripe", "Ripe", "Overripe"]  # Placeholder - update when available
        }
    
    def _create_model(self, model_config):
        """Create model architecture based on configuration"""
        import torchvision.models as models
        import torch.nn as nn
        
        num_classes = model_config.get("num_classes", 3)
        
        if model_config["model_arch"] == "convnext_tiny":
            model = models.convnext_tiny(weights=None)
            model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(768, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        elif model_config["model_arch"] == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {model_config['model_arch']}")
        
        return model
    
    def load_model(self, fruit_type):
        """
        Load the classification model for a specific fruit type
        
        Args:
            fruit_type: Type of fruit (e.g., "strawberry", "pineapple")
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        fruit_type = fruit_type.lower()
        
        # Check if model is already loaded
        if fruit_type in self.loaded_models:
            return True
        
        # Check if we have configuration for this fruit type
        if fruit_type not in self.model_configs:
            print(f"No model configuration available for {fruit_type}")
            return False
        
        try:
            # Get model configuration
            model_config = self.model_configs[fruit_type]
            
            # Download model from Hugging Face Hub
            model_path = hf_hub_download(
                repo_id=model_config["repo_id"],
                filename=model_config["filename"]
            )
            
            # Load the model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model metadata from checkpoint
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                
                # Update configuration with metadata from checkpoint if available
                if "class_names" in checkpoint:
                    model_config["class_names"] = checkpoint["class_names"]
                    model_config["num_classes"] = len(checkpoint["class_names"])
                if "model_arch" in checkpoint:
                    model_config["model_arch"] = checkpoint["model_arch"]
                if "input_size" in checkpoint:
                    model_config["input_size"] = checkpoint["input_size"]
                if "normalize_mean" in checkpoint:
                    model_config["normalize_mean"] = checkpoint["normalize_mean"]
                if "normalize_std" in checkpoint:
                    model_config["normalize_std"] = checkpoint["normalize_std"]
            else:
                state_dict = checkpoint
            
            # Create model using the updated configuration
            model = self._create_model(model_config)
            
            # Load the state dictionary
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            # Save loaded model and updated config
            self.loaded_models[fruit_type] = model
            self.model_configs[fruit_type] = model_config
            
            print(f"Successfully loaded {fruit_type} model with {model_config['num_classes']} classes: {model_config['class_names']}")
            return True
            
        except Exception as e:
            print(f"Error loading {fruit_type} model: {str(e)}")
            return False
    
    def classify_image(self, image, fruit_type):
        """
        Classify an image using the appropriate model for the fruit type
        
        Args:
            image: PIL Image
            fruit_type: Type of fruit
            
        Returns:
            Dictionary with class confidences
        """
        fruit_type = fruit_type.lower()
        
        # Make sure model is loaded
        if fruit_type not in self.loaded_models:
            success = self.load_model(fruit_type)
            if not success:
                return {"error": f"Failed to load model for {fruit_type}"}
        
        try:
            # Get model and config
            model = self.loaded_models[fruit_type]
            model_config = self.model_configs[fruit_type]
            
            # Create transform
            transform = transforms.Compose([
                transforms.Resize(model_config["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    model_config["normalize_mean"],
                    model_config["normalize_std"]
                )
            ])
            
            # Preprocess image
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Convert to dictionary
            class_names = model_config["class_names"]
            confidences = {}
            
            for i, prob in enumerate(probabilities):
                if i < len(class_names):
                    # Use standardized ripeness labels if possible
                    class_name = class_names[i]
                    if class_name.lower() in ["unripe", "strawberryunripe", "green"]:
                        label = "Unripe"
                    elif class_name.lower() in ["ripe", "strawberryripe", "red"]:
                        label = "Ripe"
                    elif class_name.lower() in ["overripe", "rotten", "strawberryrotten"]:
                        label = "Overripe"
                    else:
                        label = class_name
                    
                    confidences[label] = prob.item()
                else:
                    confidences[f"Class_{i}"] = prob.item()
            
            return confidences
            
        except Exception as e:
            return {"error": str(e)}