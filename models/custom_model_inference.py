import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from huggingface_hub import hf_hub_download
import torch.nn as nn
import torchvision.models as models

# Try to import timm conditionally
try:
    from timm.models import convnext_tiny as timm_convnext_tiny
except ImportError:
    print("Warning: timm library not found. Some models may not load correctly.")

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
        # Strawberry model configuration (keep the original implementation)
        self.model_configs["strawberry"] = {
            "repo_id": "TentenPolllo/strawberryripe",
            "filename": "best_strawberry_model_kfold.pth",
            "input_size": (224, 224),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "model_arch": "convnext_tiny",
            "num_classes": 3,  # Unripe, Ripe, Overripe
            "class_names": ["Unripe", "Ripe", "Overripe"],
            "use_timm": False  # Use torchvision implementation
        }
        
        # Pineapple model configuration (use the timm implementation)
        self.model_configs["pineapple"] = {
            "repo_id": "TentenPolllo/pineappleripeness",  # Update with your actual repo
            "filename": "best_pineapple_model.pth",
            "input_size": (224, 224),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "model_arch": "convnext_tiny",
            "num_classes": 3,  # Unripe, Ripe, Overripe
            "class_names": ["Unripe", "Ripe", "Overripe"],
            "use_timm": True,  # Use timm implementation
            "drop_rate": 0.3,
            "drop_path_rate": 0.2
        }
        
        # Add new tomato model configuration
        self.model_configs["tomato"] = {
            "repo_id": "TentenPolllo/tomato",  # Update with your actual repo once published
            "filename": "best_tomato_model.pth",
            "input_size": (256, 256),  # Matches the img_size in the training config
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "model_arch": "convnext_tiny",
            "num_classes": 3,  # Assuming 3 classes: Unripe, Ripe, Overripe
            "class_names": ["Unripe", "Ripe", "Overripe"],  # Update if your classes are different
            "use_timm": True,  # The training script uses timm implementation
            "drop_rate": 0.3,  # Matches the training config
            "drop_path_rate": 0.2  # Called 'stochastic_depth' in the training config
        }
        
        # Add new tomato model configuration
        self.model_configs["mango"] = {
            "repo_id": "TentenPolllo/mango",  # Update with your actual repo once published
            "filename": "best_mango_model.pth",
            "input_size": (256, 256),  # Matches the img_size in the training config
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "model_arch": "convnext_tiny",
            "num_classes": 4,  # Assuming 3 classes: Unripe, Ripe, Overripe
            "class_names": ["Unripe", "Ripe", "Underripe", "Overripe"],  # Update if your classes are different
            "use_timm": True,  # The training script uses timm implementation
            "drop_rate": 0.3,  # Matches the training config
            "drop_path_rate": 0.2  # Called 'stochastic_depth' in the training config
        }
        
    
    def _create_model(self, model_config):
        """Create model architecture based on configuration"""
        num_classes = model_config.get("num_classes", 3)
        
        if model_config["model_arch"] == "convnext_tiny":
            # Check if we should use timm implementation (for pineapple) or torchvision (for strawberry)
            if model_config.get("use_timm", False):
                try:
                    # Use timm implementation like in training
                    model = timm_convnext_tiny(
                        pretrained=False, 
                        drop_rate=model_config.get("drop_rate", 0.3),
                        drop_path_rate=model_config.get("drop_path_rate", 0.2)
                    )
                    # Replace head with the same architecture as in training
                    in_features = model.head.fc.in_features  
                    model.head.fc = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(in_features, num_classes)
                    )
                except (NameError, ImportError) as e:
                    print(f"Error using timm: {e}. Falling back to torchvision implementation.")
                    # Fallback to torchvision if timm is not available
                    model = models.convnext_tiny(weights=None)
                    model.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(768, 128),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, num_classes)
                    )
            else:
                # Use torchvision implementation (for backward compatibility)
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
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
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
                if "config" in checkpoint and isinstance(checkpoint["config"], dict):
                    # Extract config settings relevant to model architecture
                    config = checkpoint["config"]
                    if "drop_rate" in config:
                        model_config["drop_rate"] = config["drop_rate"]
                    if "stochastic_depth" in config:
                        model_config["drop_path_rate"] = config["stochastic_depth"]
            else:
                state_dict = checkpoint
            
            # Create model using the updated configuration
            model = self._create_model(model_config)
            
            # Handle state dict key differences if using timm vs torchvision
            if model_config.get("use_timm", False):
                # Check if we need to adapt keys from training format
                if any(k.startswith("model.") for k in state_dict.keys()):
                    # Remove 'model.' prefix if present (common in training scripts)
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("model."):
                            new_state_dict[k[6:]] = v  # Remove "model." prefix
                        else:
                            new_state_dict[k] = v
                    state_dict = new_state_dict
            
            # Try to load the state dictionary
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading state dict directly: {e}")
                # Try a more flexible loading approach that ignores missing and unexpected keys
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"Loaded with non-strict matching. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
            
            model = model.to(self.device)
            model.eval()
            
            # Save loaded model and updated config
            self.loaded_models[fruit_type] = model
            self.model_configs[fruit_type] = model_config
            
            print(f"Successfully loaded {fruit_type} model with {model_config['num_classes']} classes: {model_config['class_names']}")
            return True
            
        except Exception as e:
            print(f"Error loading {fruit_type} model: {str(e)}")
            import traceback
            traceback.print_exc()
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
                
                # ===== ADDED: Temperature scaling for mangoes =====
                if fruit_type == "mango":
                    temperature = 3.0  # Higher value = more uniform distribution
                    outputs = outputs / temperature
                # ================================================
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Convert to dictionary
            class_names = model_config["class_names"]
            confidences = {}
            
            for i, prob in enumerate(probabilities):
                if i < len(class_names):
                    # Use standardized ripeness labels if possible
                    class_name = class_names[i]
                    
                    # ===== MODIFIED: Special handling for mangoes =====
                    if fruit_type == "mango":
                        if class_name.lower() in ["unripe", "unripe-1-20-", "early_ripe-21-40-"]:
                            label = "Unripe"
                        elif class_name.lower() in ["semiripe", "semi-ripe", "partially_ripe-41-60-", "semi_ripe", "underripe"]:
                            label = "Underripe"
                        elif class_name.lower() in ["ripe", "ripe-61-80-", "fully ripe"]:
                            label = "Ripe"
                        elif class_name.lower() in ["overripe", "over_ripe-81-100-", "over-ripe", "perished", "rotten"]:
                            label = "Overripe"
                        else:
                            label = class_name
                    else:
                        # Original code for other fruits
                        if class_name.lower() in ["unripe", "strawberryunripe", "green"]:
                            label = "Unripe"
                        elif class_name.lower() in ["ripe", "strawberryripe", "red"]:
                            label = "Ripe"
                        elif class_name.lower() in ["overripe", "rotten", "strawberryrotten"]:
                            label = "Overripe"
                        else:
                            label = class_name
                    # ================================================
                    
                    confidences[label] = prob.item()
                else:
                    confidences[f"Class_{i}"] = prob.item()
            
            # ===== ADDED: Post-processing for mangoes =====
            if fruit_type == "mango":
                # Ensure minimum probability values
                for key in confidences:
                    confidences[key] = max(confidences[key], 0.02)
                
                # Renormalize
                total = sum(confidences.values())
                if total > 0:
                    for key in confidences:
                        confidences[key] /= total
            # ===============================================
            
            return confidences
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def test_model(self, image_path, fruit_type):
        """
        Test the model on a single image
        
        Args:
            image_path: Path to the image file
            fruit_type: Type of fruit to test
            
        Returns:
            Classification results
        """
        try:
            # Open the image
            img = Image.open(image_path).convert('RGB')
            
            # Classify using the specified model
            results = self.classify_image(img, fruit_type)
            
            print(f"{fruit_type.capitalize()} ripeness classification results: {results}")
            return results
        except Exception as e:
            print(f"Error testing {fruit_type} model: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}