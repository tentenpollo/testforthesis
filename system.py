import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import io

from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import os

from improved_visualization import create_enhanced_visualization, create_side_by_side_comparison, create_combined_visualization
from mask_refinement import refine_mask, get_mask_quality_metrics
from models.segmentation_model import UNetResNet50
from models.classifier_model import FruitClassifier
from utils.helpers import apply_mask_to_image
from models.custom_model_inference import CustomModelInference

def resize_and_compress_image(image_path, max_size=(800, 800), quality=85, max_file_size_kb=1024):
    
        original_size_kb = os.path.getsize(image_path) / 1024
        
        if original_size_kb <= max_file_size_kb:
            print(f"Image already within size limits ({original_size_kb:.2f} KB)")
            return image_path
        
        # Create output path
        filename, ext = os.path.splitext(image_path)
        output_path = f"{filename}_resized{ext}"
        
        # Open the image
        img = Image.open(image_path)
        
        # Resize if needed
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            print(f"Resized image to {img.width}x{img.height}")
        
        # Compress with decreasing quality until size is acceptable
        current_quality = quality
        while current_quality > 10:  # Don't go below quality 10
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=current_quality)
            file_size_kb = len(buffer.getvalue()) / 1024
            
            if file_size_kb <= max_file_size_kb:
                # Save the file
                img.save(output_path, format="JPEG", quality=current_quality)
                print(f"Compressed image to {file_size_kb:.2f} KB (quality: {current_quality})")
                return output_path
                
            current_quality -= 5
        
        current_scale = 0.9
        while current_scale > 0.1:
            new_size = (int(img.width * current_scale), int(img.height * current_scale))
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG", quality=current_quality)
            file_size_kb = len(buffer.getvalue()) / 1024
            
            if file_size_kb <= max_file_size_kb:
                # Save the file
                resized_img.save(output_path, format="JPEG", quality=current_quality)
                print(f"Further resized to {new_size[0]}x{new_size[1]} and compressed to {file_size_kb:.2f} KB")
                return output_path
                
            current_scale -= 0.1
        
        print("Warning: Could not reduce image below target size. API may still reject it.")
        img.save(output_path, format="JPEG", quality=10)
        return output_path
def debug_and_fix_enhanced_results(results):
    """Function to debug and fix issues with enhanced results display"""
    
    # Check if we have any error messages
    if "error" in results:
        print(f"Error in enhanced results: {results['error']}")
        return results
    
    # Ensure fruits_data is present and properly formatted
    if "fruits_data" not in results or not results["fruits_data"]:
        print("No fruits_data found in results - creating default entry")
        # Create a minimal default fruit data entry
        results["fruits_data"] = [{
            "index": 0,
            "bbox": {
                "x": results["original_image"].width // 2,
                "y": results["original_image"].height // 2,
                "width": results["original_image"].width,
                "height": results["original_image"].height,
                "confidence": 1.0,
                "class": results.get("fruit_type", "unknown")
            },
            # Additional fields will be added if images are provided
        }]
        
    # Ensure confidence_distributions is present
    if "confidence_distributions" not in results or not results["confidence_distributions"]:
        print("No confidence_distributions found in results - creating default entry")
        # Create a default confidence distribution for tomatoes
        fruit_type = results.get("fruit_type", "").lower()
        if fruit_type == "tomato":
            results["confidence_distributions"] = [{
                "Unripe": 0.85,
                "Partially Ripe": 0.10,
                "Ripe": 0.04,
                "Overripe": 0.01,
                "estimated": True
            }]
        elif fruit_type == "banana":
            results["confidence_distributions"] = [{
                "Unripe": 0.75,
                "Ripe": 0.20,
                "Overripe": 0.05,
                "estimated": True
            }]
        else:
            # Generic confidence distribution
            results["confidence_distributions"] = [{
                "Unripe": 0.7,
                "Partially Ripe": 0.2,
                "Ripe": 0.1,
                "Overripe": 0.0,
                "estimated": True
            }]
    
    # Ensure num_fruits is set correctly
    if "num_fruits" not in results:
        results["num_fruits"] = len(results.get("fruits_data", []))
    
    return results

def crop_bounding_box(image, bbox):
        """
        Crop a bounding box from an image
        
        Args:
            image: PIL Image or numpy array
            bbox: Dictionary with x, y, width, height (center coordinates format)
            
        Returns:
            Cropped PIL Image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert center coordinates to top-left and bottom-right
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)
        
        # Calculate bounding box coordinates
        x1 = max(0, int(x - width / 2))
        y1 = max(0, int(y - height / 2))
        x2 = min(img_array.shape[1], int(x + width / 2))
        y2 = min(img_array.shape[0], int(y + height / 2))
        
        # Crop the image
        cropped = img_array[y1:y2, x1:x2]
        
        # Convert back to PIL Image
        return Image.fromarray(cropped)
    
class FruitRipenessSystem:
    def __init__(self, seg_model_path, classifier_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Enhanced Model (from Hugging Face) ---
        from models.segmentation_model import UNetResNet50
        self.seg_model = UNetResNet50(n_classes=1).to(self.device)
        if os.path.exists(seg_model_path):
            self.seg_model.load_state_dict(torch.load(seg_model_path, map_location=self.device))
            print(f"✅ Loaded enhanced segmentation model from HF Hub: {os.path.basename(seg_model_path)}")
        else:
            raise FileNotFoundError(f"Segmentation model not found at {seg_model_path}")
        
        from models.enhanced_model import extend_unet_resnet50_class
        extend_unet_resnet50_class()
        self.seg_model.activations = {}
        self.seg_model.register_hooks()
        
        from models.baseline_unet import load_pretrained_unet, register_hooks
        try:
            self.baseline_model = load_pretrained_unet(self.device)
            self.baseline_model = register_hooks(self.baseline_model)
            self.baseline_model.eval()
        except Exception as e:
            print(f"⚠️ Error setting up pretrained U-Net: {str(e)}")
            from models.baseline_model import BasicUNet
            self.baseline_model = BasicUNet(n_classes=1).to(self.device)
            print("⚠️ Using randomly initialized baseline U-Net (comparison will be less meaningful)")
        
        self.class_names = ["banana", "mango", "pineapple", "strawberry", "tomato"]
        self.classifier_model = FruitClassifier(num_classes=len(self.class_names)).to(self.device)
        
        if os.path.exists(classifier_model_path):
            checkpoint = torch.load(classifier_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.classifier_model.load_state_dict(checkpoint)
            print(f"✅ Loaded classifier model from HF Hub: {os.path.basename(classifier_model_path)}")
        else:
            print("⚠️ Using randomly initialized classifier weights")
        
        self.custom_model_handler = CustomModelInference(self.device)
        
        self.seg_model.eval()
        self.classifier_model.eval()
        
        # Define key layers for comparison
        self.key_comparison_layers = {
            "Encoder 1": ("enc1_conv1", "encoder1_conv1"),
            "Encoder 2": ("enc2_conv1", "encoder2_conv1"),
            "Encoder 3": ("enc3_conv1", "encoder3_conv1"),
            "Encoder 4": ("enc4_conv1", "encoder4_conv1"),
            "Bottleneck": ("bottleneck", "center"),
            "Decoder 4": ("dec4_conv1", "decoder5_conv1"),
            "Decoder 3": ("dec3_conv1", "decoder4_conv1"),
            "Decoder 2": ("dec2_conv1", "decoder3_conv1"),
            "Decoder 1": ("dec1_conv1", "decoder2_conv1"),
        }
        
        self.fruit_to_model = {
            "tomato": "tomates_4_classe/1",
            "pineapple": "pineapple-maturity-project-app/1",
            "banana": "aoop/3",
            "strawberry": "strawberry-ml-detection-02/1",
            "mango": "mango-detection-goiq9/1",
        }
        
        self.classification_models = {
            "banana": {"type": "roboflow", "model_id": "single-label-classification-zf1sy/1"},
            "mango": {"type": "custom", "model_key": "mango"},
            "tomato": {"type": "custom", "model_key": "tomato"},
            "strawberry": {"type": "custom", "model_key": "strawberry"},
            "pineapple": {"type": "custom", "model_key": "pineapple"}
        }
        
        self.supported_fruits = list(self.fruit_to_model.keys())
        os.makedirs('results', exist_ok=True)

        self.roboflow_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="UNykbkEetYICFkzzjcqP",
        )
        
        self.classification_config = InferenceConfiguration(
            confidence_threshold=0.01,  # Very low threshold to get all predictions
            max_detections=100  # Increase max detections
        )
        
        self.seg_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.classifier_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print("==== HOOKS REGISTRATION CHECK ====")
        print(f"SPEAR-UNet has activations attribute: {hasattr(self.seg_model, 'activations')}")
        print(f"SPEAR-UNet has register_hooks method: {hasattr(self.seg_model, 'register_hooks')}")
        print(f"Baseline U-Net has activations attribute: {hasattr(self.baseline_model, 'activations')}")
    
    def segment_fruit_with_metrics(self, image_path_or_file, refine_segmentation=True, refinement_method="all"):
        """
        Segment fruit and collect comparative metrics between baseline and enhanced models
        """
        print(f"Segmenting fruit with metrics...")
        
        # Open and preprocess the image
        if isinstance(image_path_or_file, str):
            img = Image.open(image_path_or_file).convert('RGB')
            image_path = image_path_or_file
        else:
            img = Image.open(image_path_or_file).convert('RGB')
            timestamp = int(time.time())
            image_path = f"results/uploaded_image_{timestamp}.png"
            img.save(image_path)
        
        original_size = img.size
        
        # Apply the same transform to input for both models
        img_tensor = self.seg_transform(img).unsqueeze(0).to(self.device)
        
        # Import the adapter function for handling channel differences
        from models.baseline_unet import adapt_input_channels
        
        # Clear any existing activations
        if hasattr(self.seg_model, 'activations'):
            self.seg_model.activations.clear()
        if hasattr(self.baseline_model, 'activations'):
            self.baseline_model.activations.clear()
        
        print("==== PRE-FORWARD ACTIVATIONS CHECK ====")
        print(f"SPEAR-UNet activations before forward: {list(self.seg_model.activations.keys())}")
        print(f"Baseline U-Net activations before forward: {list(self.baseline_model.activations.keys())}")
        
        # Track inference time for both models
        start_baseline = time.time()
        with torch.no_grad():
            # Adapt input channels for the baseline model
            baseline_input = adapt_input_channels(img_tensor, target_channels=3)
            baseline_output = self.baseline_model(baseline_input)
            baseline_mask = torch.sigmoid(baseline_output) > 0.5
            baseline_mask = baseline_mask.squeeze().cpu().numpy().astype(np.uint8)
        baseline_time = time.time() - start_baseline
        
        start_enhanced = time.time()
        with torch.no_grad():
            enhanced_output = self.seg_model(img_tensor)
            enhanced_mask = torch.sigmoid(enhanced_output) > 0.5
            enhanced_mask = enhanced_mask.squeeze().cpu().numpy().astype(np.uint8)
        enhanced_time = time.time() - start_enhanced
        
        print("==== POST-FORWARD ACTIVATIONS CHECK ====")
        print(f"SPEAR-UNet activations after forward: {list(self.seg_model.activations.keys())}")
        print(f"Baseline U-Net activations after forward: {list(self.baseline_model.activations.keys())}")
        
        # Resize masks back to original image size
        baseline_mask = cv2.resize(baseline_mask, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)
        enhanced_mask = cv2.resize(enhanced_mask, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)
        
        # Process enhanced mask with refinement if requested
        if refine_segmentation:
            print(f"Refining segmentation mask with method: {refinement_method}")
            refined_mask = refine_mask(enhanced_mask, refinement_method=refinement_method)
            
            # Calculate quality metrics for comparison
            original_metrics = get_mask_quality_metrics(enhanced_mask)
            refined_metrics = get_mask_quality_metrics(refined_mask)
            
            print(f"Original mask metrics: {original_metrics}")
            print(f"Refined mask metrics: {refined_metrics}")
            
            # Use the refined mask
            enhanced_mask = refined_mask
        
        # Apply masks to original image
        baseline_img = apply_mask_to_image(img, baseline_mask)
        enhanced_img = apply_mask_to_image(img, enhanced_mask)
        
        # Save the enhanced segmented image
        timestamp = int(time.time())
        segmented_path = f"results/segmented_{timestamp}_{os.path.basename(image_path)}"
        enhanced_img.save(segmented_path)
        
        # Generate comparison metrics
        from visualization_metrics import (
            compare_model_metrics, generate_comparison_visualization,
            visualize_regularization_impact_comparison, feature_map_visualization
        )
        
        comparison_metrics = compare_model_metrics(
            self.baseline_model, self.seg_model, self.key_comparison_layers
        )
        
        visualizations = {}
        
        try:
            visualizations = generate_comparison_visualization(
                self.baseline_model, self.seg_model, self.key_comparison_layers
            )
            print(f"Generated visualizations keys: {list(visualizations.keys())}")
        except Exception as e:
            import traceback
            print(f"⚠️ Error generating visualizations: {str(e)}")
            print(traceback.format_exc())
            visualizations = {}  # Ensure we have an empty dict instead of None
        
        # Try to generate regularization comparison, but handle failure gracefully
        reg_comparison_visualization = None
        try:
            reg_comparison_visualization = visualize_regularization_impact_comparison(self.seg_model)
        except Exception as e:
            print(f"Warning: Could not create regularization comparison visualization: {e}")
        
        # Create feature map visualizations if available
        feature_visualizations = {}
        
        # Safely check for activations in baseline model
        if hasattr(self.baseline_model, 'activations') and self.baseline_model.activations:
            if "enc3_conv1" in self.baseline_model.activations:
                try:
                    feature_visualizations["Baseline Encoder"] = feature_map_visualization(
                        self.baseline_model.activations.get("enc3_conv1"), "Baseline Encoder Features"
                    )
                except Exception as e:
                    print(f"Warning: Could not visualize baseline encoder features: {e}")
        
        # Safely check for activations in SPEAR-UNet model
        if hasattr(self.seg_model, 'activations') and self.seg_model.activations:
            if "encoder3_conv1" in self.seg_model.activations:
                try:
                    feature_visualizations["SPEAR-UNet Encoder"] = feature_map_visualization(
                        self.seg_model.activations.get("encoder3_conv1"), "SPEAR-UNet Encoder Features"
                    )
                except Exception as e:
                    print(f"Warning: Could not visualize SPEAR-UNet encoder features: {e}")
            
            # Add SFP visualization if available
            if "sfp3_out" in self.seg_model.activations:
                try:
                    feature_visualizations["SFP Output"] = feature_map_visualization(
                        self.seg_model.activations["sfp3_out"], "Stochastic Feature Pyramid Output"
                    )
                except Exception as e:
                    print(f"Warning: Could not visualize SFP features: {e}")
            
            # Add dynamic regularization visualization if available
            if "reg3_out" in self.seg_model.activations:
                try:
                    feature_visualizations["Dynamic Reg Output"] = feature_map_visualization(
                        self.seg_model.activations["reg3_out"], "Dynamic Regularization Output"
                    )
                except Exception as e:
                    print(f"Warning: Could not visualize regularization features: {e}")
        
        # Calculate improved IoU between masks
        intersection = np.logical_and(baseline_mask, enhanced_mask).sum()
        union = np.logical_or(baseline_mask, enhanced_mask).sum()
        iou = intersection / union if union > 0 else 0
        
        # Calculate mask quality improvements
        baseline_coverage = baseline_mask.sum() / (baseline_mask.shape[0] * baseline_mask.shape[1])
        enhanced_coverage = enhanced_mask.sum() / (enhanced_mask.shape[0] * enhanced_mask.shape[1])
        
        # Calculate boundary complexity
        baseline_contours, _ = cv2.findContours(baseline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enhanced_contours, _ = cv2.findContours(enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        baseline_complexity = 0
        enhanced_complexity = 0
        
        if baseline_contours:
            baseline_perimeter = sum(cv2.arcLength(cnt, True) for cnt in baseline_contours)
            baseline_area = sum(cv2.contourArea(cnt) for cnt in baseline_contours)
            baseline_complexity = baseline_perimeter**2 / (4 * np.pi * baseline_area) if baseline_area > 0 else 0
            
        if enhanced_contours:
            enhanced_perimeter = sum(cv2.arcLength(cnt, True) for cnt in enhanced_contours)
            enhanced_area = sum(cv2.contourArea(cnt) for cnt in enhanced_contours)
            enhanced_complexity = enhanced_perimeter**2 / (4 * np.pi * enhanced_area) if enhanced_area > 0 else 0
        
        # Create the result dictionary for debugging
        result_dict = {
            "original_image": img,
            "segmented_image": enhanced_img,
            "mask": enhanced_mask,
            "segmented_image_path": segmented_path,
            "comparison_metrics": {
                "baseline_time": baseline_time,
                "enhanced_time": enhanced_time,
                "speedup": baseline_time / enhanced_time if enhanced_time > 0 else 0,
                "baseline_mask": baseline_mask,
                "baseline_segmented_image": baseline_img,
                "iou": iou,
                "baseline_coverage": baseline_coverage,
                "enhanced_coverage": enhanced_coverage,
                "baseline_complexity": baseline_complexity,
                "enhanced_complexity": enhanced_complexity,
                "layer_metrics": comparison_metrics
            },
            "visualizations": visualizations,
            "feature_maps": feature_visualizations,
            "regularization_comparison_viz": reg_comparison_visualization,
            "mask_metrics": get_mask_quality_metrics(enhanced_mask),
            "refine_segmentation": refine_segmentation,
            "refinement_method": refinement_method
        }
        
        print(f"Visualization keys in result_dict: {list(result_dict['visualizations'].keys())}")
        
        # Return all results with metrics
        return {
            "original_image": img,
            "segmented_image": enhanced_img,
            "mask": enhanced_mask,
            "segmented_image_path": segmented_path,
            "comparison_metrics": {
                "baseline_time": baseline_time,
                "enhanced_time": enhanced_time,
                "speedup": baseline_time / enhanced_time if enhanced_time > 0 else 0,
                "baseline_mask": baseline_mask,
                "baseline_segmented_image": baseline_img,
                "iou": iou,
                "baseline_coverage": baseline_coverage,
                "enhanced_coverage": enhanced_coverage,
                "baseline_complexity": baseline_complexity,
                "enhanced_complexity": enhanced_complexity,
                "layer_metrics": comparison_metrics
            },
            "visualizations": visualizations,
            "feature_maps": feature_visualizations,
            "regularization_comparison_viz": reg_comparison_visualization,
            "mask_metrics": get_mask_quality_metrics(enhanced_mask),
            "refine_segmentation": refine_segmentation,
            "refinement_method": refinement_method
        }
    
    def classify_fruit(self, image, confidence_threshold=0.65):
        """
        Use the fruit classifier to determine the type of fruit
        
        Args:
            image: PIL Image to classify
            confidence_threshold: Minimum confidence required to classify fruit (default: 0.7)
            
        Returns:
            The fruit type and confidence score, or "outside_scope" if confidence is too low
        """
        print(f"Classifying fruit...")
        
        # Correct class names the model was trained on
        correct_class_names = ["banana", "mango", "pineapple", "strawberry", "tomato"]
        
        img_tensor = self.classifier_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            
            # Use the correct class name mapping
            if confidence >= confidence_threshold:
                fruit_type = correct_class_names[predicted_idx]
                print(f"Classified as: {fruit_type} (confidence: {confidence:.2f})")
            else:
                fruit_type = "outside_scope"
                print(f"Confidence too low ({confidence:.2f}), fruit likely outside scope")
        
        # Create all_probs using the correct class names
        all_probs = {
            correct_class_names[i]: probabilities[i].item()
            for i in range(len(correct_class_names))
        }
        
        return fruit_type, confidence, all_probs
    
    def detect_ripeness_with_mask(self, original_image_path, segmented_image, mask, fruit_type):
        """
        Use the Roboflow client to detect ripeness, sending the segmented image
        instead of the original image
        
        Args:
            original_image_path: Path to original image (kept for reference)
            segmented_image: PIL Image of the segmented fruit
            mask: Binary mask as numpy array
            fruit_type: Type of fruit to detect ripeness for
            
        Returns:
            Ripeness detection results
        """
        if fruit_type not in self.fruit_to_model:
            raise ValueError(f"No ripeness model available for {fruit_type}")
        
        model_id = self.fruit_to_model[fruit_type]
        print(f"Using Roboflow model: {model_id} for {fruit_type}")
        
        temp_path = f"results/temp_segmented_{int(time.time())}.png"
        segmented_image.save(temp_path)
        
        try:
            # Use segmented image for ripeness detection
            print(f"DEBUG - Calling Roboflow with SEGMENTED image: {temp_path}")
            print(f"DEBUG - Using model ID: {model_id}")
            
            # Send the segmented image to Roboflow
            results = self.roboflow_client.infer(temp_path, model_id=model_id)
            print(f"Raw Roboflow results: {results}")
            
            return results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error using Roboflow model: {str(e)}")
            return {"predictions": [], "error": str(e)}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            pass
    
    def process_image_with_visualization(self, image_path_or_file, fruit_type, refine_segmentation=True, refinement_method="all", angle_name=None):
        """
        Process an image and create visualizations with optional mask refinement
        """
        try:
            if angle_name:
                print(f"Processing {angle_name} view...")
            
            # Step 1: Segment the fruit with metrics and refinement option
            segmentation_results = self.segment_fruit_with_metrics(
                image_path_or_file,
                refine_segmentation=refine_segmentation,
                refinement_method=refinement_method
            )
            
            print("\n=== PROCESS_IMAGE_WITH_VISUALIZATION DEBUG ===")
            print(f"Segmentation results keys: {list(segmentation_results.keys())}")
            if "visualizations" in segmentation_results:
                print(f"Layer visualizations from segmentation: {list(segmentation_results['visualizations'].keys())}")
            else:
                print("No 'visualizations' key in segmentation_results")

            print(f"Result keys before adding visualizations: {list(result.keys())}")
            original_img = segmentation_results["original_image"]
            segmented_img = segmentation_results["segmented_image"]
            mask = segmentation_results["mask"]
            segmented_path = segmentation_results["segmented_image_path"]
            
            # Get the original image path
            if isinstance(image_path_or_file, str):
                original_path = image_path_or_file
            else:
                original_path = segmented_path.replace("segmented_", "uploaded_image_")
            
            # Step 2: Use the user-selected fruit type and detect ripeness
            fruit_type_normalized = fruit_type.lower().strip()
            
            ripeness_result = self.detect_ripeness_with_mask(
                original_path, segmented_img, mask, fruit_type_normalized
            )
            
            # Process results
            predictions = ripeness_result.get("predictions", [])
            if predictions:
                # Format the results
                formatted_results = self.format_ripeness_results(fruit_type, predictions)
                
                # Create base result dictionary with essential fields
                result = {
                    "fruit_type": fruit_type,
                    "classification_confidence": 1.0,  # Using 1.0 as confidence since user-selected
                    "ripeness_predictions": formatted_results,
                    "segmented_image": segmented_img,
                    "original_image": original_img,
                    "mask": mask,
                    "segmented_image_path": segmented_path,
                    "original_image_path": original_path,
                    "raw_results": ripeness_result,
                    "mask_metrics": get_mask_quality_metrics(mask),
                    "refine_segmentation": refine_segmentation,
                    "refinement_method": refinement_method
                }
                
                # Add angle name if provided
                if angle_name:
                    result["angle_name"] = angle_name
                    
                print("\n=== RESULT CREATION DEBUG ===")
                print(f"Result keys before adding segmentation data: {list(result.keys())}")
                
                # CRITICAL FIX: Initialize visualizations with layer visualizations from segmentation
                if "visualizations" in segmentation_results:
                    # Create the visualizations dict in results
                    result["visualizations"] = {}
                    
                    # Copy all layer visualizations
                    for vis_key, vis_value in segmentation_results["visualizations"].items():
                        result["visualizations"][vis_key] = vis_value
                        
                    print(f"Copied layer visualizations to result: {list(result['visualizations'].keys())}")
                else:
                    print("WARNING: No visualizations found in segmentation_results!")
                
                # Add comparison metrics if available
                if "comparison_metrics" in segmentation_results:
                    result["comparison_metrics"] = segmentation_results["comparison_metrics"]
                    print("Added comparison_metrics to result")
                
                # Add feature maps if available
                if "feature_maps" in segmentation_results:
                    result["feature_maps"] = segmentation_results["feature_maps"]
                    print(f"Added feature_maps to result: {list(segmentation_results['feature_maps'].keys())}")
                
                # Add regularization comparison visualization if available
                if "regularization_comparison_viz" in segmentation_results:
                    result["regularization_comparison_viz"] = segmentation_results["regularization_comparison_viz"]
                    print("Added regularization_comparison_viz to result")
                
                # Before calling visualize_results, check what visualizations we have
                if "visualizations" in result:
                    print(f"Visualizations before ripeness visualizations: {list(result['visualizations'].keys())}")
                else:
                    print("No visualizations in result before ripeness visualizations")
                    # Ensure visualizations dict exists
                    result["visualizations"] = {}
                    
                # Create ripeness visualizations
                try:
                    vis_results = self.visualize_results(result)
                    print(f"Ripeness visualization keys from visualize_results: {list(vis_results.keys())}")
                    
                    # Store original layer visualization keys so we can verify they're preserved
                    original_vis_keys = list(result["visualizations"].keys())
                    print(f"Original visualization keys before adding ripeness vis: {original_vis_keys}")
                    
                    # Add new visualizations, preserving existing ones
                    for vis_key, vis_value in vis_results.items():
                        result["visualizations"][vis_key] = vis_value
                    
                    # Verify layer visualizations are still present
                    layer_keys = ["Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4", 
                                "Bottleneck", "Decoder 4", "Decoder 3", "Decoder 2", "Decoder 1"]
                    preserved_layer_keys = [k for k in result["visualizations"].keys() if k in layer_keys]
                    
                    print(f"Final visualization keys: {list(result['visualizations'].keys())}")
                    print(f"Layer visualizations preserved: {preserved_layer_keys}")
                    
                    # If layer visualizations were lost, add them back
                    if not preserved_layer_keys and "visualizations" in segmentation_results:
                        print("WARNING: Layer visualizations were lost! Adding them back...")
                        for vis_key, vis_value in segmentation_results["visualizations"].items():
                            if vis_key in layer_keys:
                                result["visualizations"][vis_key] = vis_value
                        
                        print(f"Restored visualization keys: {list(result['visualizations'].keys())}")
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error in visualize_results: {str(e)}")
                    result["visualization_error"] = str(e)
                
                # Final check of results
                print(f"=== FINAL RESULT CHECK ===")
                print(f"Final result keys: {list(result.keys())}")
                if "visualizations" in result:
                    all_vis_keys = list(result["visualizations"].keys())
                    print(f"Final visualization keys: {all_vis_keys}")
                    
                    # Check specifically for layer visualizations
                    layer_keys = ["Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4", 
                                "Bottleneck", "Decoder 4", "Decoder 3", "Decoder 2", "Decoder 1"]
                    final_layer_keys = [k for k in all_vis_keys if k in layer_keys]
                    print(f"Final layer visualization keys: {final_layer_keys}")
                
                return result
                
            else:
                # Handle case with no predictions - same structure as above
                result = {
                    "fruit_type": fruit_type,
                    "classification_confidence": 1.0,
                    "error": "No ripeness predictions found",
                    "segmented_image": segmented_img,
                    "original_image": original_img,
                    "mask": mask,
                    "segmented_image_path": segmented_path,
                    "original_image_path": original_path,
                    "mask_metrics": get_mask_quality_metrics(mask),
                    "refine_segmentation": refine_segmentation,
                    "refinement_method": refinement_method
                }
                
                # Add angle name if provided
                if angle_name:
                    result["angle_name"] = angle_name
                
                # CRITICAL FIX: Initialize visualizations with layer visualizations from segmentation
                if "visualizations" in segmentation_results:
                    # Create the visualizations dict in results
                    result["visualizations"] = {}
                    
                    # Copy all layer visualizations
                    for vis_key, vis_value in segmentation_results["visualizations"].items():
                        result["visualizations"][vis_key] = vis_value
                        
                    print(f"Copied layer visualizations to result (no predictions case): {list(result['visualizations'].keys())}")
                
                # Add comparison metrics if available
                if "comparison_metrics" in segmentation_results:
                    result["comparison_metrics"] = segmentation_results["comparison_metrics"]
                
                # Add visualizations if available
                if "visualizations" in segmentation_results:
                    if "visualizations" not in result:
                        result["visualizations"] = {}
                    
                    # Copy all visualizations
                    for vis_key, vis_value in segmentation_results["visualizations"].items():
                        result["visualizations"][vis_key] = vis_value
                
                # Add feature maps if available
                if "feature_maps" in segmentation_results:
                    result["feature_maps"] = segmentation_results["feature_maps"]
                
                # Add regularization comparison visualization if available
                if "regularization_comparison_viz" in segmentation_results:
                    result["regularization_comparison_viz"] = segmentation_results["regularization_comparison_viz"]
                
                # Create ripeness visualizations anyway
                try:
                    vis_results = self.visualize_results(result)
                    
                    # Store existing visualization keys
                    existing_keys = list(result.get("visualizations", {}).keys())
                    
                    # Ensure visualizations dict exists
                    if "visualizations" not in result:
                        result["visualizations"] = {}
                    
                    # Add new visualizations, preserving existing ones
                    for vis_key, vis_value in vis_results.items():
                        result["visualizations"][vis_key] = vis_value
                    
                    # Check if layer visualizations were preserved
                    layer_keys = ["Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4", 
                                "Bottleneck", "Decoder 4", "Decoder 3", "Decoder 2", "Decoder 1"]
                    preserved_layer_keys = [k for k in existing_keys if k in layer_keys and k in result["visualizations"]]
                    
                    # If layer visualizations were lost, add them back
                    if any(k in layer_keys for k in existing_keys) and not preserved_layer_keys:
                        print("WARNING: Layer visualizations were lost! Adding them back...")
                        for vis_key, vis_value in segmentation_results["visualizations"].items():
                            if vis_key in layer_keys:
                                result["visualizations"][vis_key] = vis_value
                    
                except Exception as e:
                    print(f"Error in visualize_results (no predictions): {str(e)}")
                    result["visualization_error"] = str(e)
                
                return result
                        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing image: {str(e)}")
            result = {"error": str(e)}
            if angle_name:
                result["angle_name"] = angle_name
            return result
    
    def format_ripeness_results(self, fruit_type, predictions):
        """
        Format the prediction results based on the fruit type
        Different fruits have different ripeness classes
        
        Args:
            fruit_type: Type of fruit
            predictions: Predictions from Roboflow API
            
        Returns:
            Formatted list of ripeness predictions
        """
        formatted = []
        
        fruit_type_normalized = fruit_type.lower()
        
        print(f"Formatting ripeness results for {fruit_type_normalized}")
        print(f"Raw predictions: {predictions}")
        
        for pred in predictions:
            class_name = pred.get("class", "unknown")
            confidence = pred.get("confidence", 0)
            
            # Skip "flower" class for all fruit types
            if class_name.lower() == "flower" or class_name.lower() == "null":
                print(f"Skipping '{class_name}' class for {fruit_type_normalized}")
                continue
            
            if fruit_type_normalized == "tomato":
                ripeness_label = class_name.replace("_", " ").title()
            elif fruit_type_normalized == "pineapple":
                ripeness_label = class_name.replace("_", " ").title()
            elif fruit_type_normalized == "banana":
                # Custom mapping for banana ripeness classes
                if class_name.lower() in ["freshunripe"]:
                    ripeness_label = "Unripe"
                elif class_name.lower() in ["freshripe"]:
                    ripeness_label = "Ripe"
                elif class_name.lower() in ["rotten", "overripe"]:
                    ripeness_label = "Overripe"
                else:
                    # Just capitalize the existing class name if not in our mapping
                    ripeness_label = class_name.capitalize()
            elif fruit_type_normalized == "strawberry":
                # Custom mapping for strawberry ripeness classes
                if class_name.lower() == "strawberryripe" or class_name.lower() == "ripe":
                    ripeness_label = "Ripe"
                elif class_name.lower() == "strawberryunripe" or class_name.lower() == "unripe":
                    ripeness_label = "Unripe"
                elif class_name.lower() == "strawberryrotten" or class_name.lower() == "rotten":
                    ripeness_label = "Overripe"
                else:
                    # Just capitalize the existing class name if not in our mapping
                    ripeness_label = class_name.capitalize()
            elif fruit_type_normalized == "mango":
                # Custom mapping for mango ripeness classes
                class_name_lower = class_name.lower()
                if class_name_lower in ["early_ripe-21-40-", "unripe-1-20-"]:
                    ripeness_label = "Unripe"
                elif class_name_lower == "partially_ripe-41-60-":
                    ripeness_label = "Underripe"
                elif class_name_lower == "ripe-61-80-":
                    ripeness_label = "Ripe"
                elif class_name_lower == "over_ripe-81-100-":
                    ripeness_label = "Overripe"
                else:
                    # Just capitalize the existing class name if not in our mapping
                    ripeness_label = class_name.capitalize()
            else:
                ripeness_label = class_name
                    
            formatted.append({
                "ripeness": ripeness_label,
                "confidence": confidence
            })
                
        return sorted(formatted, key=lambda x: x["confidence"], reverse=True)
    
    def visualize_results(self, results, save_dir="results"):
        """
        Create visualizations for the ripeness detection results
        
        Args:
            results: Dictionary containing processing results
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary of visualization paths
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time())
        
        # Get images from results
        original_img = results.get("original_image")
        segmented_img = results.get("segmented_image")
        
        if not original_img or not segmented_img:
            return {"error": "Missing images in results"}
        
        # Create enhanced visualization with bounding boxes
        box_vis_path = f"{save_dir}/boxes_{timestamp}.png"
        bbox_visualization = create_enhanced_visualization(
            results, original_img, segmented_img, box_vis_path
        )
        
        # Create side-by-side comparison
        comparison_path = f"{save_dir}/comparison_{timestamp}.png"
        comparison_visualization = create_side_by_side_comparison(
            results, original_img, segmented_img, comparison_path
        )
        
        # Create combined visualization
        combined_path = f"{save_dir}/combined_{timestamp}.png"
        combined_visualization = create_combined_visualization(
            results, original_img, segmented_img, combined_path
        )
        
        return {
            "bounding_box_visualization": bbox_visualization,
            "comparison_visualization": comparison_visualization,
            "combined_visualization": combined_visualization
        }

    def process_image_without_segmentation(self, image_path_or_file, fruit_type, angle_name=None):
        """
        Process an image without segmentation, sending the original image directly to Roboflow
        
        Args:
            image_path_or_file: Path to image file or file-like object
            fruit_type: Type of fruit to process (user-selected)
            angle_name: Optional name for the angle (for patch-based analysis)
            
        Returns:
            Dictionary with processing results
        """
        try:
            print(f"\nProcessing image without segmentation...")
            if angle_name:
                print(f"Processing {angle_name} view without segmentation...")
            
            # Open the image
            if isinstance(image_path_or_file, str):
                img = Image.open(image_path_or_file).convert('RGB')
                original_path = image_path_or_file
            else:
                img = Image.open(image_path_or_file).convert('RGB')
                # Create a temporary filename for saving
                timestamp = int(time.time())
                original_path = f"results/uploaded_image_{timestamp}.png"
                img.save(original_path)
            
            # Use the user-selected fruit type
            fruit_type_normalized = fruit_type.lower().strip()
            
            # Get model ID for the selected fruit type
            model_id = self.fruit_to_model.get(fruit_type_normalized)
            
            if not model_id:
                raise ValueError(f"No ripeness model available for {fruit_type}")
                
            # Debug information
            print(f"DEBUG - Fruit type: {fruit_type_normalized}")
            print(f"DEBUG - Model ID from dictionary: {model_id}")
            
            # Create a dummy mask (just for API compatibility)
            dummy_mask = np.zeros((img.height, img.width), dtype=np.uint8)
            
            # Detect ripeness using original image
            ripeness_result = self.detect_ripeness_without_mask(
                original_path, fruit_type_normalized, model_id
            )
            
            # Process results
            predictions = ripeness_result.get("predictions", [])
            print(f"Raw predictions for visualization: {predictions}")
            
            if predictions:
                # Format the results
                formatted_results = self.format_ripeness_results(fruit_type, predictions)
                
                result = {
                    "fruit_type": fruit_type,
                    "classification_confidence": 1.0,  # Using 1.0 as confidence since user-selected
                    "ripeness_predictions": formatted_results,
                    "original_image": img,
                    "original_image_path": original_path,
                    "raw_results": ripeness_result,

                    "segmented_image": img,
                    "mask": dummy_mask,
                    "mask_metrics": {"num_objects": 0, "coverage_ratio": 0.0, "boundary_complexity": 0.0}
                }
                
                # Add angle name if provided
                if angle_name:
                    result["angle_name"] = angle_name
                
                # If successful, add visualizations
                try:
                    vis_results = self.visualize_results(result)
                    result.update({"visualizations": vis_results})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    result["visualization_error"] = str(e)
                
                return result
            else:
                result = {
                    "fruit_type": fruit_type,
                    "classification_confidence": 1.0,  # Using 1.0 as confidence since user-selected
                    "error": "No ripeness predictions found",
                    "original_image": img,
                    "original_image_path": original_path,
                    # Important: Use the same image object for both original and segmented
                    "segmented_image": img,
                    "mask": dummy_mask,
                    "mask_metrics": {"num_objects": 0, "coverage_ratio": 0.0, "boundary_complexity": 0.0}
                }
                
                # Add angle name if provided
                if angle_name:
                    result["angle_name"] = angle_name
                
                # Add visualizations even if no predictions are found
                try:
                    vis_results = self.visualize_results(result)
                    result.update({"visualizations": vis_results})
                except Exception as e:
                    result["visualization_error"] = str(e)
                
                return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing image without segmentation: {str(e)}")
            result = {"error": str(e)}
            if angle_name:
                result["angle_name"] = angle_name
            return result

    def detect_ripeness_without_mask(self, original_image_path, fruit_type, model_id=None):
        """
        Use the Roboflow client to detect ripeness, sending the original image
        directly without segmentation
        
        Args:
            original_image_path: Path to original image
            fruit_type: Type of fruit to detect ripeness for
            model_id: Optional override for the model ID
            
        Returns:
            Ripeness detection results
        """
        try:
            # Determine which model to use
            if model_id is None and fruit_type in self.fruit_to_model:
                model_id = self.fruit_to_model[fruit_type]
            elif model_id is None:
                raise ValueError(f"No ripeness model available for {fruit_type}")
                
            # Add debug print to verify the model_id being used
            print(f"Using Roboflow model: {model_id} for {fruit_type} (without segmentation)")
            
            # Ensure model_id is in the correct format
            if "/" not in model_id:
                raise ValueError(f"Invalid model ID format: {model_id}. Expected format: project_id/model_version_id")
            
            # Resize/compress image before sending to Roboflow
            processed_image_path = resize_and_compress_image(
                original_image_path,
                max_size=(800, 800),  # Adjust based on model requirements
                quality=85,
                max_file_size_kb=950  # Setting below 1MB to be safe
            )
            
            # Now use the processed image with Roboflow
            results = self.roboflow_client.infer(processed_image_path, model_id=model_id)
            
            # Clean up temporary file if it's different from original
            if processed_image_path != original_image_path and os.path.exists(processed_image_path):
                os.remove(processed_image_path)
                
            return results
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error using Roboflow model: {str(e)}")
            return {"predictions": [], "error": str(e)}
        
    def analyze_ripeness_enhanced(self, image_path_or_file, fruit_type, is_patch_based=False, use_segmentation=True):
        """
        Enhanced ripeness analysis flow with fruit type verification:
        1. Initial segmentation to remove background (if use_segmentation is True)
        2. Verify fruit type using classifier
        3. For patch-based: skip detection and go straight to classification
        4. For non-patch-based: detect objects then classify each
        
        Args:
            image_path_or_file: Path to image or file-like object
            fruit_type: Type of fruit to analyze
            is_patch_based: Whether this is part of patch-based analysis
            use_segmentation: Whether to use segmentation or not
        """
        try:
            # Open and preprocess the image
            if isinstance(image_path_or_file, str):
                img = Image.open(image_path_or_file).convert('RGB')
                image_path = image_path_or_file
            else:
                img = Image.open(image_path_or_file).convert('RGB')
                timestamp = int(time.time())
                image_path = f"results/uploaded_image_{timestamp}.png"
                img.save(image_path)
            
            # Normalize fruit type
            fruit_type_normalized = fruit_type.lower().strip()
            
            # Initialize variables
            segmentation_results = None
            segmented_img = None
            mask = None
            
            if use_segmentation:
                print(f"Performing initial segmentation...")
                segmentation_results = self.segment_fruit_with_metrics(
                    image_path_or_file,
                    refine_segmentation=True,
                    refinement_method="all"
                )
                
                # Store segmentation results
                segmented_img = segmentation_results["segmented_image"]
                mask = segmentation_results["mask"]
            else:
                # Skip segmentation, use original image
                print(f"Skipping segmentation as requested...")
                segmented_img = img
                # Create a dummy mask (just for API compatibility)
                mask = np.zeros((img.height, img.width), dtype=np.uint8)
                    
                # Create minimal segmentation results for compatibility
                segmentation_results = {
                    "original_image": img,
                    "segmented_image": img,
                    "mask": mask,
                    "mask_metrics": {"num_objects": 0, "coverage_ratio": 0.0, "boundary_complexity": 0.0}
                }
            
            # Store original image
            original_img = segmentation_results["original_image"]
            
            # ----- ADDED: FRUIT TYPE VERIFICATION STEP -----
            # Use the classifier to verify fruit type
            detected_fruit_type, confidence, all_probs = self.classify_fruit(segmented_img)
            
            if detected_fruit_type.lower() != fruit_type_normalized:
                print(f"Warning: Detected fruit type '{detected_fruit_type}' doesn't match selected fruit type '{fruit_type_normalized}'")
                # Return warning result with both types and confidence scores
                return {
                    "warning": "fruit_type_mismatch",
                    "fruit_type_selected": fruit_type_normalized,
                    "fruit_type_detected": detected_fruit_type.lower(),
                    "detection_confidence": confidence,
                    "all_probabilities": all_probs,
                    "original_image": original_img,
                    "segmented_image": segmented_img,
                    "mask": mask,
                    "suggestion": f"The uploaded image appears to be a {detected_fruit_type.lower()}, not a {fruit_type_normalized}. Please confirm or select the correct fruit type."
                }
            
            if is_patch_based:
                print("Patch-based analysis - skipping object detection")
                img_width, img_height = original_img.size
                predictions = [{
                    "x": img_width / 2,
                    "y": img_height / 2,
                    "width": img_width,
                    "height": img_height,
                    "confidence": 1.0,
                    "class": fruit_type_normalized
                }]
                detection_results = {"predictions": predictions}
            else:
                # Regular flow with object detection
                temp_segmented_path = f"results/temp_segmented_{int(time.time())}.png"
                segmented_img.save(temp_segmented_path)
                
                # STEP 2: Use Roboflow to detect objects in the segmented image
                print(f"Detecting fruits using Roboflow model...")
                detection_model_id = self.fruit_to_model.get(fruit_type_normalized)
                if not detection_model_id:
                    raise ValueError(f"No detection model available for {fruit_type}")
                
                try:
                    # Get detection results
                    detection_results = self.roboflow_client.infer(
                        temp_segmented_path, 
                        model_id=detection_model_id
                    )
                    predictions = detection_results.get("predictions", [])
                    
                    # Clean up temporary file
                    if os.path.exists(temp_segmented_path):
                        os.remove(temp_segmented_path)
                        
                    print(f"Detected {len(predictions)} fruits in the image")
                    
                    # If no fruits detected, use fallback approach
                    if not predictions:
                        print("No fruits detected by detection model. Using fallback approach...")
                        if use_segmentation:
                            # Use connected components as fallback when segmentation is on
                            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
                            num_fruits = num_labels - 1
                            
                            if num_fruits >= 1:
                                # Create bounding boxes for each connected component
                                predictions = []
                                for label in range(1, num_labels):  # Skip background (0)
                                    component_mask = (labels == label).astype(np.uint8)
                                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if contours:
                                        x, y, w, h = cv2.boundingRect(contours[0])
                                        predictions.append({
                                            "x": x + w/2,
                                            "y": y + h/2,
                                            "width": w,
                                            "height": h,
                                            "confidence": 0.9,
                                            "class": fruit_type_normalized
                                        })
                        
                        # If still no fruits or segmentation was off, create a single bounding box
                        if not predictions:
                            img_width, img_height = original_img.size
                            predictions = [{
                                "x": img_width / 2,
                                "y": img_height / 2,
                                "width": img_width,
                                "height": img_height,
                                "confidence": 1.0,
                                "class": fruit_type_normalized
                            }]
                        
                        # Create synthetic detection_results
                        detection_results = {"predictions": predictions}
                except Exception as e:
                    print(f"Error using Roboflow detection model: {str(e)}")
                    # Fallback: use the whole image
                    img_width, img_height = original_img.size
                    predictions = [{
                        "x": img_width / 2,
                        "y": img_height / 2,
                        "width": img_width,
                        "height": img_height,
                        "confidence": 1.0,
                        "class": fruit_type_normalized
                    }]
                    detection_results = {"predictions": predictions}
            
            # STEP 3: Process each detected fruit (only one for patch-based)
            fruits_data = []
            classification_results = []
            confidence_distributions = []
            
            for i, pred in enumerate(predictions):
                # Get bounding box coordinates
                bbox = {
                    "x": pred.get("x", 0),
                    "y": pred.get("y", 0),
                    "width": pred.get("width", 0),
                    "height": pred.get("height", 0),
                    "confidence": pred.get("confidence", 0),
                    "class": pred.get("class", "unknown")
                }
                
                # Process this fruit
                print(f"Processing fruit #{i+1}...")
                result = self._process_single_fruit(
                    original_img, 
                    segmented_img,
                    bbox, 
                    fruit_type_normalized,
                    self.classification_models,
                    i
                )
                
                fruits_data.append(result["fruit_data"])
                classification_results.append(result["classification"])
                confidence_distributions.append(result["confidence_distribution"])
            
            # Combine all results
            final_result = {
                "fruit_type": fruit_type,
                "num_fruits": len(predictions),
                "original_image": original_img,
                "segmented_image": segmented_img,
                "fruits_data": fruits_data,
                "classification_results": classification_results,
                "confidence_distributions": confidence_distributions,
                "segmentation_results": segmentation_results,
                "mask": mask,
                "raw_results": detection_results,
                "is_patch_based": is_patch_based,  # Add flag for display functions
                "use_segmentation": use_segmentation,  # Add flag to indicate if segmentation was used
                "analysis_type": "enhanced_two_stage"  # Explicitly mark as two-stage analysis
            }
            
            # Create visualizations
            try:
                vis_results = self.visualize_results(final_result)
                final_result["visualizations"] = vis_results
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error creating visualizations: {str(e)}")
            
            if hasattr(self, 'debug_and_fix_enhanced_results'):
                final_result = self.debug_and_fix_enhanced_results(final_result)
            
            return final_result
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in enhanced ripeness analysis: {str(e)}")
            return {"error": str(e)}
    
    def _process_single_fruit(self, original_img, segmented_img, bbox, fruit_type, classification_models, fruit_index):
        """
        Process a single fruit through crop → segment → classify pipeline
        
        Args:
            original_img: Original PIL image
            segmented_img: Segmented PIL image
            bbox: Bounding box dictionary
            fruit_type: Normalized fruit type string
            classification_models: Dictionary of classification models
            fruit_index: Index of this fruit
            
        Returns:
            Dictionary with fruit data and classification results
        """
        try:
            # STEP 1: Crop the bounding box from both original and segmented images
            original_crop = crop_bounding_box(original_img, bbox)
            segmented_crop = crop_bounding_box(segmented_img, bbox)
            
            # Save crops for reference
            timestamp = int(time.time())
            original_crop_path = f"results/fruit{fruit_index}_original_{timestamp}.png"
            segmented_crop_path = f"results/fruit{fruit_index}_segmented_{timestamp}.png"
            original_crop.save(original_crop_path)
            segmented_crop.save(segmented_crop_path)
            
            # STEP 2: Segment the cropped image
            # Convert segmented crop to binary mask
            segmented_crop_array = np.array(segmented_crop)
            # If RGB image, convert to grayscale first
            if len(segmented_crop_array.shape) == 3 and segmented_crop_array.shape[2] == 3:
                segmented_crop_gray = cv2.cvtColor(segmented_crop_array, cv2.COLOR_RGB2GRAY)
            else:
                segmented_crop_gray = segmented_crop_array
                
            # Threshold to get binary mask
            _, crop_mask = cv2.threshold(segmented_crop_gray, 10, 255, cv2.THRESH_BINARY)
            
            # Apply mask to original crop to get clean fruit
            crop_mask_rgb = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2RGB)
            masked_crop = cv2.bitwise_and(np.array(original_crop), crop_mask_rgb)
            masked_crop_img = Image.fromarray(masked_crop)
            
            # Save masked crop
            masked_crop_path = f"results/fruit{fruit_index}_masked_{timestamp}.png"
            masked_crop_img.save(masked_crop_path)
            
            # STEP 3: Classification
            classification_result = {}
            confidence_distribution = {}
            
            # Check if classification model is available for this fruit type
            model_config = classification_models.get(fruit_type)
            
            if model_config:
                # Handle different model types
                if model_config["type"] == "custom":
                    # Use custom PyTorch model
                    confidence_distribution = self.custom_model_handler.classify_image(
                        masked_crop_img, 
                        model_config["model_key"]
                    )
                    
                    # Check if error occurred
                    if "error" in confidence_distribution:
                        print(f"Error in custom classification: {confidence_distribution['error']}")
                        # Try to use detection result as fallback
                        confidence_distribution = self._estimate_from_detection(bbox)
                    
                elif model_config["type"] == "roboflow":
                    # Use Roboflow classification API
                    model_id = model_config["model_id"]
                    
                    try:
                        # Resize image to appropriate size for classification
                        resized_crop = masked_crop_img.resize((224, 224))
                        resized_path = f"results/fruit{fruit_index}_resized_{timestamp}.png"
                        resized_crop.save(resized_path)
                        
                        # Use classification configuration specifically for classification models
                        with self.roboflow_client.use_configuration(self.classification_config):
                            classification_result = self.roboflow_client.infer(
                                resized_path, 
                                model_id=model_id
                            )
                        
                        print(f"Raw classification results: {classification_result}")
                        if "predictions" in classification_result:
                            confidence_distribution = {}
                            
                            if "raw_predictions" in classification_result:
                                all_predictions = classification_result["raw_predictions"]
                                for pred in all_predictions:
                                    class_name = pred.get("class", "unknown")
                                    confidence = pred.get("confidence", 0)
                                    
                                    ripeness = self._standardize_ripeness_label(fruit_type, class_name)
                                    if ripeness:
                                        confidence_distribution[ripeness] = confidence
                            else:
                                # Fall back to regular predictions
                                for pred in classification_result["predictions"]:
                                    class_name = pred.get("class", "unknown")
                                    confidence = pred.get("confidence", 0)
                                    
                                    ripeness = self._standardize_ripeness_label(fruit_type, class_name)
                                    if ripeness:
                                        confidence_distribution[ripeness] = confidence
                        
                        # After processing all predictions, consolidate categories and normalize confidences for bananas
                        if fruit_type == "banana" and confidence_distribution:
                            # Create new consolidated distribution with just the three categories
                            consolidated = {
                                "Unripe": 0.0,
                                "Ripe": 0.0,
                                "Overripe": 0.0
                            }
                            
                            # Sum up confidences for each group
                            for ripeness, confidence in confidence_distribution.items():
                                if ripeness == "Unripe":
                                    consolidated["Unripe"] += confidence
                                elif ripeness in ["Ripe", "half_ripe"]:
                                    consolidated["Ripe"] += confidence
                                elif ripeness in ["Overripe", "nearly_rotten"]:
                                    consolidated["Overripe"] += confidence
                            
                            # Normalize to ensure sum is 1.0
                            total = sum(consolidated.values())
                            if total > 0:
                                for key in consolidated:
                                    consolidated[key] /= total
                            
                            # Replace the original distribution
                            confidence_distribution = consolidated
                        
                        # Clean up
                        if os.path.exists(resized_path):
                            os.remove(resized_path)
                            
                    except Exception as e:
                        print(f"Error in Roboflow classification: {str(e)}")
                        # Try to use detection result as fallback
                        confidence_distribution = self._estimate_from_detection(bbox)
                        confidence_distribution["estimated"] = True
                else:
                    print(f"Unknown model type: {model_config['type']}")
                    confidence_distribution = self._estimate_from_detection(bbox)
                    confidence_distribution["estimated"] = True
            else:
                # No classification model available, use detection result
                confidence_distribution = self._estimate_from_detection(bbox)
                confidence_distribution["estimated"] = True
            
            # Combine results
            result = {
                "fruit_data": {
                    "index": fruit_index,
                    "bbox": bbox,
                    "original_crop_path": original_crop_path,
                    "segmented_crop_path": segmented_crop_path,
                    "masked_crop_path": masked_crop_path
                },
                "classification": classification_result,
                "confidence_distribution": confidence_distribution
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing single fruit: {str(e)}")
            return {
                "fruit_data": {
                    "index": fruit_index,
                    "bbox": bbox,
                    "error": str(e)
                },
                "classification": {},
                "confidence_distribution": {"error": str(e)}
            }

    def _estimate_from_detection(self, bbox):
        """
        Estimate confidence distribution based on detection result
        
        Args:
            bbox: Bounding box dictionary with class and confidence
            
        Returns:
            Estimated confidence distribution
        """
        ripeness = self._standardize_ripeness_label("", bbox.get("class", "unknown"))
        confidence = bbox.get("confidence", 0)
        
        # Create estimated distribution
        if ripeness == "Unripe":
            distribution = {
                "Unripe": confidence,
                "Partially Ripe": (1 - confidence) * 0.7,
                "Ripe": (1 - confidence) * 0.2,
                "Overripe": (1 - confidence) * 0.1
            }
        elif ripeness == "Ripe":
            distribution = {
                "Unripe": (1 - confidence) * 0.1,
                "Partially Ripe": (1 - confidence) * 0.3,
                "Ripe": confidence,
                "Overripe": (1 - confidence) * 0.6
            }
        elif ripeness == "Overripe":
            distribution = {
                "Unripe": (1 - confidence) * 0.05,
                "Partially Ripe": (1 - confidence) * 0.15,
                "Ripe": (1 - confidence) * 0.3,
                "Overripe": confidence
            }
        else:
            distribution = {
                "Unripe": 0.25,
                "Partially Ripe": 0.25,
                "Ripe": 0.25,
                "Overripe": 0.25
            }
        
        # Mark as estimated
        distribution["estimated"] = True
        
        return distribution
    
    def _standardize_ripeness_label(self, fruit_type, class_name):
        """
        Standardize ripeness label across different models
        
        Args:
            fruit_type: Type of fruit
            class_name: Original class name from model
            
        Returns:
            Standardized ripeness label
        """
        class_name_lower = class_name.lower()
        
        if class_name_lower in ["strawberryripe", "ripe", "freshripe", "pisang_matang"]:
            return "Ripe"
        elif class_name_lower in ["strawberryunripe", "unripe", "freshunripe", "green", "pisang_mentah"]:
            return "Unripe"
        elif class_name_lower in ["strawberryrotten", "overripe", "rotten", "pisang_busuk"]:
            return "Overripe"
        
        if fruit_type == "banana":
            # Map banana ripeness classes to just three categories
            if class_name_lower in ["unripe", "freshunripe", "pisang_mentah"]:
                return "Unripe"
            elif class_name_lower in ["half_ripe", "ripe", "freshripe", "pisang_matang"]:
                return "Ripe"
            elif class_name_lower in ["nearly_rotten", "overripe", "rotten", "pisang_busuk"]:
                return "Overripe"
        elif fruit_type == "tomato":
            if class_name_lower == "green":
                return "Unripe"
            elif class_name_lower in ["breaker", "turning"]:
                return "Partially Ripe"
            elif class_name_lower in ["pink", "light_red", "ripe"]:
                return "Ripe"
            elif class_name_lower == "red":
                return "Fully Ripe"
            elif class_name_lower in ["old", "damaged"]:
                return "Overripe"
        elif fruit_type == "mango":
            # Updated mango classifications
            if class_name_lower in ["unripe", "unripe-1-20-", "early_ripe-21-40-"]:
                return "Unripe"
            elif class_name_lower in ["semiripe", "semi-ripe", "partially_ripe-41-60-", "semi_ripe"]:
                return "Underripe" 
            elif class_name_lower in ["ripe", "ripe-61-80-", "fully ripe"]:
                return "Ripe"
            elif class_name_lower in ["overripe", "over_ripe-81-100-", "over-ripe", "perished", "rotten"]:
                return "Overripe"  # Added "perished" to Overripe category
        
        # If no mapping found, return the original class
        return class_name
    
    def init_fruit_verifier(self):
        """
        Add this method to FruitRipenessSystem class
        It initializes the fruit verifier with the existing classifier
        """
        from fruit_verification import FruitVerifier
        self.fruit_verifier = FruitVerifier(
            self.classifier_model, 
            self.class_names, 
            self.device,
            self.classifier_transform
        )

    def verify_fruit_type(self, image, expected_type):
        """
        Add this method to FruitRipenessSystem class
        It provides a convenient interface to the fruit verifier
        """
        if not hasattr(self, 'fruit_verifier'):
            self.init_fruit_verifier()
        
        return self.fruit_verifier.verify_fruit_type(image, expected_type)
    
    