import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import io

from inference_sdk import InferenceHTTPClient
import os

from improved_visualization import create_enhanced_visualization, create_side_by_side_comparison, create_combined_visualization
from mask_refinement import refine_mask, get_mask_quality_metrics
from models.segmentation_model import UNetResNet50
from models.classifier_model import FruitClassifier
from utils.helpers import apply_mask_to_image

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
        
        self.class_names = ["tomato", "pineapple", "apple", "banana", "orange"]
        self.classifier_model = FruitClassifier(num_classes=len(self.class_names)).to(self.device)
        
        if os.path.exists(classifier_model_path):
            checkpoint = torch.load(classifier_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.classifier_model.load_state_dict(checkpoint)
            print(f"✅ Loaded classifier model locally: {os.path.basename(classifier_model_path)}")
        else:
            print("⚠️ Using randomly initialized classifier weights")
        
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
            "banana": "banana-project/2",
            "strawberry": "strawberry-ml-detection-02/1",
            "mango": "mango-detection-goiq9/1",
        }
        self.supported_fruits = list(self.fruit_to_model.keys())
        os.makedirs('results', exist_ok=True)
        
        self.roboflow_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="UNykbkEetYICFkzzjcqP"
        )
        
        self.seg_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.classifier_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
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
            visualize_regularization_impact, feature_map_visualization
        )
        
        comparison_metrics = compare_model_metrics(
            self.baseline_model, self.seg_model, self.key_comparison_layers
        )
        
        # Generate visualizations
        visualizations = generate_comparison_visualization(
            self.baseline_model, self.seg_model, self.key_comparison_layers
        )
        
        try:
            reg_visualization = visualize_regularization_impact(self.seg_model)
        except Exception as e:
            print(f"Warning: Could not create regularization visualization: {e}")
            reg_visualization = None
        
        
        feature_visualizations = {
            "Baseline Encoder": feature_map_visualization(
                self.baseline_model.activations.get("enc3_conv1"), "Baseline Encoder Features"
            ),
            "Enhanced Encoder": feature_map_visualization(
                self.seg_model.activations.get("encoder3_conv1"), "Enhanced Encoder Features"
            )
        }
        
        # If available, add SFP and dynamic reg visualizations
        if hasattr(self.seg_model, 'activations'):
            if "sfp3_out" in self.seg_model.activations:
                feature_visualizations["SFP Output"] = feature_map_visualization(
                    self.seg_model.activations["sfp3_out"], "Stochastic Feature Pyramid Output"
                )
            if "reg3_out" in self.seg_model.activations:
                feature_visualizations["Dynamic Reg Output"] = feature_map_visualization(
                    self.seg_model.activations["reg3_out"], "Dynamic Regularization Output"
                )
        
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
            "regularization_viz": reg_visualization,
            "mask_metrics": get_mask_quality_metrics(enhanced_mask),
            "refine_segmentation": refine_segmentation,
            "refinement_method": refinement_method
        }
    
    def classify_fruit(self, image):
        """
        Use the fruit classifier to determine the type of fruit
        This method is kept but not used in the main workflow
        
        Args:
            image: PIL Image to classify
            
        Returns:
            The fruit type and confidence score
        """
        print(f"Classifying fruit...")
        
        img_tensor = self.classifier_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            fruit_type = self.class_names[predicted_idx]
        
        print(f"Classified as: {fruit_type} (confidence: {confidence:.2f})")
        
        all_probs = {
            self.class_names[i]: probabilities[i].item()
            for i in range(len(self.class_names))
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
            
            # Extract key components from results
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
                
                # Create result dictionary
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
                
                if angle_name:
                    result["angle_name"] = angle_name
                
                # Add the comparison metrics
                result.update({
                    "comparison_metrics": segmentation_results["comparison_metrics"],
                    "visualizations": segmentation_results["visualizations"],
                    "feature_maps": segmentation_results["feature_maps"],
                    "regularization_viz": segmentation_results["regularization_viz"]
                })
                
                # Create ripeness visualizations
                try:
                    vis_results = self.visualize_results(result)
                    result.update({"visualizations": {**result["visualizations"], **vis_results}})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    result["visualization_error"] = str(e)
                
                return result
            else:
                # Handle case with no predictions
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
                
                # Add the comparison metrics
                result.update({
                    "comparison_metrics": segmentation_results["comparison_metrics"],
                    "visualizations": segmentation_results["visualizations"],
                    "feature_maps": segmentation_results["feature_maps"],
                    "regularization_viz": segmentation_results["regularization_viz"]
                })
                
                # Create visualizations anyway
                try:
                    vis_results = self.visualize_results(result)
                    result.update({"visualizations": {**result["visualizations"], **vis_results}})
                except Exception as e:
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