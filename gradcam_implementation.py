import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import time
import io
from torchvision import transforms

class ImprovedConvNeXtGradCAM:
    """
    Improved Grad-CAM implementation specifically for ConvNeXt architecture
    with better visualization quality.
    """
    def __init__(self, model, device):
        """
        Initialize Grad-CAM for ConvNeXt models
        
        Args:
            model: PyTorch ConvNeXt model
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        
        # Initialize attributes to capture activations and gradients
        self.activations = None
        self.gradients = None
        
        # Find a suitable target layer regardless of model architecture
        target_layer = self._find_target_layer(model)
        
        if target_layer is not None:
            print(f"Found target layer: {target_layer.__class__.__name__}")
            target_layer.register_forward_hook(self._save_activation)
            target_layer.register_full_backward_hook(self._save_gradient)
        else:
            raise ValueError("Could not find a suitable convolutional layer in the model")
    
    def _find_target_layer(self, model):
        """Find a suitable target layer for Grad-CAM in any ConvNeXt variant"""
        # Strategy 1: Check for specific architecture patterns
        
        # For ConvNeXt models, we want a layer that's deep enough to capture semantic information
        # but not too deep to lose spatial information
        
        # Approach 1: Try to find specific layer patterns from the model architecture
        for name, module in model.named_modules():
            # Look for specific layers depending on the architecture
            if isinstance(module, torch.nn.Conv2d):
                # ConvNeXt typically has "stages" or "features" followed by a number
                if any(pattern in name for pattern in ['stage.2', 'features.6', 'layer3', 'block3']):
                    print(f"Found ideal target layer at: {name}")
                    return module
        
        # Strategy 2: Find any convolutional layer at an appropriate depth
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Calculate depth by counting dots in the name
                depth = name.count('.') + 1
                conv_layers.append((name, module, depth))
        
        # Sort by depth (descending)
        conv_layers.sort(key=lambda x: x[2], reverse=True)
        
        # Get a layer from about 2/3 through the network
        if conv_layers:
            target_index = min(len(conv_layers) // 3 * 2, len(conv_layers) - 1)
            name, module, depth = conv_layers[target_index]
            print(f"Selected Conv2d layer at: {name} (depth: {depth})")
            return module
        
        # Fallback: just find any Conv2d
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                return module
        
        return None
    
    def _save_activation(self, module, input, output):
        """Hook to save activations during forward pass"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate class activation map
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index for CAM generation (default: highest scoring class)
            
        Returns:
            Normalized heatmap as numpy array
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        
        # Perform forward pass
        outputs = self.model(input_tensor)
        
        # If target_class is None, use the class with highest score
        if target_class is None:
            target_class = torch.argmax(outputs, dim=1).item()
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Create one-hot encoding for backpropagation
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        
        # Backward pass
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        # Check if activations and gradients were captured
        if self.activations is None or self.gradients is None:
            raise ValueError("Failed to capture activations or gradients. Check model architecture.")
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Compute weighted activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to focus on features that positively influence the target class
        cam = torch.nn.functional.relu(cam)
        
        # Resize and normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # Improve contrast with CLAHE
        if cam.max() > cam.min():
            # Normalize to 0-255 range
            cam_normalized = np.uint8(255 * (cam - cam.min()) / (cam.max() - cam.min()))
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cam_enhanced = clahe.apply(cam_normalized)
            
            # Normalize back to 0-1
            cam = cam_enhanced.astype(float) / 255.0
        else:
            # Fallback if all values are the same
            cam = cam - cam.min()
        
        return cam

    def visualize(self, input_tensor, original_image, target_class=None, save_path=None, class_name=None):
        """
        Visualize Grad-CAM heatmap overlaid on original image with improved visualization
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            original_image: Original PIL image
            target_class: Target class index (default: highest scoring class)
            save_path: Path to save visualization (default: None, just returns image)
            class_name: Class name to add to the title (optional)
            
        Returns:
            Visualization as PIL Image and path if saved
        """
        try:
            # Generate CAM
            cam = self.generate_cam(input_tensor, target_class)
            
            # Create a figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 1. Original image
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # 2. Heatmap only
            heatmap_display = axes[1].imshow(cam, cmap='jet')
            axes[1].set_title("Class Activation Map")
            axes[1].axis('off')
            fig.colorbar(heatmap_display, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 3. Overlay with adjustable transparency
            # Convert original image to numpy array
            img_array = np.array(original_image)
            
            # Convert heatmap to RGB colormap with jet colormap
            heatmap_rgb = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
            
            # Resize heatmap to match original image size
            heatmap_rgb = cv2.resize(heatmap_rgb, (img_array.shape[1], img_array.shape[0]))
            
            # Use 40% transparency for clearer visualization
            alpha = 0.4
            overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_rgb, alpha, 0)
            
            # Display overlay
            axes[2].imshow(overlay)
            axes[2].set_title(f"Grad-CAM: {class_name} Class" if class_name else "Grad-CAM Overlay")
            axes[2].axis('off')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                # Also save just the overlay for quick display
                overlay_img = Image.fromarray(overlay)
                overlay_path = save_path.replace('.png', '_overlay.png')
                overlay_img.save(overlay_path)
                
                return Image.open(save_path), save_path
            
            # Convert figure to PIL image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            buf.seek(0)
            return Image.open(buf)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating Grad-CAM visualization: {str(e)}")
            # Return original image in case of error
            return original_image

def apply_improved_gradcam(system, fruit_data, fruit_type, ripeness_class=None):
    """
    Apply improved Grad-CAM to a fruit image
    
    Args:
        system: FruitRipenessSystem instance
        fruit_data: Fruit data dictionary from results
        fruit_type: Type of fruit being analyzed
        ripeness_class: Ripeness class name (e.g., "Ripe", "Unripe") 
        
    Returns:
        Path to saved Grad-CAM visualization
    """
    try:
        # Get the custom model handler
        custom_model_handler = system.custom_model_handler
        
        # Load the model if not already loaded
        if fruit_type.lower() not in custom_model_handler.loaded_models:
            success = custom_model_handler.load_model(fruit_type)
            if not success:
                print(f"Failed to load model for {fruit_type}")
                return None
        
        # Get the model and class names
        model = custom_model_handler.loaded_models.get(fruit_type.lower())
        model_config = custom_model_handler.model_configs.get(fruit_type.lower())
        
        if not model or not model_config:
            print(f"Model or config not found for {fruit_type}")
            return None
        
        # Get class names and mapping to indices
        class_names = model_config.get("class_names", ["Unripe", "Ripe", "Overripe"])
        
        # Check for image paths
        if "masked_crop_path" in fruit_data and os.path.exists(fruit_data["masked_crop_path"]):
            image_path = fruit_data["masked_crop_path"]
        elif "original_crop_path" in fruit_data and os.path.exists(fruit_data["original_crop_path"]):
            image_path = fruit_data["original_crop_path"]
        else:
            print("No valid image path found in fruit data")
            return None
            
        print(f"Using image from: {image_path}")
        
        # Load the image
        image = Image.open(image_path).convert('RGB')
        
        # Create transform using the transforms module
        transform = transforms.Compose([
            transforms.Resize(model_config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                model_config["normalize_mean"],
                model_config["normalize_std"]
            )
        ])
        
        # Preprocess image
        img_tensor = transform(image).unsqueeze(0).to(system.device)
        
        # Initialize improved Grad-CAM
        grad_cam = ImprovedConvNeXtGradCAM(model, system.device)
        
        # Map ripeness class to index if provided
        target_class = None
        if ripeness_class:
            # Try to find ripeness class in class_names
            try:
                # First try exact match
                target_class = class_names.index(ripeness_class)
                print(f"Found class index for {ripeness_class}: {target_class}")
            except ValueError:
                # Try case-insensitive match
                for i, name in enumerate(class_names):
                    if name.lower() == ripeness_class.lower():
                        target_class = i
                        print(f"Found class index for {ripeness_class} (case-insensitive): {target_class}")
                        break
        
        # If class not found, use model to predict class
        if target_class is None:
            # Run inference to get prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
                print(f"Using predicted class index: {target_class}")
            
            # Get class name for visualization
            ripeness_class = class_names[target_class] if target_class < len(class_names) else f"Class {target_class}"
            print(f"Class name: {ripeness_class}")
        
        # Generate timestamp for unique filename
        timestamp = int(time.time())
        os.makedirs("results", exist_ok=True)
        save_path = f"results/gradcam_{fruit_type}_{ripeness_class}_{timestamp}.png"
        
        # Apply GradCAM visualization with improved version
        _, vis_path = grad_cam.visualize(
            img_tensor, 
            image, 
            target_class=target_class,
            save_path=save_path,
            class_name=ripeness_class
        )
        
        return vis_path
    
    except Exception as e:
        print(f"Error generating improved Grad-CAM visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def add_gradcam_to_technical_details(st, results, system):
    """
    Add improved Grad-CAM visualization to the technical details section
    
    Args:
        st: Streamlit instance
        results: Dictionary containing processing results
        system: FruitRipenessSystem instance
    """
    # Generate a unique base ID for this results object to avoid key conflicts
    result_base_id = id(results) % 10000
    
    print("Starting Grad-CAM visualization process")
    
    st.subheader("🔍 Grad-CAM Visualization")
    
    st.write("""
    Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations for 
    CNN-based models by highlighting the regions of the image that influenced the classification decision.
    """)
    
    # If we have multiple fruits, create tabs
    num_fruits = results.get("num_fruits", 1)
    print(f"Number of fruits detected: {num_fruits}")
    
    if num_fruits > 1 and "fruits_data" in results:
        fruit_tabs = st.tabs([f"Fruit #{i+1}" for i in range(num_fruits)])
        
        for i, (tab, fruit_data) in enumerate(zip(fruit_tabs, results.get("fruits_data", []))):
            with tab:
                try:
                    # Generate a unique key for this specific fruit tab
                    fruit_key = f"fruit_{i}_{result_base_id}"
                    print(f"Processing Grad-CAM for fruit #{i+1} with key {fruit_key}")
                    
                    # Get confidence distribution for this fruit
                    confidence_dist = results.get("confidence_distributions", [])[i] if i < len(results.get("confidence_distributions", [])) else {}
                    
                    # Skip if error in confidence distribution
                    if "error" in confidence_dist:
                        st.warning("Cannot generate Grad-CAM for this fruit due to classification errors.")
                        continue
                    
                    # Get ripeness class with highest confidence
                    filtered_dist = {k: v for k, v in confidence_dist.items() if k not in ["error", "estimated"]}
                    if not filtered_dist:
                        st.warning("No valid ripeness classes found in confidence distribution.")
                        continue
                        
                    ripeness_class, confidence = max(filtered_dist.items(), key=lambda x: x[1])
                    print(f"Selected ripeness class: {ripeness_class} (confidence: {confidence:.2f})")
                    
                    # Check if we already have a generated Grad-CAM path
                    existing_path = None
                    if "gradcam_paths" in results and i < len(results.get("gradcam_paths", [])):
                        existing_path = results["gradcam_paths"][i]
                        if existing_path and os.path.exists(existing_path):
                            print(f"Using existing Grad-CAM path: {existing_path}")
                    
                    # Use existing path or generate new one
                    if existing_path and os.path.exists(existing_path):
                        gradcam_path = existing_path
                    else:
                        print(f"Generating new Grad-CAM visualization for fruit #{i+1}")
                        # Apply improved Grad-CAM 
                        gradcam_path = apply_improved_gradcam(
                            system, 
                            fruit_data, 
                            results.get("fruit_type", "unknown"),
                            ripeness_class
                        )
                    
                    if gradcam_path and os.path.exists(gradcam_path):
                        # Check if overlay exists (for smaller display)
                        overlay_path = gradcam_path.replace('.png', '_overlay.png')
                        if os.path.exists(overlay_path):
                            st.write(f"### Grad-CAM for '{ripeness_class}' Class")
                            overlay_img = Image.open(overlay_path)
                            st.image(overlay_img, use_container_width=True)
                        else:
                            st.write(f"### Grad-CAM for '{ripeness_class}' Class")
                            gradcam_img = Image.open(gradcam_path)
                            st.image(gradcam_img, use_container_width=True)
                        
                        st.write("""
                        The highlighted regions show areas of the fruit that most influenced the classification decision.
                        Red/yellow areas had stronger influence on the model's prediction.
                        """)
                        
                        # Add button to view full visualization with UNIQUE KEY
                        if st.button(f"View Full Grad-CAM Analysis", key=f"view_gradcam_{fruit_key}"):
                            st.image(Image.open(gradcam_path), use_container_width=True)
                    else:
                        # Generate button with unique key for generating Grad-CAM
                        if st.button("Generate Grad-CAM Visualization", key=f"generate_gradcam_{fruit_key}"):
                            with st.spinner(f"Generating Grad-CAM for Fruit #{i+1}..."):
                                try:
                                    gradcam_path = apply_improved_gradcam(
                                        system, 
                                        fruit_data, 
                                        results.get("fruit_type", "unknown"),
                                        ripeness_class
                                    )
                                    
                                    if gradcam_path and os.path.exists(gradcam_path):
                                        st.success("Grad-CAM generated successfully!")
                                        st.image(Image.open(gradcam_path), use_container_width=True)
                                    else:
                                        st.warning("Could not generate Grad-CAM visualization.")
                                except Exception as e:
                                    st.error(f"Error generating Grad-CAM: {str(e)}")
                        else:
                            st.warning("Click the button above to generate Grad-CAM visualization.")
                        
                except Exception as e:
                    st.error(f"Error in Grad-CAM processing for fruit #{i+1}: {str(e)}")
                    print(f"Exception in Grad-CAM for fruit #{i+1}: {str(e)}")
                    import traceback
                    trace = traceback.format_exc()
                    print(trace)
                    st.text(trace)
    else:
        # Single fruit case
        try:
            fruit_data = results.get("fruits_data", [])[0] if results.get("fruits_data") else {}
            
            # Generate a unique key for this single fruit
            single_key = f"single_{result_base_id}"
            print(f"Processing Grad-CAM for single fruit with key {single_key}")
            
            # Get confidence distribution
            confidence_dist = results.get("confidence_distributions", [])[0] if results.get("confidence_distributions") else {}
            
            # Skip if error in confidence distribution
            if "error" in confidence_dist:
                st.warning("Cannot generate Grad-CAM due to classification errors.")
                return
            
            # Get ripeness class with highest confidence
            filtered_dist = {k: v for k, v in confidence_dist.items() if k not in ["error", "estimated"]}
            if not filtered_dist:
                st.warning("No valid ripeness classes found in confidence distribution.")
                return
                
            ripeness_class, confidence = max(filtered_dist.items(), key=lambda x: x[1])
            print(f"Selected ripeness class: {ripeness_class} (confidence: {confidence:.2f})")
            
            # Check if we already have a generated Grad-CAM path
            existing_path = results.get("gradcam_path") if "gradcam_path" in results else None
            if existing_path and os.path.exists(existing_path):
                print(f"Using existing Grad-CAM path: {existing_path}")
                gradcam_path = existing_path
            else:
                print("Generating new Grad-CAM visualization for single fruit")
                # Apply improved Grad-CAM
                gradcam_path = apply_improved_gradcam(
                    system, 
                    fruit_data, 
                    results.get("fruit_type", "unknown"),
                    ripeness_class
                )
            
            if gradcam_path and os.path.exists(gradcam_path):
                # Check if overlay exists (for smaller display)
                overlay_path = gradcam_path.replace('.png', '_overlay.png')
                if os.path.exists(overlay_path):
                    st.write(f"### Grad-CAM for '{ripeness_class}' Class")
                    overlay_img = Image.open(overlay_path)
                    st.image(overlay_img, use_container_width=True)
                else:
                    st.write(f"### Grad-CAM for '{ripeness_class}' Class")
                    gradcam_img = Image.open(gradcam_path)
                    st.image(gradcam_img, use_container_width=True)
                
                st.write("""
                The highlighted regions show areas of the fruit that most influenced the classification decision.
                Red/yellow areas had stronger influence on the model's prediction.
                """)
                
                # Add button to view full visualization with UNIQUE KEY
                if st.button("View Full Grad-CAM Analysis", key=f"view_gradcam_{single_key}"):
                    st.image(Image.open(gradcam_path), use_container_width=True)
            else:
                # Generate button with unique key for generating Grad-CAM
                if st.button("Generate Grad-CAM Visualization", key=f"generate_gradcam_{single_key}"):
                    with st.spinner("Generating Grad-CAM visualization..."):
                        try:
                            gradcam_path = apply_improved_gradcam(
                                system, 
                                fruit_data, 
                                results.get("fruit_type", "unknown"),
                                ripeness_class
                            )
                            
                            if gradcam_path and os.path.exists(gradcam_path):
                                st.success("Grad-CAM generated successfully!")
                                st.image(Image.open(gradcam_path), use_container_width=True)
                            else:
                                st.warning("Could not generate Grad-CAM visualization.")
                        except Exception as e:
                            st.error(f"Error generating Grad-CAM: {str(e)}")
                else:
                    st.warning("Click the button above to generate Grad-CAM visualization.")
                
        except Exception as e:
            st.error(f"Error in Grad-CAM processing: {str(e)}")
            print(f"Exception in Grad-CAM for single fruit: {str(e)}")
            import traceback
            trace = traceback.format_exc()
            print(trace)
            st.text(trace)
            
def add_gradcam_to_card_technical_details(st, results, system):
    """
    Automatic Grad-CAM visualization for patch-based analysis
    without requiring button clicks
    """
    try:
        st.subheader("🔍 Grad-CAM Visualizations")
        
        # Only proceed if we have angle results
        if "angle_results" not in results or len(results["angle_results"]) == 0:
            st.info("No angle results found for Grad-CAM visualization.")
            return
        
        # Get fruit type
        fruit_type = results.get("fruit_type", "unknown")
        print(f"Using fruit type: {fruit_type}")
        
        # Create columns for angles (max 2)
        angles_to_show = min(len(results["angle_results"]), 2)
        cols = st.columns(angles_to_show)
        
        # Process each angle
        for i in range(angles_to_show):
            with cols[i]:
                angle_result = results["angle_results"][i]
                angle_name = results["angle_names"][i] if "angle_names" in results and i < len(results["angle_names"]) else f"Angle {i+1}"
                
                st.write(f"### {angle_name}")
                
                # Check if we already have a GradCAM image
                existing_path = None
                if "gradcam_paths" in results and i < len(results.get("gradcam_paths", [])):
                    existing_path = results["gradcam_paths"][i]
                elif "gradcam_path" in angle_result:
                    existing_path = angle_result["gradcam_path"]
                
                # Use existing or generate new
                if existing_path and os.path.exists(existing_path):
                    st.image(Image.open(existing_path), use_container_width=True)
                    st.success(f"Grad-CAM visualization for {angle_name}")
                else:
                    with st.spinner(f"Generating Grad-CAM for {angle_name}..."):
                        # Get fruit data
                        if "fruits_data" not in angle_result or len(angle_result["fruits_data"]) == 0:
                            st.error(f"No fruit data found for {angle_name}")
                            continue
                        
                        fruit_data = angle_result["fruits_data"][0]
                        
                        # Get ripeness class
                        ripeness_class = "Ripe"  # Default
                        
                        # Try to get the actual class from confidence distributions
                        if "confidence_distributions" in angle_result and len(angle_result["confidence_distributions"]) > 0:
                            conf_dist = angle_result["confidence_distributions"][0]
                            filtered_dist = {k: v for k, v in conf_dist.items() if k not in ["error", "estimated"]}
                            if filtered_dist:
                                ripeness_class, _ = max(filtered_dist.items(), key=lambda x: x[1])
                        
                        # Generate the GradCAM
                        try:
                            gradcam_path = apply_improved_gradcam(
                                system,
                                fruit_data,
                                fruit_type,
                                ripeness_class
                            )
                            
                            if gradcam_path and os.path.exists(gradcam_path):
                                # Show the image
                                st.image(Image.open(gradcam_path), use_container_width=True)
                                st.success(f"Grad-CAM visualization for {angle_name}")
                                
                                # Store paths for future use
                                if "gradcam_paths" not in results:
                                    results["gradcam_paths"] = [None] * len(results["angle_results"])
                                if i < len(results["gradcam_paths"]):
                                    results["gradcam_paths"][i] = gradcam_path
                                
                                angle_result["gradcam_path"] = gradcam_path
                            else:
                                st.error(f"Failed to generate Grad-CAM for {angle_name}")
                        
                        except Exception as e:
                            st.error(f"Error generating Grad-CAM: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
    
    except Exception as e:
        st.error(f"Error in Grad-CAM visualization: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")