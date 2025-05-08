import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import io
import base64

class FruitVerifier:
    """Class for fruit type verification"""
    
    def __init__(self, classifier_model, class_names, device, transform=None):
        """
        Initialize the fruit verifier
        
        Args:
            classifier_model: Pretrained fruit classifier model
            class_names: List of class names for classifier
            device: Torch device (cuda/cpu)
            transform: Optional custom transform
        """
        self.classifier_model = classifier_model
        self.class_names = class_names
        self.device = device
        
        # Set model to evaluation mode
        self.classifier_model.eval()
        
        # Define transform if not provided
        if transform is None:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Thresholds
        self.confidence_threshold = 0.75
        self.second_best_ratio = 0.75
    
    def classify_image(self, image):
        """
        Classify a fruit image
        
        Args:
            image: PIL Image to classify
            
        Returns:
            fruit_type, confidence, all_probabilities, visualization_path
        """
        # Prepare image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform classification
        with torch.no_grad():
            outputs = self.classifier_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get predicted class
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            fruit_type = self.class_names[predicted_idx]
            
            # Get all probabilities
            all_probs = {
                self.class_names[i]: probabilities[i].item()
                for i in range(len(self.class_names))
            }
            
            # Create visualization
            viz_path = self._create_visualization(image, all_probs)
            
            return fruit_type, confidence, all_probs, viz_path
    
    def verify_fruit_type(self, image, expected_type):
        """
        Verify if image contains the expected fruit type
        
        Args:
            image: PIL Image to verify
            expected_type: Expected fruit type (string)
            
        Returns:
            Dictionary with verification results
        """
        # Classify the image
        detected_type, confidence, all_probs, viz_path = self.classify_image(image)
        
        # Sort probabilities
        sorted_probs = {k: v for k, v in sorted(
            all_probs.items(), key=lambda item: item[1], reverse=True)}
        
        # Get second best prediction
        second_best = list(sorted_probs.values())[1] if len(sorted_probs) > 1 else 0
        second_best_ratio = second_best / max(confidence, 0.001)
        
        # Special handling for out-of-scope fruits
        if detected_type == "outside_scope":
            message = (
                f"The image appears to contain a fruit that's not in our supported list. "
                f"The highest match is {list(sorted_probs.keys())[0]} at only {list(sorted_probs.values())[0]:.2f} confidence. "
                f"Please try a different image or select a supported fruit type."
            )
            
            return {
                "is_match": False,
                "is_uncertain": True,
                "expected_type": expected_type.lower(),
                "detected_type": "unknown",
                "confidence": confidence,
                "second_best_ratio": second_best_ratio,
                "all_probabilities": sorted_probs,
                "message": message,
                "visualization_path": viz_path
            }
    
    def _create_visualization(self, image, probabilities):
        """
        Create visualization of classification results
        
        Args:
            image: PIL Image
            probabilities: Dictionary of class probabilities
            
        Returns:
            Path to saved visualization
        """
        # Create a copy to avoid modifying original
        img_copy = image.copy()
        # Resize for display if needed
        if max(img_copy.size) > 300:
            img_copy.thumbnail((300, 300))
        
        # Sort probabilities
        sorted_probs = {k: v for k, v in sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True)}
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display image
        ax1.imshow(np.array(img_copy))
        ax1.set_title("Input Image")
        ax1.axis("off")
        
        # Create bar chart
        labels = list(sorted_probs.keys())
        values = list(sorted_probs.values())
        
        # Limit to top 5 classes
        if len(labels) > 5:
            labels = labels[:5]
            values = values[:5]
        
        # Define colors - highlight top result
        colors = ['skyblue' for _ in values]
        colors[0] = 'royalblue'
        
        # Create horizontal bar chart
        bars = ax2.barh(labels, [v * 100 for v in values], color=colors)
        ax2.set_title("Fruit Type Probabilities")
        ax2.set_xlabel("Confidence (%)")
        ax2.set_xlim(0, 100)
        
        # Add percentage labels
        for bar, value in zip(bars, values):
            ax2.text(min(value * 100 + 3, 95), bar.get_y() + bar.get_height()/2, 
                     f"{value*100:.1f}%", va='center')
        
        # Save visualization
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        timestamp = int(time.time())
        save_path = f"results/classification_{timestamp}.png"
        plt.savefig(save_path)
        plt.close(fig)
        
        return save_path


def display_verification_warning(verification_results):
    """
    Display fruit type verification warning UI
    
    Args:
        verification_results: Results from fruit verifier
    """
    st.warning("⚠️ Fruit Type Mismatch Detected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(verification_results["original_image"], use_container_width=True)
    
    with col2:
        st.subheader("Segmented Image")
        st.image(verification_results["segmented_image"], use_container_width=True)
    
    # Display classification visualization if available
    if "visualization_path" in verification_results and verification_results["visualization_path"]:
        viz_path = verification_results["visualization_path"]
        if os.path.exists(viz_path):
            st.image(viz_path, use_container_width=True)
    
    st.markdown(f"### Fruit Type Verification Results")
    
    # Display the mismatch information
    if verification_results.get('detected_type') == "unknown":
        st.markdown(f"""
        - **You selected:** {verification_results['fruit_type_selected'].title()}
        - **Detected fruit:** Unknown (outside of supported fruit types)
        - **Highest match:** {list(verification_results['all_probabilities'].keys())[0].title()} at only {list(verification_results['all_probabilities'].values())[0]:.2f} confidence
        """)
    else:
        st.markdown(f"""
        - **You selected:** {verification_results['fruit_type_selected'].title()}
        - **Detected fruit:** {verification_results['detected_type'].title()} (confidence: {verification_results['confidence']:.2f})
        """)
    
    # Display message
    st.info(verification_results.get("message", "Please verify the fruit type."))
    
    # Display all probabilities as a table
    if 'all_probabilities' in verification_results and verification_results['all_probabilities']:
        probs = verification_results['all_probabilities']
        
        st.markdown("### Fruit Classification Probabilities")
        
        # Display as a table
        st.table({
            "Fruit Type": [k.title() for k in probs.keys()],
            "Confidence": [f"{v:.2%}" for v in probs.values()]
        })
    
    # Provide options to the user
    st.markdown("### What would you like to do?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Proceed Anyway", type="primary"):
            # Proceed with original fruit type
            st.session_state.verified_fruit_type = verification_results['fruit_type_selected']
            st.session_state.ignore_verification = True
            # Remove verification results
            if "verification_results" in st.session_state:
                del st.session_state.verification_results
            st.rerun()
    
    with col2:
        if verification_results.get('detected_type') != "unknown":
            if st.button(f"Switch to {verification_results['detected_type'].title()}", type="primary"):
                # Switch to detected fruit type
                st.session_state.selected_fruit = verification_results['detected_type'].title()
                st.session_state.verified_fruit_type = verification_results['detected_type']
                # Remove verification results
                if "verification_results" in st.session_state:
                    del st.session_state.verification_results
                st.rerun()
        else:
            # No specific fruit was reliably detected
            st.info("No specific fruit detected with sufficient confidence")
    
    with col3:
        if st.button("Cancel Analysis", type="secondary"):
            # Go back to fruit selection
            st.session_state.selected_fruit = None
            st.session_state.analysis_step = "select_fruit"
            st.session_state.uploaded_file = None
            st.session_state.camera_image = None
            # Remove verification results
            if "verification_results" in st.session_state:
                del st.session_state.verification_results
            st.rerun()