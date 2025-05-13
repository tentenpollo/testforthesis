import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import torch
import io
import json

from system import FruitRipenessSystem
from utils.helpers import get_image_download_link, seed_everything, make_serializable
from user_management import initialize_database, save_user_result, save_enhanced_user_result
from authentication import show_login_page, get_current_user
from user_history import show_history_page, show_result_details
from huggingface_hub import hf_hub_download
from evaluation_page import add_evaluation_page_to_app
from confidence_visualization import visualize_confidence_distribution, visualize_all_fruits_confidence

st.set_page_config(
    page_title="Fruit Ripeness Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed" 
)
initialize_database()

if "page" not in st.session_state:
    st.session_state.page = "login"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "view_details" not in st.session_state:
    st.session_state.view_details = False
if "selected_result_id" not in st.session_state:
    st.session_state.selected_result_id = None
    
if "selected_fruit" not in st.session_state:
    st.session_state.selected_fruit = None
if "analysis_step" not in st.session_state:
    st.session_state.analysis_step = "select_fruit"  # Possible values: "select_fruit", "upload_image", "analyze"
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "camera_image" not in st.session_state:
    st.session_state.camera_image = None
    
seed_everything(42)

os.makedirs('results', exist_ok=True)

@st.cache_resource
def load_models(
    seg_model_repo="TentenPolllo/fruitripeness",
    classifier_model_repo="TentenPolllo/FruitClassifier"
):
    """Load both segmentation and classifier models from HF Hub without showing alerts"""
    import time
    from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
    
    class DummyContext:
        def __enter__(self):
            # This placeholder is needed for the context manager
            pass
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # This placeholder is needed for the context manager
            pass
    
    # Replace st.info and st.success with no-op functions temporarily
    original_info = st.info
    original_success = st.success
    original_warning = st.warning
    original_error = st.error
    
    # Create no-op functions that do nothing
    def silent_info(message, *args, **kwargs):
        pass
    
    def silent_success(message, *args, **kwargs):
        pass
    
    def silent_warning(message, *args, **kwargs):
        pass
    
    def silent_error(message, *args, **kwargs):
        # For errors, we might still want to log them somewhere, but not display
        # You could add logging here if needed
        pass
    
    # Replace the streamlit functions with silent versions
    st.info = silent_info
    st.success = silent_success
    st.warning = silent_warning
    st.error = silent_error
    
    try:
        # Original code for model loading but without the visible alerts
        max_retries = 5
        retry_delay = 2

        seg_model_path = None
        for attempt in range(max_retries):
            try:
                # Silent version of the original st.info message
                seg_model_path = hf_hub_download(
                    repo_id=seg_model_repo,
                    filename="best_model.pth",
                )
                # Silent version of the success message
                break
            except (HfHubHTTPError, LocalEntryNotFoundError) as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Silent version of the warning
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    # Silent version of the error
                    break

        classifier_model_path = None
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                # Silent version of the info message
                classifier_model_path = hf_hub_download(
                    repo_id=classifier_model_repo,
                    filename="fruit_classifier_full.pth",
                )
                # Silent version of the success message
                break
            except (HfHubHTTPError, LocalEntryNotFoundError) as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Silent version of the warning
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    # Silent version of the error
                    break
        
        # Initialize system with downloaded models
        return FruitRipenessSystem(
            seg_model_path=seg_model_path,
            classifier_model_path=classifier_model_path
        )
    except Exception as e:
        # Silent version of the error
        # Return a minimal system with error handling
        class FallbackSystem:
            def __init__(self):
                self.device = "cpu"
                self.supported_fruits = ["banana", "tomato", "pineapple", "strawberry", "mango"]
                
            def process_image_with_visualization(self, *args, **kwargs):
                return {"error": "Models could not be loaded. Please try again later."}
                
            def process_image_without_segmentation(self, *args, **kwargs):
                return {"error": "Models could not be loaded. Please try again later."}
                
            def analyze_ripeness_enhanced(self, *args, **kwargs):
                return {"error": "Models could not be loaded. Please try again later."}
        
        return FallbackSystem()
    finally:
        # Restore the original streamlit functions
        st.info = original_info
        st.success = original_success
        st.warning = original_warning
        st.error = original_error

def display_card_ripeness_results(results, system, username):
    """Display ripeness analysis results in a card format with percentages"""
    if "error" in results and "fruits_data" not in results:
        st.error(f"Error: {results['error']}")
        return
    
    # Get basic info from results
    use_segmentation = results.get("use_segmentation", True)
    fruit_type = results.get("fruit_type", "Unknown")
    num_fruits = results.get("num_fruits", 0)
    
    # Display header with fruit emoji
    fruit_emoji = get_fruit_emoji(fruit_type)
    st.header(f"{fruit_emoji} {fruit_type.title()} Analysis")
    
    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        if "visualizations" in results and "bounding_box_visualization" in results["visualizations"]:
            bbox_path = results["visualizations"]["bounding_box_visualization"]
            bbox_img = Image.open(bbox_path)
            st.image(bbox_img, use_container_width=True)
        else:
            st.image(results["original_image"], use_container_width=True)
    
    with col2:
        if use_segmentation:
            st.subheader("Processed Fruit")
            st.image(results["segmented_image"], use_container_width=True)
        else:
            st.subheader("Processed Image")
            st.image(results["original_image"], use_container_width=True)
    
    # Handle multiple fruits
    if num_fruits > 1:
        # Create tabs for each fruit
        fruit_tabs = st.tabs([f"Fruit #{i+1}" for i in range(num_fruits)])
        
        for i, (tab, fruit_data) in enumerate(zip(fruit_tabs, results.get("fruits_data", []))):
            with tab:
                confidence_distribution = results.get("confidence_distributions", [])[i] if i < len(results.get("confidence_distributions", [])) else {}
                display_ripeness_card(fruit_data, confidence_distribution, fruit_type, i+1)
    else:
        # Single fruit display
        confidence_distribution = results.get("confidence_distributions", [])[0] if results.get("confidence_distributions") else {}
        fruit_data = results.get("fruits_data", [])[0] if results.get("fruits_data") else {}
        display_ripeness_card(fruit_data, confidence_distribution, fruit_type)
    
    # Technical details in an expander
    with st.expander("Technical Details"):
        # Display confidence distributions and visualizations
        if num_fruits > 1:
            for i, distribution in enumerate(results.get("confidence_distributions", [])):
                if distribution and "error" not in distribution:
                    st.subheader(f"Ripeness Confidence for Fruit #{i+1}")
                    
                    # Filter out non-confidence keys
                    filtered_distribution = {k: v for k, v in distribution.items() 
                                          if k not in ["error", "estimated"]}
                    
                    # Create table
                    confidence_data = {
                        "Ripeness Level": list(filtered_distribution.keys()),
                        "Confidence": [f"{v:.2f}" for v in filtered_distribution.values()],
                        "Percentage": [f"{v*100:.1f}%" for v in filtered_distribution.values()]
                    }
                    
                    st.table(confidence_data)
                    
                    # Display visualization if available
                    viz_path = visualize_confidence_distribution(
                        results.get("fruits_data", [])[i], distribution, fruit_type
                    )
                    viz_img = Image.open(viz_path)
                    st.image(viz_img, use_container_width=True)
        else:
            distribution = results.get("confidence_distributions", [])[0] if results.get("confidence_distributions") else {}
            if distribution and "error" not in distribution:
                st.subheader("Ripeness Confidence Values")
                
                # Filter out non-confidence keys
                filtered_distribution = {k: v for k, v in distribution.items() 
                                      if k not in ["error", "estimated"]}
                
                # Create table
                confidence_data = {
                    "Ripeness Level": list(filtered_distribution.keys()),
                    "Confidence": [f"{v:.2f}" for v in filtered_distribution.values()],
                    "Percentage": [f"{v*100:.1f}%" for v in filtered_distribution.values()]
                }
                
                st.table(confidence_data)
                
                # Display visualization if available
                viz_path = visualize_confidence_distribution(
                    results.get("fruits_data", [])[0], distribution, fruit_type
                )
                viz_img = Image.open(viz_path)
                st.image(viz_img, use_container_width=True)
        
        # Add segmentation mask and model information
        if use_segmentation:
            st.subheader("Segmentation Mask")
            st.image(results["mask"] * 255, clamp=True, use_container_width=True)
            
            if "mask_metrics" in results:
                st.write("**Mask Quality Metrics:**")
                metrics = results["mask_metrics"]
                st.write(f"- Mask coverage: {metrics['coverage_ratio']:.2%} of image")
                st.write(f"- Boundary complexity: {metrics['boundary_complexity']:.2f}")
                
        try:
            from gradcam_implementation import add_gradcam_to_technical_details
            
            add_gradcam_to_technical_details(st, results, system)
        except Exception as e:
            st.warning(f"Could not add Grad-CAM visualization: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
        
        # Add model information
        st.subheader("Model Information")
        st.write(f"- Device used: {system.device}")
        st.write(f"- Segmentation: {'Enabled' if use_segmentation else 'Disabled'}")
        # Add classification model details if available
        if "classification_results" in results:
            if isinstance(results["classification_results"], list) and len(results["classification_results"]) > 0:
                first_result = results["classification_results"][0]
                if isinstance(first_result, dict):
                    st.write(f"- Classification model: {first_result.get('model_name', 'Custom Model')}")
                    st.write(f"- Classification model input size: {first_result.get('input_size', '224x224')}")
    
    # Save results section
    if username and username != "guest":
        save_user_results(results, username)
    else:
        st.info("💡 Log in with a user account to save your analysis results for future reference.")

def display_ripeness_card(fruit_data, confidence_distribution, fruit_type, fruit_number=None):
    """Display fruit ripeness information with a clean, functional layout"""
    
    # Get ripeness assessment and percentages
    ripeness_level, confidence, all_percentages = get_ripeness_with_percentages(confidence_distribution)
    
    # Create main container
    with st.container():
        # Header row with fruit image and title
        header_cols = st.columns([1, 3]) if "masked_crop_path" in fruit_data else [1]
        
        # Image column
        with header_cols[0]:
            if "masked_crop_path" in fruit_data and os.path.exists(fruit_data["masked_crop_path"]):
                img = Image.open(fruit_data["masked_crop_path"])
                st.image(img, width=150)
        
        # Title and ripeness info column
        with header_cols[1] if len(header_cols) > 1 else header_cols[0]:
            # Fruit title with emoji
            emoji = get_fruit_emoji(fruit_type)
            title = f"{emoji} {fruit_type.title()}"
            if fruit_number:
                title += f" #{fruit_number}"
            st.markdown(f"## {title}")
            
            # Ripeness indicator
            ripeness_icon = get_ripeness_icon(ripeness_level)
            st.markdown(f"### {ripeness_icon} {ripeness_level} ({confidence*100:.1f}%)")
        
        # Assessment section
        st.markdown("## Assessment & Recommendations")
        
        # Create three columns for assessment sections
        col1, col2, col3 = st.columns(3)
        
        # Condition column
        with col1:
            st.markdown("### Condition")
            condition_text = get_condition_text(ripeness_level, fruit_type)
            st.markdown(condition_text)
            
            # Add a colored box or indicator based on ripeness
            if ripeness_level == "Ripe":
                st.success("Optimal condition")
            elif ripeness_level == "Unripe":
                st.warning("Not ready yet")
            elif ripeness_level == "Overripe":
                st.error("Past prime condition")
            else:
                st.info("Condition unknown")
        
        # Shelf Life column
        with col2:
            st.markdown("### Shelf Life")
            shelf_life = get_shelf_life(ripeness_level, fruit_type)
            shelf_icon = get_shelf_life_icon(ripeness_level)
            st.markdown(f"{shelf_icon} {shelf_life}")
            
            # Add a visual indicator
            if ripeness_level == "Ripe":
                st.warning("Limited time remaining")
            elif ripeness_level == "Unripe":
                st.success("Good shelf life")
            elif ripeness_level == "Overripe":
                st.error("Use immediately")
            else:
                st.info("Shelf life unknown")
        
        # Recommendation column
        with col3:
            st.markdown("### Recommendation")
            recommendation = get_recommendation(ripeness_level, fruit_type)
            rec_icon = get_recommendation_icon(ripeness_level)
            st.markdown(f"{rec_icon} {recommendation}")
            
            # Add action button or suggestion
            if ripeness_level == "Ripe":
                st.success("Ready to eat")
            elif ripeness_level == "Unripe":
                st.info("Wait before consuming")
            elif ripeness_level == "Overripe":
                st.warning("Cooking recommended")
            else:
                st.info("No specific recommendation")

def get_ripeness_with_percentages(confidence_distribution):
    """Get ripeness level with percentages for all levels"""
    if not confidence_distribution or "error" in confidence_distribution:
        return "Unknown", 0.0, {"Unknown": 1.0}
    
    # Filter out non-confidence keys
    filtered_distribution = {k: v for k, v in confidence_distribution.items() 
                           if k not in ["error", "estimated"]}
    
    if not filtered_distribution:
        return "Unknown", 0.0, {"Unknown": 1.0}
    
    # Find the ripeness level with highest confidence
    ripeness_level, confidence = max(filtered_distribution.items(), key=lambda x: x[1])
    
    # Normalize to just the three main levels if needed
    if ripeness_level not in ["Ripe", "Unripe", "Overripe"]:
        # Map other labels to the three main categories
        if ripeness_level in ["Partially Ripe", "Underripe"]:
            ripeness_level = "Unripe"
        elif ripeness_level in ["Fully Ripe"]:
            ripeness_level = "Ripe"
    
    # Return the main ripeness level, its confidence, and all percentages
    return ripeness_level, confidence, filtered_distribution

def get_ripeness_icon(ripeness_level):
    """Get appropriate icon for ripeness level"""
    if ripeness_level == "Ripe":
        return "✅"
    elif ripeness_level == "Unripe":
        return "🔸"
    elif ripeness_level == "Overripe":
        return "🔶"
    else:
        return "❓"

def get_shelf_life_icon(ripeness_level):
    """Get shelf life icon based on ripeness level"""
    if ripeness_level == "Unripe":
        return "🟢"
    elif ripeness_level == "Ripe":
        return "🟡"
    elif ripeness_level == "Overripe":
        return "🔶"
    else:
        return "⚠️"

def get_recommendation_icon(ripeness_level):
    """Get recommendation icon based on ripeness level"""
    if ripeness_level == "Unripe":
        return "⏳"
    elif ripeness_level == "Ripe":
        return "✅"
    elif ripeness_level == "Overripe":
        return "⚠️"
    else:
        return "❓"

def get_condition_text(ripeness_level, fruit_type):
    """Get condition text based on ripeness level and fruit type"""
    fruit_type = fruit_type.lower()
    
    if ripeness_level == "Unripe":
        if fruit_type == "banana":
            return "Firm texture with green color. Not sweet yet."
        elif fruit_type == "tomato":
            return "Firm with green or light red color. Tart flavor."
        elif fruit_type == "strawberry":
            return "Firm with white or light red areas. Tart flavor."
        elif fruit_type == "mango":
            return "Hard texture with green color. Not sweet yet."
        elif fruit_type == "pineapple":
            return "Firm texture with green exterior. Acidic flavor."
        return "Not ready for optimal consumption."
    
    elif ripeness_level == "Ripe":
        if fruit_type == "banana":
            return "Yellow with slight brown specks. Sweet flavor and soft texture."
        elif fruit_type == "tomato":
            return "Uniform red color with slight give when pressed. Sweet-acidic balance."
        elif fruit_type == "strawberry":
            return "Bright red color with slight give. Sweet flavor."
        elif fruit_type == "mango":
            return "Yields to gentle pressure. Sweet aroma and flavor."
        elif fruit_type == "pineapple":
            return "Golden yellow color with sweet aroma. Sweet-tart balance."
        return "Perfect for consumption now."
    
    elif ripeness_level == "Overripe":
        if fruit_type == "banana":
            return "Brown spots cover most of peel. Very soft and very sweet."
        elif fruit_type == "tomato":
            return "Very soft with possible wrinkles. Less acidic but may be mealy."
        elif fruit_type == "strawberry":
            return "Dark red with soft spots. Very sweet but may be mushy."
        elif fruit_type == "mango":
            return "Very soft with wrinkles. Very sweet but possibly stringy."
        elif fruit_type == "pineapple":
            return "Orange-yellow color with very soft feel. May taste fermented."
        return "Past optimal consumption window."
    
    return "Condition unknown."

def get_shelf_life(ripeness_level, fruit_type):
    """Get shelf life information based on ripeness level and fruit type"""
    if ripeness_level == "Unripe":
        if fruit_type.lower() == "banana":
            return "4-5 days at room temperature"
        elif fruit_type.lower() == "tomato":
            return "7-10 days at room temperature"
        elif fruit_type.lower() == "strawberry":
            return "1-2 days at room temperature"
        elif fruit_type.lower() == "mango":
            return "7-14 days at room temperature"
        elif fruit_type.lower() == "pineapple":
            return "5-7 days at room temperature"
        return "Several days as it ripens"
    
    elif ripeness_level == "Ripe":
        if fruit_type.lower() == "banana":
            return "1-2 days at room temperature"
        elif fruit_type.lower() == "tomato":
            return "2-3 days at room temperature"
        elif fruit_type.lower() == "strawberry":
            return "1-2 days refrigerated"
        elif fruit_type.lower() == "mango":
            return "1-2 days at room temperature"
        elif fruit_type.lower() == "pineapple":
            return "1-2 days at room temperature"
        return "Limited shelf life, consume soon"
    
    elif ripeness_level == "Overripe":
        if fruit_type.lower() == "banana":
            return "Use immediately or freeze"
        elif fruit_type.lower() == "tomato":
            return "Use immediately"
        elif fruit_type.lower() == "strawberry":
            return "Use immediately or freeze"
        elif fruit_type.lower() == "mango":
            return "Use immediately or freeze"
        elif fruit_type.lower() == "pineapple":
            return "Use immediately"
        return "Use immediately"
    
    return "Shelf life unknown"

def get_recommendation(ripeness_level, fruit_type):
    """Get recommendation based on ripeness level and fruit type"""
    if ripeness_level == "Unripe":
        if fruit_type.lower() == "banana":
            return "Wait 2-3 days before eating."
        elif fruit_type.lower() == "tomato":
            return "Place in paper bag to speed ripening."
        elif fruit_type.lower() == "strawberry":
            return "Wait 1-2 days at room temperature."
        elif fruit_type.lower() == "mango":
            return "Place in paper bag to speed ripening."
        elif fruit_type.lower() == "pineapple":
            return "Place upside down to promote even ripening."
        return "Allow to ripen before consuming."
    
    elif ripeness_level == "Ripe":
        if fruit_type.lower() == "banana":
            return "Perfect for eating now."
        elif fruit_type.lower() == "tomato":
            return "Ideal for fresh eating in salads."
        elif fruit_type.lower() == "strawberry":
            return "Perfect for eating now. Refrigerate to extend life."
        elif fruit_type.lower() == "mango":
            return "Ready to eat. Refrigerate to extend life."
        elif fruit_type.lower() == "pineapple":
            return "Perfect for eating fresh now."
        return "Ready for consumption now."
    
    elif ripeness_level == "Overripe":
        if fruit_type.lower() == "banana":
            return "Use for baking or smoothies."
        elif fruit_type.lower() == "tomato":
            return "Use for cooking, sauces or soups."
        elif fruit_type.lower() == "strawberry":
            return "Use for smoothies or jams."
        elif fruit_type.lower() == "mango":
            return "Use for smoothies or purees."
        elif fruit_type.lower() == "pineapple":
            return "Use for smoothies or cooking."
        return "Best used for cooking or processing."
    
    return "No specific recommendation available."

def get_fruit_emoji(fruit_type):
    """Get emoji for fruit type"""
    fruit_type = fruit_type.lower()
    
    if fruit_type == "banana":
        return "🍌"
    elif fruit_type == "tomato":
        return "🍅"
    elif fruit_type == "strawberry":
        return "🍓"
    elif fruit_type == "mango":
        return "🥭"
    elif fruit_type == "pineapple":
        return "🍍"
    
    return "🍎"

def save_user_results(results, username):
    """Save user results with simplified UI"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        save_note = st.text_input("Add note (optional)", key="save_note")
    
    with col2:
        save_button = st.button("💾 Save Results", type="primary", key="save_button")
        
        if save_button:
            # Create serializable copy of results
            save_results = make_serializable(results)
            
            # Add user's note
            save_results["user_note"] = save_note
            
            # Add analysis type info
            save_results["analysis_type"] = "enhanced"
            
            # Save image paths
            image_paths = {}
            timestamp = int(time.time())
            
            # Save original image
            if isinstance(results.get("original_image"), Image.Image):
                original_path = f"results/original_{timestamp}.png"
                results["original_image"].save(original_path)
                image_paths["original"] = original_path
                save_results["original_image_path"] = original_path
            elif "original_image_path" in results:
                image_paths["original"] = results["original_image_path"]
            
            # Save segmented image
            if isinstance(results.get("segmented_image"), Image.Image):
                segmented_path = f"results/segmented_{timestamp}.png"
                results["segmented_image"].save(segmented_path)
                image_paths["segmented"] = segmented_path
                save_results["segmented_image_path"] = segmented_path
            elif "segmented_image_path" in results:
                image_paths["segmented"] = results["segmented_image_path"]
            
            try:
                # Save the results
                result_id = save_enhanced_user_result(username, save_results, image_paths)
                st.success(f"✅ Results saved! (ID: {result_id})")
                
                # Add button to view history
                if st.button("View History"):
                    st.session_state.page = "history"
                    st.rerun()
            except Exception as e:
                st.error(f"Error saving results: {str(e)}")
                
def display_fruit_verification_warning(results, system, username):
    """
    Display warning to user when detected fruit type doesn't match selected fruit type.
    
    Args:
        results: Dictionary containing verification results
        system: FruitRipenessSystem instance
        username: Current username
    """
    st.warning("⚠️ Fruit Type Mismatch Detected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(results["original_image"], use_container_width=True)
    
    with col2:
        st.subheader("Segmented Image")
        st.image(results["segmented_image"], use_container_width=True)
    
    st.markdown(f"### Fruit Type Verification Results", unsafe_allow_html=True)
    
    # Display the mismatch information
    st.markdown(f"""
    - **You selected:** {results['fruit_type_selected'].title()}
    - **Detected fruit:** {results['fruit_type_detected'].title()} (confidence: {results['detection_confidence']:.2f})
    """, unsafe_allow_html=True)
    
    # Display all probabilities as a bar chart
    if 'all_probabilities' in results and results['all_probabilities']:
        probs = results['all_probabilities']
        
        # Sort probabilities by confidence
        sorted_probs = {k: v for k, v in sorted(probs.items(), key=lambda item: item[1], reverse=True)}
        
        st.markdown("### Fruit Classification Probabilities", unsafe_allow_html=True)
        
        chart_data = {
            "Fruit Type": list(sorted_probs.keys()),
            "Confidence": list(sorted_probs.values())
        }
        
        # Display as a table
        st.table({
            "Fruit Type": [k.title() for k in sorted_probs.keys()],
            "Confidence": [f"{v:.2%}" for v in sorted_probs.values()]
        })
    
    # Provide options to the user
    st.markdown("### What would you like to do?", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Proceed Anyway", type="primary"):
            # Proceed with original fruit type
            st.session_state.verified_fruit_type = results['fruit_type_selected']
            st.session_state.ignore_verification = True
            st.rerun()
    
    with col2:
        if st.button(f"Switch to {results['fruit_type_detected'].title()}", type="primary"):
            # Switch to detected fruit type
            st.session_state.selected_fruit = results['fruit_type_detected'].title()
            st.session_state.verified_fruit_type = results['fruit_type_detected']
            st.session_state.analysis_step = "analyze"  # Stay on analysis page
            st.rerun()
    
    with col3:
        if st.button("Cancel Analysis", type="secondary"):
            # Go back to fruit selection
            st.session_state.selected_fruit = None
            st.session_state.analysis_step = "select_fruit"
            st.session_state.uploaded_file = None
            st.session_state.camera_image = None
            st.rerun()
            
def combine_multi_angle_results(results_list):
    """
    Combine results from multiple angles of the same fruit with improved consistency
    
    Args:
        results_list: List of results dictionaries from different angles
        
    Returns:
        Combined results dictionary
    """
    
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        return {"error": "No valid results to combine"}
    
    combined = valid_results[0].copy()

    # IMPROVED: Standardize confidence distribution across angles
    if "confidence_distributions" in valid_results[0]:
        print("Using enhanced confidence distribution combining method")
        
        # Initialize lists to collect all confidence distributions
        all_distributions = []
        all_weights = []
        
        # Gather confidence distributions from all angles
        for result in valid_results:
            if "confidence_distributions" in result and result["confidence_distributions"]:
                for distribution in result["confidence_distributions"]:
                    # Skip distributions with errors
                    if "error" in distribution:
                        continue
                        
                    # Remove metadata keys
                    clean_distribution = {k: v for k, v in distribution.items() 
                                         if k not in ["error", "estimated"]}
                    
                    if clean_distribution:
                        all_distributions.append(clean_distribution)
                        
                        # Use max confidence as weight for this distribution
                        max_confidence = max(clean_distribution.values())
                        all_weights.append(max_confidence)
        
        # Combine distributions using weighted average
        if all_distributions:
            # Normalize weights
            if sum(all_weights) > 0:
                normalized_weights = [w/sum(all_weights) for w in all_weights]
            else:
                normalized_weights = [1.0/len(all_weights)] * len(all_weights)
            
            # Get all unique ripeness categories
            all_categories = set()
            for dist in all_distributions:
                all_categories.update(dist.keys())
            
            # Calculate weighted average for each category
            combined_distribution = {}
            for category in all_categories:
                weighted_sum = 0
                for dist, weight in zip(all_distributions, normalized_weights):
                    weighted_sum += dist.get(category, 0) * weight
                combined_distribution[category] = weighted_sum
            
            # Normalize the combined distribution
            total = sum(combined_distribution.values())
            if total > 0:
                combined_distribution = {k: v/total for k, v in combined_distribution.items()}
            
            # Create prediction format expected by display functions
            combined_predictions = []
            for ripeness, confidence in sorted(combined_distribution.items(), 
                                               key=lambda x: x[1], reverse=True):
                combined_predictions.append({
                    "ripeness": ripeness,
                    "confidence": confidence
                })
            
            combined["ripeness_predictions"] = combined_predictions
            combined["confidence_distributions"] = [combined_distribution]
        else:
            # Fallback to old method if no distributions found
            print("No valid confidence distributions found, using legacy method")
            ripeness_predictions = {}
            for r in valid_results:
                if "ripeness_predictions" in r and r["ripeness_predictions"]:
                    for pred in r["ripeness_predictions"]:
                        ripeness = pred["ripeness"]
                        confidence = pred["confidence"]
                        
                        if ripeness in ripeness_predictions:
                            ripeness_predictions[ripeness].append(confidence)
                        else:
                            ripeness_predictions[ripeness] = [confidence]
            
            combined_predictions = []
            for ripeness, confidences in ripeness_predictions.items():
                avg_confidence = sum(confidences) / len(confidences)
                combined_predictions.append({
                    "ripeness": ripeness, 
                    "confidence": avg_confidence
                })
            
            combined_predictions = sorted(combined_predictions, key=lambda x: x["confidence"], reverse=True)
            combined["ripeness_predictions"] = combined_predictions
    else:
        # Legacy method for old-style results
        ripeness_predictions = {}
        for r in valid_results:
            if "ripeness_predictions" in r and r["ripeness_predictions"]:
                for pred in r["ripeness_predictions"]:
                    ripeness = pred["ripeness"]
                    confidence = pred["confidence"]
                    
                    if ripeness in ripeness_predictions:
                        ripeness_predictions[ripeness].append(confidence)
                    else:
                        ripeness_predictions[ripeness] = [confidence]
        
        combined_predictions = []
        for ripeness, confidences in ripeness_predictions.items():
            avg_confidence = sum(confidences) / len(confidences)
            combined_predictions.append({
                "ripeness": ripeness,
                "confidence": avg_confidence
            })
        
        combined_predictions = sorted(combined_predictions, key=lambda x: x["confidence"], reverse=True)
        combined["ripeness_predictions"] = combined_predictions
    
    # Add multi-angle metadata
    combined["multi_angle"] = True
    combined["num_angles"] = len(valid_results)
    combined["angle_results"] = valid_results
    combined["angle_names"] = [r.get("angle_name", "Unknown") for r in valid_results]
    
    return combined

def process_angle_image(system, image_file, fruit_type, angle_name, use_segmentation, refine_segmentation, refinement_method, use_enhanced_analysis=False):
    """Process an image for a specific angle in patch-based analysis"""
    try:
        if use_enhanced_analysis:
            results = system.analyze_ripeness_enhanced(
                image_file,
                fruit_type=fruit_type,
                is_patch_based=True,  # Flag as patch-based but still use object detection
                use_segmentation=use_segmentation  # Pass the user's segmentation preference
            )
            # Add angle name to results
            results["angle_name"] = angle_name
            return results
        elif use_segmentation:
            results = system.process_image_with_visualization(
                image_file,
                fruit_type=fruit_type,
                refine_segmentation=refine_segmentation,
                refinement_method=refinement_method,
                angle_name=angle_name
            )
        else:
            results = system.process_image_without_segmentation(
                image_file,
                fruit_type=fruit_type,
                angle_name=angle_name
            )
        
        return results
    except Exception as e:
        st.error(f"Error processing {angle_name} image: {str(e)}")
        return None
    
def display_results(results, system, use_segmentation, username):
    if "error" in results and "original_image" not in results:
        st.error(f"Error: {results['error']}")
    else:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(results["original_image"], use_container_width=True)
        
        with col2:
            if use_segmentation:
                st.subheader("Segmented Fruit")
                st.image(results["segmented_image"], use_container_width=True)
                st.markdown(
                    get_image_download_link(
                        results["segmented_image"],
                        "segmented_fruit.png",
                        "Download Segmented Image"
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.subheader("Processed Image (No Segmentation)")
                
                if "ripeness_predictions" in results and results["ripeness_predictions"]:
                    
                    if "visualizations" in results and "bounding_box_visualization" in results["visualizations"]:
                        bbox_path = results["visualizations"]["bounding_box_visualization"]
                        try:
                            bbox_img = Image.open(bbox_path)
                            st.image(bbox_img, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not load bounding box visualization: {str(e)}")
                            st.image(results["original_image"], use_container_width=True)
                    else:
                        st.image(results["original_image"], use_container_width=True)
                else:
                    st.image(results["original_image"], use_container_width=True)
        
        
        st.subheader("Fruit Analysis")
        st.write(f"**Selected Fruit:** {results['fruit_type'].title()}")
        
        
        st.subheader("Ripeness Detection")
        
        if "ripeness_predictions" in results and results["ripeness_predictions"]:
            ripeness_data = {
                "Ripeness Level": [],
                "Confidence": []
            }
            
            for pred in results["ripeness_predictions"]:
                ripeness_data["Ripeness Level"].append(pred["ripeness"])
                ripeness_data["Confidence"].append(f"{pred['confidence']:.2f}")
                
            st.table(ripeness_data)

            if "visualizations" in results:
                st.subheader("Enhanced Visualizations")
                
                # Show combined visualization if available
                if "combined_visualization" in results["visualizations"]:
                    combined_path = results["visualizations"]["combined_visualization"]
                    try:
                        combined_img = Image.open(combined_path)
                        st.image(combined_img, use_container_width=True)
                        
                        st.markdown(
                            get_image_download_link(
                                combined_img, 
                                "complete_analysis.png",
                                "Download Complete Visualization"
                            ),
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.warning(f"Could not load combined visualization: {str(e)}")
                
                # Create tabs for different visualization types
                viz_tab1, viz_tab2 = st.tabs(["Bounding Box Detection", "Comparison View"])
                
                with viz_tab1:
                    if "bounding_box_visualization" in results["visualizations"]:
                        bbox_path = results["visualizations"]["bounding_box_visualization"]
                        try:
                            bbox_img = Image.open(bbox_path)
                            st.image(bbox_img, use_container_width=True)
                            
                            st.markdown(
                                get_image_download_link(
                                    bbox_img, 
                                    "ripeness_detection.png",
                                    "Download Bounding Box Visualization"
                                ),
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.warning(f"Could not load bounding box visualization: {str(e)}")
                    else:
                        st.info("Bounding box visualization not available")
                
                with viz_tab2:
                    if "comparison_visualization" in results["visualizations"]:
                        comparison_path = results["visualizations"]["comparison_visualization"]
                        try:
                            comparison_img = Image.open(comparison_path)
                            st.image(comparison_img, use_container_width=True)
                            
                            st.markdown(
                                get_image_download_link(
                                    comparison_img, 
                                    "comparison_result.png",
                                    "Download Comparison Visualization"
                                ),
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.warning(f"Could not load comparison visualization: {str(e)}")
                    else:
                        st.info("Comparison visualization not available")
            
        elif "error" in results:
            st.warning(f"Ripeness detection error: {results['error']}")
        
        with st.expander("Technical Details"):
            if use_segmentation:
                st.write("**Segmentation Mask:**")
                st.image(results["mask"] * 255, clamp=True, use_container_width=True)
                
                if "mask_metrics" in results:
                    st.write("**Mask Quality Metrics:**")
                    metrics = results["mask_metrics"]
                    st.write(f"- Mask coverage: {metrics['coverage_ratio']:.2%} of image")
                    st.write(f"- Boundary complexity: {metrics['boundary_complexity']:.2f}")
                    
                # Add Model Comparison section if available
                if "comparison_metrics" in results:
                    st.write("---")
                    st.subheader("🔍 Base U-Net vs SPEAR-UNet Comparison")
                    
                    comparison = results["comparison_metrics"]
                    
                    # Performance Metrics
                    st.write("**Performance Comparison:**")
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.metric("Base U-Net Time", f"{comparison['baseline_time']*1000:.1f} ms")
                    
                    with perf_col2:
                        st.metric("SPEAR-UNet Time", f"{comparison['enhanced_time']*1000:.1f} ms")
                    
                    with perf_col3:
                        speedup = comparison['speedup']
                        st.metric("Speedup", f"{speedup:.2f}x", f"{(speedup-1)*100:.1f}%")
                    
                    # Segmentation Quality Metrics
                    st.write("**Segmentation Quality Comparison:**")
                    qual_col1, qual_col2, qual_col3 = st.columns(3)
                    
                    with qual_col1:
                        st.metric("IoU between models", f"{comparison['iou']:.2f}")
                    
                    
                    with qual_col2:
                        st.metric("Boundary Detail", f"{comparison['enhanced_complexity']:.2f}", 
                                f"{comparison['enhanced_complexity'] - comparison['baseline_complexity']:.2f} vs Base")
                    
                    # Side by side image comparison
                    st.write("**Visual Segmentation Comparison:**")
                    base_col, enhanced_col = st.columns(2)
                    
                    with base_col:
                        st.write("Base U-Net Result")
                        st.image(comparison["baseline_segmented_image"], use_container_width=True)
                    
                    with enhanced_col:
                        st.write("SPEAR-UNet Result")
                        st.image(results["segmented_image"], use_container_width=True)
                    
                    # Neural Network Visualizations
                    st.write("---")
                    st.subheader("🧠 Neural Network Layer Visualizations")
                    
                    tab1, tab2, tab3 = st.tabs(["Feature Maps", "Layer-by-Layer Comparison", "Regularization"])
                    
                    with tab1:
                        if "feature_maps" in results:
                            for name, viz in results["feature_maps"].items():
                                st.write(f"**{name}:**")
                                st.image(viz, use_container_width=True)
                    
                    with tab2:
                        if "visualizations" in results:
                            for layer_name, viz in results["visualizations"].items():
                                # Skip non-layer visualizations
                                if layer_name in ["bounding_box_visualization", "comparison_visualization", "combined_visualization"]:
                                    continue
                                    
                                st.write(f"**{layer_name} Layer Comparison:**")
                                st.image(viz, use_container_width=True)
                                
                                # Add metrics table for this layer
                                if "layer_metrics" in comparison and layer_name in comparison["layer_metrics"]:
                                    layer_metric = comparison["layer_metrics"][layer_name]
                                    
                                    metric_data = {
                                        "Metric": [
                                            "Standard Deviation (Objective 3: ResNet Integration)", 
                                            "Feature Entropy (Objective 1: Stochastic Feature Pyramid)",
                                            "Mean Activation (Objective 3: ResNet Integration)"
                                        ],
                                        "Base U-Net": [
                                            f"{layer_metric['baseline']['std_activation']:.4f}",
                                            f"{layer_metric['baseline']['entropy']:.4f}",
                                            f"{layer_metric['baseline']['mean_activation']:.4f}"
                                        ],
                                        "SPEAR-UNet": [
                                            f"{layer_metric['enhanced']['std_activation']:.4f}",
                                            f"{layer_metric['enhanced']['entropy']:.4f}",
                                            f"{layer_metric['enhanced']['mean_activation']:.4f}"
                                        ],
                                        "Improvement": [
                                            f"{(layer_metric['enhanced']['std_activation'] - layer_metric['baseline']['std_activation']) / abs(layer_metric['baseline']['std_activation'] + 1e-8) * 100:.1f}%",
                                            f"{layer_metric['entropy_improvement']:.1f}%",
                                            f"{(layer_metric['enhanced']['mean_activation'] - layer_metric['baseline']['mean_activation']) / abs(layer_metric['baseline']['mean_activation'] + 1e-8) * 100:.1f}%"
                                        ]
                                    }
                                    
                                    st.table(metric_data)
                    
                    with tab3:
                        if "regularization_comparison_viz" in results and results["regularization_comparison_viz"] is not None:
                            st.write("**Impact of Dynamic Regularization:**")
                            st.image(results["regularization_comparison_viz"], use_container_width=True)
                            
                            st.write("""
                            The chart above shows the L1 regularization values applied by the dynamic regularization modules.
                            Higher values indicate stronger regularization, which helps prevent overfitting.
                            """)
                    
                    # Explanation of architectural improvements
                    st.write("---")
                    st.subheader("📚 SPEAR-UNet Architecture Improvements")
                    
                    st.write("""
                    1. **Stochastic Feature Pyramid (SFP)** - Improves multi-scale feature extraction
                    2. **Dynamic Regularization** - Adaptive regularization to prevent overfitting
                    3. **ResNet Integration** - Leverages pretrained ResNet-50 weights for better gradient flow
                    """)
            else:
                st.write("**Note:** Segmentation was disabled for this analysis.")
                
            st.write("**Model Processing Information:**")
            st.write(f"- Device used: {system.device}")
            st.write(f"- Segmentation: {'Enabled' if use_segmentation else 'Disabled'}")
            if use_segmentation:
                st.write(f"- Segmentation model input size: 256x256")
                st.write(f"- Mask refinement: {'Enabled' if 'refine_segmentation' in results and results['refine_segmentation'] else 'Disabled'}")
                if "refine_segmentation" in results and results["refine_segmentation"]:
                    st.write(f"- Refinement method: {results.get('refinement_method', 'all')}")
            st.write(f"- Classifier model input size: 224x224")
            
            if "raw_results" in results:
                st.write("**Raw Ripeness Detection Results:**")
                st.json(results["raw_results"])
    
    if username and username != "guest":
        save_col1, save_col2 = st.columns([3, 1])
        
        with save_col1:
            save_note = st.text_input("Add a note about this analysis (optional)")
        
        with save_col2:
            save_button = st.button("💾 Save Analysis Results", type="primary")
            
            if save_button:
                # Prepare a copy of results for saving
                save_results = results.copy()
                
                # Add user's note
                save_results["user_note"] = save_note
                
                # Add segmentation info
                save_results["segmentation"] = use_segmentation
                
                # IMPROVED IMAGE PATH SAVING
                image_paths = {}
                
                # Make sure to save the original image
                if isinstance(results["original_image"], Image.Image):
                    # Save original image if it's a PIL Image
                    original_path = f"results/original_{int(time.time())}.png"
                    results["original_image"].save(original_path)
                    image_paths["original"] = original_path
                    save_results["original_image_path"] = original_path
                elif "original_image_path" in results:
                    # Use existing path if available
                    image_paths["original"] = results["original_image_path"] 
                
                # Segmented image
                if "segmented_image" in results and isinstance(results["segmented_image"], Image.Image):
                    segmented_path = f"results/segmented_{int(time.time())}.png"
                    results["segmented_image"].save(segmented_path)
                    image_paths["segmented"] = segmented_path
                    save_results["segmented_image_path"] = segmented_path
                elif "segmented_image_path" in results:
                    image_paths["segmented"] = results["segmented_image_path"]
                
                # Visualizations
                if "visualizations" in results:
                    for viz_key, viz_path in results["visualizations"].items():
                        if os.path.exists(viz_path):
                            image_paths[viz_key] = viz_path
                
                # Save the results
                try:
                    result_id = save_user_result(username, save_results, image_paths)
                    
                    st.success(f"✅ Analysis results saved successfully! (ID: {result_id})")
                    
                    # Add button to view history
                    if st.button("View Saved Results"):
                        st.session_state.page = "history"
                        st.rerun()
                except Exception as e:
                    import traceback
                    st.error(f"Error saving results: {str(e)}")
                    st.error(traceback.format_exc())
    else:
        st.info("💡 Log in with a user account to save your analysis results for future reference.")

def display_enhanced_results(results, system, username):
    """Display enhanced ripeness analysis results with confidence distributions"""
    if "error" in results and "fruits_data" not in results:
        st.error(f"Error: {results['error']}")
        return
    
    # Merge layer visualizations to make them available
    if ("visualizations" in results and 
        "segmentation_results" in results and 
        "visualizations" in results["segmentation_results"]):
        
        # Define layer visualization keys
        layer_keys = ["Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4", 
                      "Bottleneck", "Decoder 4", "Decoder 3", "Decoder 2", "Decoder 1"]
        
        # Copy layer visualizations to the main visualizations dict
        for key in layer_keys:
            if key in results["segmentation_results"]["visualizations"]:
                results["visualizations"][key] = results["segmentation_results"]["visualizations"][key]
                
        print(f"Merged layer visualizations. Updated keys: {list(results['visualizations'].keys())}")
    
    print("==== DISPLAY FUNCTION VISUALIZATIONS CHECK ====")
    if "visualizations" in results:
        print(f"Visualizations keys in results: {list(results['visualizations'].keys())}")
    else:
        print("No 'visualizations' key in results dictionary")
        
    use_segmentation = results.get("use_segmentation", True)
    fruit_type = results.get("fruit_type", "Unknown")
    num_fruits = results.get("num_fruits", 0)
    
    # Display header info
    st.header(f"🍎 {fruit_type.title()} Ripeness Analysis")
    st.write(f"Detected {num_fruits} fruit{'s' if num_fruits != 1 else ''} in the image")
    
    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image with Detection")
        
        if "visualizations" in results and "bounding_box_visualization" in results["visualizations"]:
            bbox_path = results["visualizations"]["bounding_box_visualization"]
            bbox_img = Image.open(bbox_path)
            st.image(bbox_img, use_container_width=True)
            
            # Add download link for bounding box visualization
            st.markdown(
                get_image_download_link(
                    bbox_img,
                    "detection_result.png",
                    "Download Detection Result"
                ),
                unsafe_allow_html=True
            )
        else:
            # If no bounding box visualization, show original image
            st.image(results["original_image"], use_container_width=True)
    
    with col2:
        if use_segmentation:
            st.subheader("Segmented Image")
            st.image(results["segmented_image"], use_container_width=True)
            
            # Add download link
            st.markdown(
                get_image_download_link(
                    results["segmented_image"],
                    "segmented_fruit.png",
                    "Download Segmented Image"
                ),
                unsafe_allow_html=True
            )
        else:
            st.subheader("Processed Image (No Segmentation)")
            st.image(results["original_image"], use_container_width=True)
    
    # Handle multiple fruits
    if num_fruits > 1:
        # Create visualization showing all fruits
        all_fruits_viz_path = visualize_all_fruits_confidence(results)
        
        # Display the combined visualization
        st.subheader("All Fruits Ripeness Analysis")
        all_fruits_img = Image.open(all_fruits_viz_path)
        st.image(all_fruits_img, use_container_width=True)
        
        # Create tabs for each individual fruit
        st.subheader("Individual Fruit Analysis")
        fruit_tabs = st.tabs([f"Fruit #{i+1}" for i in range(num_fruits)])
        
        for i, (tab, fruit_data) in enumerate(zip(fruit_tabs, results.get("fruits_data", []))):
            with tab:
                confidence_distribution = results.get("confidence_distributions", [])[i] if i < len(results.get("confidence_distributions", [])) else {}
                
                # Display fruit image
                if "masked_crop_path" in fruit_data and os.path.exists(fruit_data["masked_crop_path"]):
                    st.subheader(f"Fruit #{i+1} Image")
                    fruit_img = Image.open(fruit_data["masked_crop_path"])
                    st.image(fruit_img, width=300)
                
                # Display confidence distribution
                if confidence_distribution and "error" not in confidence_distribution:
                    # Generate visualization
                    viz_path = visualize_confidence_distribution(
                        fruit_data, confidence_distribution, fruit_type
                    )
                    
                    # Display visualization
                    viz_img = Image.open(viz_path)
                    st.image(viz_img, use_container_width=True)
                    
                    # Add download link
                    st.markdown(
                        get_image_download_link(
                            viz_img,
                            f"fruit_{i+1}_ripeness.png",
                            f"Download Fruit #{i+1} Visualization"
                        ),
                        unsafe_allow_html=True
                    )
                    
                # Show detailed values
                with st.expander(f"Detailed Confidence Values for Fruit #{i+1}"):
                    if confidence_distribution and "error" not in confidence_distribution:
                        # Filter out non-confidence keys
                        filtered_distribution = {k: v for k, v in confidence_distribution.items() 
                                              if k not in ["error", "estimated"]}
                        
                        # Create table
                        confidence_data = {
                            "Ripeness Level": list(filtered_distribution.keys()),
                            "Confidence": [f"{v:.4f}" for v in filtered_distribution.values()],
                            "Percentage": [f"{v*100:.1f}%" for v in filtered_distribution.values()]
                        }
                        
                        st.table(confidence_data)
                    else:
                        st.write("No confidence distribution data available.")
    else:
        # Single fruit display
        confidence_distribution = results.get("confidence_distributions", [])[0] if results.get("confidence_distributions") else {}
        fruit_data = results.get("fruits_data", [])[0] if results.get("fruits_data") else {}
        
        st.subheader("Ripeness Analysis")
        
        # Generate visualization
        if confidence_distribution and "error" not in confidence_distribution:
            viz_path = visualize_confidence_distribution(
                fruit_data, confidence_distribution, fruit_type
            )
            
            # Display visualization
            viz_img = Image.open(viz_path)
            st.image(viz_img, use_container_width=True)
            
            # Add download link
            st.markdown(
                get_image_download_link(
                    viz_img,
                    "ripeness_distribution.png",
                    "Download Ripeness Visualization"
                ),
                unsafe_allow_html=True
            )
            
            # Display confidence breakdown
            st.subheader("Ripeness Confidence Breakdown")
            
            # Filter out non-confidence keys
            filtered_distribution = {k: v for k, v in confidence_distribution.items() 
                                  if k not in ["error", "estimated"]}
            
            # Create table
            confidence_data = {
                "Ripeness Level": list(filtered_distribution.keys()),
                "Confidence": [f"{v:.4f}" for v in filtered_distribution.values()],
                "Percentage": [f"{v*100:.1f}%" for v in filtered_distribution.values()]
            }
            
            st.table(confidence_data)
        else:
            st.warning("No confidence distribution data available.")
    
    # Technical details
    with st.expander("Technical Details"):
        if use_segmentation:
            st.write("**Segmentation Mask:**")
            st.image(results["mask"] * 255, clamp=True, use_container_width=True)
            
            if "segmentation_results" in results and "mask_metrics" in results["segmentation_results"]:
                st.write("**Mask Quality Metrics:**")
                metrics = results["segmentation_results"]["mask_metrics"]
                st.write(f"- Mask coverage: {metrics['coverage_ratio']:.2%} of image")
                st.write(f"- Boundary complexity: {metrics['boundary_complexity']:.2f}")
            elif "mask_metrics" in results:
                st.write("**Mask Quality Metrics:**")
                metrics = results["mask_metrics"]
                st.write(f"- Mask coverage: {metrics['coverage_ratio']:.2%} of image")
                st.write(f"- Boundary complexity: {metrics['boundary_complexity']:.2f}")
        else:
            st.write("**Note:** Segmentation was disabled for this analysis.")
            
        # Add model information
        st.write("**Model Processing Information:**")
        st.write(f"- Device used: {system.device}")
        st.write(f"- Two-Stage Analysis: Enabled")
        st.write(f"- Segmentation: {'Enabled' if use_segmentation else 'Disabled'}")
        st.write(f"- Segmentation model input size: 256x256")
        
        if "classification_results" in results:
            if isinstance(results["classification_results"], list):
                if len(results["classification_results"]) > 0:
                    # If it's a list, use the first item if available
                    first_result = results["classification_results"][0]
                    if isinstance(first_result, dict):
                        st.write(f"- Classification model: {first_result.get('model_name', 'Custom Model')}")
                        st.write(f"- Classification model input size: {first_result.get('input_size', '224x224')}")
                    else:
                        st.write("- Classification model: Custom Model")
                        st.write("- Classification model input size: 224x224")
                else:
                    st.write("- Classification model: Custom Model")
                    st.write("- Classification model input size: 224x224")
            elif isinstance(results["classification_results"], dict):
                # If it's a dictionary, use it directly
                st.write(f"- Classification model: {results['classification_results'].get('model_name', 'Custom Model')}")
                st.write(f"- Classification model input size: {results['classification_results'].get('input_size', '224x224')}")
            else:
                # Fallback if it's neither list nor dict
                st.write("- Classification model: Custom Model")
                st.write("- Classification model input size: 224x224")
        else:
            st.write("- Classification model: Custom Model")
            st.write("- Classification model input size: 224x224")
        
        # Add raw classification results if available
        for i, distribution in enumerate(results.get("confidence_distributions", [])):
            if distribution and "error" not in distribution:
                st.write(f"**Raw Classification Results (Fruit #{i+1}):**")
                st.json(distribution)
                
        # Neural Network Analysis Tabs
        st.write("---")
        st.subheader("🧠 Neural Network Analysis")
        
        # Get the comparison metrics from the correct location
        comparison = None
        if "comparison_metrics" in results:
            comparison = results["comparison_metrics"]
        elif "segmentation_results" in results and "comparison_metrics" in results["segmentation_results"]:
            comparison = results["segmentation_results"]["comparison_metrics"]
            
        if comparison:
            # Performance Metrics
            st.write("**Performance Comparison: Base U-Net vs SPEAR-UNet**")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("Base U-Net Time", f"{comparison['baseline_time']*1000:.1f} ms")
            
            with perf_col2:
                st.metric("SPEAR-UNet Time", f"{comparison['enhanced_time']*1000:.1f} ms")
            
            with perf_col3:
                speedup = comparison['speedup']
                st.metric("Speedup", f"{speedup:.2f}x", f"{(speedup-1)*100:.1f}%")
            
            # Segmentation Quality Metrics
            st.write("**Segmentation Quality Comparison:**")
            qual_col1, qual_col2, qual_col3 = st.columns(3)
            
            with qual_col1:
                st.metric("IoU between models", f"{comparison['iou']:.2f}")
            
            with qual_col2:
                st.metric("Boundary Detail", f"{comparison['enhanced_complexity']:.2f}", 
                        f"{comparison['enhanced_complexity'] - comparison['baseline_complexity']:.2f} vs Base")
            
            # Side by side image comparison
            st.write("**Visual Segmentation Comparison:**")
            base_col, enhanced_col = st.columns(2)
            
            with base_col:
                st.write("Base U-Net Result")
                if "baseline_segmented_image" in comparison:
                    st.image(comparison["baseline_segmented_image"], use_container_width=True)
                else:
                    st.write("Base U-Net image not available")
            
            with enhanced_col:
                st.write("SPEAR-UNet Result")
                st.image(results["segmented_image"], use_container_width=True)
        
        # Create tabs for different visualization types
        tab1, tab2, tab3 = st.tabs(["Feature Maps", "Layer-by-Layer Comparison", "Regularization"])
            
        with tab1:
            # Check both locations for feature maps
            feature_maps = None
            if "feature_maps" in results:
                feature_maps = results["feature_maps"]
            elif "segmentation_results" in results and "feature_maps" in results["segmentation_results"]:
                feature_maps = results["segmentation_results"]["feature_maps"]
                
            if feature_maps:
                for name, viz in feature_maps.items():
                    st.write(f"**{name}:**")
                    st.image(viz, use_container_width=True)
            else:
                st.write("No feature map visualizations available")

        with tab2:
            layer_vis = None
            layer_keys = ["Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4", 
                        "Bottleneck", "Decoder 4", "Decoder 3", "Decoder 2", "Decoder 1"]
                
            if "visualizations" in results:
                vis_keys = [k for k in results["visualizations"].keys() if k in layer_keys]
                if vis_keys:
                    layer_vis = results["visualizations"]
            
            # If not found, also check in segmentation_results directly as a fallback
            if not layer_vis and "segmentation_results" in results and "visualizations" in results["segmentation_results"]:
                vis_keys = [k for k in results["segmentation_results"]["visualizations"].keys() if k in layer_keys]
                if vis_keys:
                    layer_vis = results["segmentation_results"]["visualizations"]
                    
            if comparison and "layer_metrics" in comparison:
                st.subheader("Layer Metrics Comparison")
                
                # Define the layer order (encoder to decoder)
                layer_order = [
                    "Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4", 
                    "Bottleneck", 
                    "Decoder 4", "Decoder 3", "Decoder 2", "Decoder 1"
                ]
                
                # Filter and sort layers based on the defined order
                available_layers = []
                for layer in layer_order:
                    if layer in comparison["layer_metrics"]:
                        available_layers.append(layer)
                
                # Create 3 separate tables - one for each metric type
                st.markdown("### Standard Deviation", unsafe_allow_html=True)
                std_table = {
                    "Layer": available_layers,
                    "Base U-Net": [],
                    "SPEAR-UNet": [],
                    "Improvement": []
                }
                
                for layer in available_layers:
                    layer_metric = comparison["layer_metrics"][layer]
                    # Add Standard Deviation
                    base_std = layer_metric['baseline']['std_activation']
                    spear_std = layer_metric['enhanced']['std_activation']
                    std_improvement = ((spear_std - base_std) / abs(base_std + 1e-8)) * 100
                    
                    std_table["Base U-Net"].append(f"{base_std:.4f}")
                    std_table["SPEAR-UNet"].append(f"{spear_std:.4f}")
                    
                    # Add improvement with arrow indicators
                    if std_improvement > 0:
                        std_table["Improvement"].append(f"↑ {std_improvement:.1f}%")
                    else:
                        std_table["Improvement"].append(f"↓ {std_improvement:.1f}%")
                
                st.table(std_table)
                
                st.markdown("### Feature Entropy", unsafe_allow_html=True)
                entropy_table = {
                    "Layer": available_layers,
                    "Base U-Net": [],
                    "SPEAR-UNet": [],
                    "Improvement": []
                }
                
                for layer in available_layers:
                    layer_metric = comparison["layer_metrics"][layer]
                    # Add Feature Entropy
                    base_entropy = layer_metric['baseline']['entropy']
                    spear_entropy = layer_metric['enhanced']['entropy']
                    entropy_improvement = layer_metric['entropy_improvement']
                    
                    entropy_table["Base U-Net"].append(f"{base_entropy:.4f}")
                    entropy_table["SPEAR-UNet"].append(f"{spear_entropy:.4f}")
                    
                    # Add improvement with arrow indicators
                    if entropy_improvement > 0:
                        entropy_table["Improvement"].append(f"↑ {entropy_improvement:.1f}%")
                    else:
                        entropy_table["Improvement"].append(f"↓ {entropy_improvement:.1f}%")
                
                st.table(entropy_table)
                
                st.markdown("### Mean Activation", unsafe_allow_html=True)
                mean_table = {
                    "Layer": available_layers,
                    "Base U-Net": [],
                    "SPEAR-UNet": [],
                    "Improvement": []
                }
                
                for layer in available_layers:
                    layer_metric = comparison["layer_metrics"][layer]
                    # Add Mean Activation
                    base_mean = layer_metric['baseline']['mean_activation']
                    spear_mean = layer_metric['enhanced']['mean_activation']
                    mean_improvement = ((spear_mean - base_mean) / abs(base_mean + 1e-8)) * 100
                    
                    mean_table["Base U-Net"].append(f"{base_mean:.4f}")
                    mean_table["SPEAR-UNet"].append(f"{spear_mean:.4f}")
                    
                    # Add improvement with arrow indicators
                    if mean_improvement > 0:
                        mean_table["Improvement"].append(f"↑ {mean_improvement:.1f}%")
                    else:
                        mean_table["Improvement"].append(f"↓ {mean_improvement:.1f}%")
                
                st.table(mean_table)
                
                # Calculate overall average improvements
                avg_std_improvement = sum([((comparison["layer_metrics"][layer]['enhanced']['std_activation'] - 
                                        comparison["layer_metrics"][layer]['baseline']['std_activation']) / 
                                        abs(comparison["layer_metrics"][layer]['baseline']['std_activation'] + 1e-8)) * 100 
                                    for layer in available_layers]) / len(available_layers)
                
                avg_entropy_improvement = sum([comparison["layer_metrics"][layer]['entropy_improvement'] 
                                            for layer in available_layers]) / len(available_layers)
                
                avg_mean_improvement = sum([((comparison["layer_metrics"][layer]['enhanced']['mean_activation'] - 
                                        comparison["layer_metrics"][layer]['baseline']['mean_activation']) / 
                                        abs(comparison["layer_metrics"][layer]['baseline']['mean_activation'] + 1e-8)) * 100 
                                        for layer in available_layers]) / len(available_layers)
                
                # Show average improvements
                st.markdown("### Overall Average Improvements", unsafe_allow_html=True)
                avg_table = {
                    "Metric": ["Standard Deviation", "Feature Entropy", "Mean Activation"],
                    "Average Improvement": [
                        f"{avg_std_improvement:.1f}%" if avg_std_improvement > 0 else f"{avg_std_improvement:.1f}%",
                        f"{avg_entropy_improvement:.1f}%" if avg_entropy_improvement > 0 else f"{avg_entropy_improvement:.1f}%",
                        f"{avg_mean_improvement:.1f}%" if avg_mean_improvement > 0 else f"{avg_mean_improvement:.1f}%"
                    ]
                }
                st.table(avg_table)
                
                st.markdown("**What do these metrics mean?**", unsafe_allow_html=True)
                st.markdown("""
                - **Standard Deviation**: Measures the variability of activations. Higher values in SPEAR-UNet indicate better feature discrimination.
                - **Feature Entropy**: Quantifies information content in the feature maps. Higher entropy in SPEAR-UNet demonstrates improved information capture.
                - **Mean Activation**: Average activation level. Differences show how ResNet integration affects feature response.
                - **Improvement**: Percentage improvement of SPEAR-UNet over Base U-Net. Positive values indicate better performance.
                """, unsafe_allow_html=True)
                    
            # If we found layer visualizations, display them
            if layer_vis:
                vis_keys = [k for k in layer_vis.keys() if k in layer_keys]
                if vis_keys:
                    st.subheader("Layer Visualizations")
                    for key in vis_keys:
                        st.write(f"**{key} Layer Comparison:**")
                        st.image(layer_vis[key], use_container_width=True)
                        
                        # Add metrics table for this layer if available (keeping individual metrics with the visualizations)
                        if comparison and "layer_metrics" in comparison and key in comparison["layer_metrics"]:
                            layer_metric = comparison["layer_metrics"][key]
                            
                            metric_data = {
                                "Metric": [
                                    "Standard Deviation (Objective 1 and 3: ResNet Integration)", 
                                    "Feature Entropy (Objective 1: Stochastic Feature Pyramid)",
                                    "Mean Activation (Objective 3: ResNet Integration)"
                                ],
                                "Base U-Net": [
                                    f"{layer_metric['baseline']['std_activation']:.4f}",
                                    f"{layer_metric['baseline']['entropy']:.4f}",
                                    f"{layer_metric['baseline']['mean_activation']:.4f}"
                                ],
                                "SPEAR-UNet": [
                                    f"{layer_metric['enhanced']['std_activation']:.4f}",
                                    f"{layer_metric['enhanced']['entropy']:.4f}",
                                    f"{layer_metric['enhanced']['mean_activation']:.4f}"
                                ],
                                "Improvement": [
                                    f"{(layer_metric['enhanced']['std_activation'] - layer_metric['baseline']['std_activation']) / abs(layer_metric['baseline']['std_activation'] + 1e-8) * 100:.1f}%",
                                    f"{layer_metric['entropy_improvement']:.1f}%",
                                    f"{(layer_metric['enhanced']['mean_activation'] - layer_metric['baseline']['mean_activation']) / abs(layer_metric['baseline']['mean_activation'] + 1e-8) * 100:.1f}%"
                                ]
                            }
                            
                            st.table(metric_data)
                else:
                    st.write("No layer-by-layer visualizations available")
            else:
                st.write("No layer-by-layer visualizations available")
        
        with tab3:
            # Get regularization visualization
            reg_viz = None
            if "regularization_comparison_viz" in results:
                reg_viz = results["regularization_comparison_viz"]
            elif "segmentation_results" in results and "regularization_comparison_viz" in results["segmentation_results"]:
                reg_viz = results["segmentation_results"]["regularization_comparison_viz"]

            if reg_viz is not None:
                st.write("**Impact of Dynamic Regularization:**")
                st.image(reg_viz, use_container_width=True)
                
                st.write("""
                The chart above shows the L1 regularization values applied by the dynamic regularization modules.
                Higher values indicate stronger regularization, which helps prevent overfitting.
                """)
            else:
                st.write("No regularization visualization available")
        
        # Explanation of architectural improvements
        st.write("---")
        st.subheader("📚 SPEAR-UNet Architecture Improvements")
        
        st.write("""
        1. **Stochastic Feature Pyramid (SFP)** - Improves multi-scale feature extraction
        2. **Dynamic Regularization** - Adaptive regularization to prevent overfitting
        3. **ResNet Integration** - Leverages pretrained ResNet-50 weights for better gradient flow
        """)
    
    st.subheader("Save Results")

    if username and username != "guest":
        save_col1, save_col2 = st.columns([3, 1])
        
        with save_col1:
            save_note = st.text_input("Add a note about this analysis (optional)", key="enhanced_save_note")
        
        with save_col2:
            save_button = st.button("💾 Save Analysis Results", type="primary", key="enhanced_save_button")
            
            if save_button:
                # Create serializable copy of results
                save_results = make_serializable(results)
                
                # Add user's note
                save_results["user_note"] = save_note
                
                # Add analysis type info
                save_results["analysis_type"] = "enhanced_two_stage"
                
                # IMPROVED IMAGE PATH SAVING
                image_paths = {}
                timestamp = int(time.time())
                
                # Save original image
                if isinstance(results.get("original_image"), Image.Image):
                    original_path = f"results/original_{timestamp}.png"
                    results["original_image"].save(original_path)
                    image_paths["original"] = original_path
                    save_results["original_image_path"] = original_path
                elif "original_image_path" in results:
                    image_paths["original"] = results["original_image_path"]
                
                # Save segmented image
                if isinstance(results.get("segmented_image"), Image.Image):
                    segmented_path = f"results/segmented_{timestamp}.png"
                    results["segmented_image"].save(segmented_path)
                    image_paths["segmented"] = segmented_path
                    save_results["segmented_image_path"] = segmented_path
                elif "segmented_image_path" in results:
                    image_paths["segmented"] = results["segmented_image_path"]
                
                # Save bounding box visualization
                if "visualizations" in results and "bounding_box_visualization" in results["visualizations"]:
                    bbox_path = results["visualizations"]["bounding_box_visualization"]
                    if os.path.exists(bbox_path):
                        image_paths["bounding_box"] = bbox_path
                
                # Save confidence distribution visualizations
                for i, (fruit_data, distribution) in enumerate(zip(
                    results.get("fruits_data", []),
                    results.get("confidence_distributions", [])
                )):
                    if distribution and "error" not in distribution:
                        try:
                            viz_path = visualize_confidence_distribution(
                                fruit_data, distribution, fruit_type, 
                                save_path=f"results/fruit_{i+1}_dist_{timestamp}.png"
                            )
                            image_paths[f"fruit_{i+1}_distribution"] = viz_path
                        except Exception as e:
                            st.warning(f"Could not save visualization for fruit #{i+1}: {str(e)}")
                
                # Save all fruits visualization if multiple fruits
                if num_fruits > 1:
                    try:
                        all_viz_path = visualize_all_fruits_confidence(
                            results, 
                            save_path=f"results/all_fruits_dist_{timestamp}.png"
                        )
                        image_paths["all_fruits_distribution"] = all_viz_path
                    except Exception as e:
                        st.warning(f"Could not save combined visualization: {str(e)}")
                
                try:
                    # Save the results
                    result_id = save_enhanced_user_result(username, save_results, image_paths)
                    st.success(f"✅ Analysis results saved successfully! (ID: {result_id})")
                    
                    # Add button to view history
                    if st.button("View Saved Results", key="enhanced_view_history"):
                        st.session_state.page = "history"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error saving results: {str(e)}")
                    # Add detailed error output for debugging
                    import traceback
                    st.text(traceback.format_exc())
    else:
        st.info("💡 Log in with a user account to save your analysis results for future reference.")

def display_patch_based_results(combined_results, system, use_segmentation, username):
    """Display results for patch-based analysis, aligned with display_results"""
    if "error" in combined_results and "angle_results" not in combined_results:
        st.error(f"Error: {combined_results['error']}")
        return
    
    # Header section similar to display_results
    st.header(f"🍎 {combined_results['fruit_type'].title()} Ripeness Analysis")
    st.info(f"Analyzed {combined_results['num_angles']} angles of the fruit")
    
    # Main content columns - similar structure to display_results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image (Front View)")
        if "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
            st.image(combined_results["angle_results"][0]["original_image"], use_container_width=True)
    
    with col2:
        if use_segmentation:
            st.subheader("Segmented Image (Front View)")
            if "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
                st.image(combined_results["angle_results"][0]["segmented_image"], use_container_width=True)
        else:
            st.subheader("Processed Image (Front View)")
            if "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
                if "visualizations" in combined_results["angle_results"][0] and "bounding_box_visualization" in combined_results["angle_results"][0]["visualizations"]:
                    bbox_path = combined_results["angle_results"][0]["visualizations"]["bounding_box_visualization"]
                    try:
                        bbox_img = Image.open(bbox_path)
                        st.image(bbox_img, use_container_width=True)
                    except:
                        st.image(combined_results["angle_results"][0]["original_image"], use_container_width=True)
                else:
                    st.image(combined_results["angle_results"][0]["original_image"], use_container_width=True)
    
    # Ripeness detection section - similar to display_results but for combined results
    st.subheader("Combined Ripeness Detection")
    
    if "ripeness_predictions" in combined_results and combined_results["ripeness_predictions"]:
        ripeness_data = {
            "Ripeness Level": [],
            "Average Confidence": []
        }
        
        for pred in combined_results["ripeness_predictions"]:
            ripeness_data["Ripeness Level"].append(pred["ripeness"])
            ripeness_data["Average Confidence"].append(f"{pred['confidence']:.2f}")
        
        st.table(ripeness_data)
    elif "error" in combined_results:
        st.warning(f"Ripeness detection error: {combined_results['error']}")
    
    # Angle-specific tabs - unique to patch-based
    st.subheader("Angle-Specific Analysis")
    angle_tabs = st.tabs(combined_results["angle_names"])
    
    for i, (tab, angle_result) in enumerate(zip(angle_tabs, combined_results["angle_results"])):
        with tab:
            angle_col1, angle_col2 = st.columns(2)
            
            with angle_col1:
                st.subheader(f"Original {combined_results['angle_names'][i]}")
                st.image(angle_result["original_image"], use_container_width=True)
            
            with angle_col2:
                if use_segmentation:
                    st.subheader(f"Segmented {combined_results['angle_names'][i]}")
                    st.image(angle_result["segmented_image"], use_container_width=True)
                else:
                    st.subheader(f"Processed {combined_results['angle_names'][i]}")
                    if "visualizations" in angle_result and "bounding_box_visualization" in angle_result["visualizations"]:
                        bbox_path = angle_result["visualizations"]["bounding_box_visualization"]
                        try:
                            bbox_img = Image.open(bbox_path)
                            st.image(bbox_img, use_container_width=True)
                        except:
                            st.image(angle_result["original_image"], use_container_width=True)
                    else:
                        st.image(angle_result["original_image"], use_container_width=True)
            
            # Angle-specific ripeness predictions
            st.subheader(f"{combined_results['angle_names'][i]} Ripeness")
            if "ripeness_predictions" in angle_result and angle_result["ripeness_predictions"]:
                angle_ripeness_data = {
                    "Ripeness Level": [],
                    "Confidence": []
                }
                
                for pred in angle_result["ripeness_predictions"]:
                    angle_ripeness_data["Ripeness Level"].append(pred["ripeness"])
                    angle_ripeness_data["Confidence"].append(f"{pred['confidence']:.2f}")
                
                st.table(angle_ripeness_data)
    
    # Technical details section - similar to display_results
    with st.expander("Technical Details"):
        if use_segmentation and "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
            st.write("**Segmentation Mask (Front View):**")
            st.image(combined_results["angle_results"][0]["mask"] * 255, clamp=True, use_container_width=True)
            
            if "mask_metrics" in combined_results["angle_results"][0]:
                st.write("**Mask Quality Metrics:**")
                metrics = combined_results["angle_results"][0]["mask_metrics"]
                st.write(f"- Mask coverage: {metrics['coverage_ratio']:.2%} of image")
                st.write(f"- Boundary complexity: {metrics['boundary_complexity']:.2f}")
        
        st.write("**Model Processing Information:**")
        st.write(f"- Device used: {system.device}")
        st.write(f"- Segmentation: {'Enabled' if use_segmentation else 'Disabled'}")
        if use_segmentation:
            st.write(f"- Segmentation model input size: 256x256")
            st.write(f"- Mask refinement: {'Enabled' if 'refine_segmentation' in combined_results and combined_results['refine_segmentation'] else 'Disabled'}")
            if "refine_segmentation" in combined_results and combined_results['refine_segmentation']:
                st.write(f"- Refinement method: {combined_results.get('refinement_method', 'all')}")
        st.write(f"- Classifier model input size: 224x224")
        
        if "raw_results" in combined_results:
            st.write("**Raw Ripeness Detection Results:**")
            st.json(combined_results["raw_results"])
    
    # Save results section - similar to display_results
    if username and username != "guest":
        save_col1, save_col2 = st.columns([3, 1])
        
        with save_col1:
            save_note = st.text_input("Add a note about this analysis (optional)")
        
        with save_col2:
            save_button = st.button("💾 Save Analysis Results", type="primary")
            
            if save_button:
                save_results = combined_results.copy()
                save_results["user_note"] = save_note
                save_results["segmentation"] = use_segmentation
                
                image_paths = {}
                
                # Save images from first angle as representative
                if "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
                    first_angle = combined_results["angle_results"][0]
                    
                    if isinstance(first_angle.get("original_image"), Image.Image):
                        original_path = f"results/original_{int(time.time())}.png"
                        first_angle["original_image"].save(original_path)
                        image_paths["original"] = original_path
                    
                    if use_segmentation and isinstance(first_angle.get("segmented_image"), Image.Image):
                        segmented_path = f"results/segmented_{int(time.time())}.png"
                        first_angle["segmented_image"].save(segmented_path)
                        image_paths["segmented"] = segmented_path
                
                try:
                    result_id = save_user_result(username, save_results, image_paths)
                    st.success(f"✅ Analysis results saved successfully! (ID: {result_id})")
                    
                    if st.button("View Saved Results"):
                        st.session_state.page = "history"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error saving results: {str(e)}")
    else:
        st.info("💡 Log in with a user account to save your analysis results for future reference.")

def display_enhanced_patch_based_results(combined_results, system, username):
    """Display enhanced patch-based results, aligned with display_enhanced_results"""
    if "error" in combined_results and "angle_results" not in combined_results:
        st.error(f"Error: {combined_results['error']}")
        return
    
    use_segmentation = combined_results.get("use_segmentation", True)
                                            
    fruit_type = combined_results.get("fruit_type", "Unknown")
    num_angles = combined_results.get("num_angles", 0)
    
    # Header section similar to display_enhanced_results
    st.header(f"🍎 {fruit_type.title()} Ripeness Analysis")
    st.write(f"Analyzed {num_angles} angles with enhanced two-stage analysis")
    
    # Main content columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image (Front View)")
        if "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
            st.image(combined_results["angle_results"][0]["original_image"], use_container_width=True)
    
    with col2:
        if use_segmentation:
            st.subheader("Segmented Image (Front View)")
            if "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
                st.image(combined_results["angle_results"][0]["segmented_image"], use_container_width=True)
        else:
            st.subheader("Processed Image (No Segmentation)")
            if "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
                if "visualizations" in combined_results["angle_results"][0] and "bounding_box_visualization" in combined_results["angle_results"][0]["visualizations"]:
                    bbox_path = combined_results["angle_results"][0]["visualizations"]["bounding_box_visualization"]
                    try:
                        bbox_img = Image.open(bbox_path)
                        st.image(bbox_img, use_container_width=True)
                    except:
                        st.image(combined_results["angle_results"][0]["original_image"], use_container_width=True)
                else:
                    st.image(combined_results["angle_results"][0]["original_image"], use_container_width=True)
        
    # Combined ripeness analysis
    st.subheader("Combined Ripeness Analysis")
    
    if "ripeness_predictions" in combined_results and combined_results["ripeness_predictions"]:
        ripeness_data = {
            "Ripeness Level": [],
            "Average Confidence": []
        }
        
        for pred in combined_results["ripeness_predictions"]:
            ripeness_data["Ripeness Level"].append(pred["ripeness"])
            ripeness_data["Average Confidence"].append(f"{pred['confidence']:.2f}")
        
        st.table(ripeness_data)
    
    # Angle-specific tabs
    st.subheader("Angle-Specific Analysis")
    angle_tabs = st.tabs(combined_results["angle_names"])
    
    for i, (tab, angle_result) in enumerate(zip(angle_tabs, combined_results["angle_results"])):
        with tab:
            angle_name = combined_results["angle_names"][i]
            
            # Display images for this angle
            angle_col1, angle_col2 = st.columns(2)
            
            with angle_col1:
                st.subheader(f"Original {angle_name}")
                st.image(angle_result["original_image"], use_container_width=True)
            
            with angle_col2:
                if use_segmentation:
                    st.subheader(f"Segmented {angle_name}")
                    st.image(angle_result["segmented_image"], use_container_width=True)
                else:
                    st.subheader(f"Processed {angle_name}")
                    if "visualizations" in angle_result and "bounding_box_visualization" in angle_result["visualizations"]:
                        bbox_path = angle_result["visualizations"]["bounding_box_visualization"]
                        try:
                            bbox_img = Image.open(bbox_path)
                            st.image(bbox_img, use_container_width=True)
                        except:
                            st.image(angle_result["original_image"], use_container_width=True)
                    else:
                        st.image(angle_result["original_image"], use_container_width=True)
            
            # Display confidence distributions for this angle
            if "confidence_distributions" in angle_result:
                st.subheader(f"{angle_name} Confidence Distribution")
                
                if len(angle_result["confidence_distributions"]) > 1:
                    # Multiple fruits in this angle
                    fruit_tabs = st.tabs([f"Fruit #{j+1}" for j in range(len(angle_result["confidence_distributions"]))])
                    
                    for j, (fruit_tab, distribution) in enumerate(zip(fruit_tabs, angle_result["confidence_distributions"])):
                        with fruit_tab:
                            if distribution and "error" not in distribution:
                                viz_path = visualize_confidence_distribution(
                                    {},  # Pass empty dict since we don't have fruit_data here
                                    distribution,
                                    fruit_type
                                )
                                viz_img = Image.open(viz_path)
                                st.image(viz_img, use_container_width=True)
                else:
                    # Single fruit in this angle
                    distribution = angle_result["confidence_distributions"][0]
                    if distribution and "error" not in distribution:
                        viz_path = visualize_confidence_distribution(
                            {},
                            distribution,
                            fruit_type
                        )
                        viz_img = Image.open(viz_path)
                        st.image(viz_img, use_container_width=True)
    
    with st.expander("Technical Details"):
        if use_segmentation and "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
            st.write("**Segmentation Mask (Front View):**")
            st.image(combined_results["angle_results"][0]["mask"] * 255, clamp=True, use_container_width=True)
            
            if "mask_metrics" in combined_results["angle_results"][0]:
                st.write("**Mask Quality Metrics:**")
                metrics = combined_results["angle_results"][0]["mask_metrics"]
                st.write(f"- Mask coverage: {metrics['coverage_ratio']:.2%} of image")
                st.write(f"- Boundary complexity: {metrics['boundary_complexity']:.2f}")
        else:
            st.write("**Note:** Segmentation was disabled for this analysis.")
        
        st.write("**Model Processing Information:**")
        st.write(f"- Device used: {system.device}")
        st.write(f"- Two-Stage Analysis: Enabled")
        st.write(f"- Segmentation: {'Enabled' if use_segmentation else 'Disabled'}")
        st.write(f"- Segmentation model input size: 256x256")
        st.write(f"- Classification model input size: 224x224")
    
    # Save results section - similar to display_enhanced_results
    if username and username != "guest":
        save_col1, save_col2 = st.columns([3, 1])
        
        with save_col1:
            save_note = st.text_input("Add a note about this analysis (optional)", key="enhanced_patch_note")
        
        with save_col2:
            save_button = st.button("💾 Save Analysis Results", type="primary", key="enhanced_patch_save")
            
            if save_button:
                save_results = make_serializable(combined_results)
                save_results["user_note"] = save_note
                save_results["analysis_type"] = "enhanced_patch_based"
                
                image_paths = {}
                timestamp = int(time.time())
                
                # Save representative images from first angle
                if "angle_results" in combined_results and len(combined_results["angle_results"]) > 0:
                    first_angle = combined_results["angle_results"][0]
                    
                    if isinstance(first_angle.get("original_image"), Image.Image):
                        original_path = f"results/original_{timestamp}.png"
                        first_angle["original_image"].save(original_path)
                        image_paths["original"] = original_path
                    
                    if isinstance(first_angle.get("segmented_image"), Image.Image):
                        segmented_path = f"results/segmented_{timestamp}.png"
                        first_angle["segmented_image"].save(segmented_path)
                        image_paths["segmented"] = segmented_path
                
                try:
                    result_id = save_enhanced_user_result(username, save_results, image_paths)
                    st.success(f"✅ Analysis results saved successfully! (ID: {result_id})")
                    
                    if st.button("View Saved Results", key="enhanced_patch_history"):
                        st.session_state.page = "history"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error saving results: {str(e)}")
    else:
        st.info("💡 Log in with a user account to save your analysis results for future reference.")

def main():
    fruit_icons = {
        "Tomato": "🍅",
        "Pineapple": "🍍",
        "Banana": "🍌",
        "Strawberry": "🍓",
        "Mango": "🥭"
    }

    with st.sidebar:
        st.title("🍎 Fruit Ripeness")
        
        current_user = get_current_user()
        
        if current_user:
            st.write(f"👋 Welcome, {current_user}!")
            
            st.subheader("Navigation")
            nav_options = ["Home", "Analysis History", "About"]
            
            nav_map = {
                "Home": "home",
                "Analysis History": "history",
                "About": "about"
            }
            
            for nav_option in nav_options:
                if st.button(nav_option, use_container_width=True, 
                            type="primary" if st.session_state.page == nav_map[nav_option] else "secondary"):
                    st.session_state.page = nav_map[nav_option]
                    if nav_map[nav_option] != "history":
                        st.session_state.view_details = False
                        st.session_state.selected_result_id = None
                    
                    if nav_map[nav_option] == "home":
                        st.session_state.analysis_step = "select_fruit"
                        st.session_state.selected_fruit = None
                        st.session_state.uploaded_file = None
                        st.session_state.camera_image = None
                        st.session_state.start_analysis = False
                        st.session_state.show_top = False
                        st.session_state.show_bottom = False
                        
                        # Also clear these if they exist
                        if "front_file" in st.session_state:
                            del st.session_state.front_file
                        if "back_file" in st.session_state:
                            del st.session_state.back_file
                        if "top_file" in st.session_state:
                            del st.session_state.top_file
                        if "bottom_file" in st.session_state:
                            del st.session_state.bottom_file
                        
                        # Clear verification-related state
                        for key in ["verification_results", "verified_fruit_type", "ignore_verification"]:
                            if key in st.session_state:
                                del st.session_state[key]
                    
                    st.rerun()
            
            if st.button("Logout", use_container_width=True):
                for key in ["logged_in", "username", "page"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        st.divider()    
        st.caption("Fruit Ripeness Detection System v1.2")
        st.caption("© 2025 FruitThesis Industries")
    
    if not st.session_state.logged_in:
        if show_login_page():
            st.session_state.page = "home"
            st.rerun()
        return
    
    system = load_models()
    
    # Initialize fruit verifier if needed
    if not hasattr(system, 'fruit_verifier') and hasattr(system, 'init_fruit_verifier'):
        system.init_fruit_verifier()
    
    username = get_current_user()
    
    if st.session_state.page == "history":
        if st.session_state.view_details and st.session_state.selected_result_id:
            # Show detailed view of a single result
            show_result_details(username, st.session_state.selected_result_id)
        else:
            show_history_page(username)
            
    elif st.session_state.page == "about":
        # About page
        st.title("About Fruit Ripeness Detection System")
        
        st.write("""
        ## How It Works
        
        This fruit ripeness detection system uses a three-stage approach:
        
        1. **Segmentation**: A UNet-ResNet50 based model called SPEAR-net identifies and isolates the fruit in the image
        2. **Fruit Selection**: User selects the type of fruit to analyze
        3. **Ripeness Detection**: A specialized model for each fruit type determines its ripeness level
        
        The segmentation improves ripeness detection by focusing only on the fruit
        and removing background distractions.
        
        ## Supported Fruits
        
        Currently, the system can detect ripeness for:
        - Tomatoes
        - Pineapples
        - Bananas
        - Strawberries
        - Mangoes
        
        ## Multi-Angle Analysis
        
        For more accurate results, you can upload images of the same fruit from multiple angles.
        The system analyzes each angle separately and combines the results for a comprehensive assessment.
        """)
        
    elif st.session_state.page == "evaluation":
        add_evaluation_page_to_app()
        
    else:
        st.title("🍎 Fruit Ripeness Detection System")
        
        system = load_models()
        
        st.write("""
        This application detects the ripeness of fruits in images. 
        Follow the guided process to analyze your fruit!
        """)
        
        st.write("""
        Make sure there is proper lighting in your photo for more accurate results.
        Additionally, keep the fruit in focus and center, and avoid any obstructions.
        """)
        
        if "show_top" not in st.session_state:
            st.session_state.show_top = False
        if "show_bottom" not in st.session_state:
            st.session_state.show_bottom = False
        if "start_analysis" not in st.session_state:
            st.session_state.start_analysis = False
        
        fruit_options = ["Tomato", "Pineapple", "Banana", "Strawberry", "Mango"]
        
        # Step 1: Select fruit type
        if st.session_state.analysis_step == "select_fruit":
            st.header("Step 1: Select Your Fruit")
            
            # Create a visually appealing fruit selection layout
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                tomato_card = st.container()
                with tomato_card:
                    st.markdown(f"### {fruit_icons['Tomato']} Tomato", unsafe_allow_html=True)
                    if st.button("Select Tomato", key="tomato_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Tomato"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
            
            with col2:
                pineapple_card = st.container()
                with pineapple_card:
                    st.markdown(f"### {fruit_icons['Pineapple']} Pineapple", unsafe_allow_html=True)
                    if st.button("Select Pineapple", key="pineapple_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Pineapple"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
            
            with col3:
                banana_card = st.container()
                with banana_card:
                    st.markdown(f"### {fruit_icons['Banana']} Banana", unsafe_allow_html=True)
                    if st.button("Select Banana", key="banana_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Banana"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
            
            with col4:
                strawberry_card = st.container()
                with strawberry_card:
                    st.markdown(f"### {fruit_icons['Strawberry']} Strawberry", unsafe_allow_html=True)
                    if st.button("Select Strawberry", key="strawberry_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Strawberry"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
            
            with col5:
                mango_card = st.container()
                with mango_card:
                    st.markdown(f"### {fruit_icons['Mango']} Mango", unsafe_allow_html=True)
                    if st.button("Select Mango", key="mango_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Mango"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
        
        elif st.session_state.analysis_step == "upload_image":
            st.header(f"Step 2: Upload {st.session_state.selected_fruit} Image")
            
            if st.button("← Back to Fruit Selection", key="back_to_fruit"):
                st.session_state.selected_fruit = None
                st.session_state.analysis_step = "select_fruit"
                st.session_state.uploaded_file = None
                st.session_state.camera_image = None
                st.rerun()
            
            st.write(f"You selected: {fruit_icons.get(st.session_state.selected_fruit, '🍎')} **{st.session_state.selected_fruit}**")
            
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Single Image", "Patch-Based (Multiple Angles)"],
                horizontal=True,
                help="Single Image: Analyze one image. Patch-Based: Analyze multiple angles of the same fruit."
            )
            
            with st.expander("Advanced Options"):
                use_segmentation = st.checkbox("Use Segmentation", value=True, 
                                            help="Segment the fruit from the background before analysis. Disable this to send the original image directly to the ripeness detector.")
                
                # ADD ENHANCED ANALYSIS OPTION
                use_enhanced_analysis = st.checkbox("Use Enhanced Two-Stage Analysis", value=True,
                                                help="Use the two-stage analysis approach with detailed confidence distributions")
                
                # ADD FRUIT VERIFICATION OPTION
                verify_fruit_type = st.checkbox("Enable Fruit Type Verification", value=False,
                                             help="Verify that the uploaded image contains the selected fruit type")
                
                if use_segmentation:
                    refine_segmentation = st.checkbox("Refine Segmentation Mask", value=True, 
                                                help="Apply post-processing to improve the segmentation mask")
                    
                    refinement_method = st.radio(
                        "Refinement Method",
                        ["all", "morphological", "contour", "boundary"],
                        help="Select the method to refine the segmentation mask"
                    )
                else:
                    refine_segmentation = False
                    refinement_method = "all"
                
                # Save to session state
                st.session_state.use_segmentation = use_segmentation
                st.session_state.refine_segmentation = refine_segmentation
                st.session_state.refinement_method = refinement_method
                st.session_state.use_enhanced_analysis = use_enhanced_analysis
                st.session_state.verify_fruit_type = verify_fruit_type
            
            if analysis_type == "Single Image":
                st.write("### Upload Your Image")
                
                upload_container = st.container()
                with upload_container:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        # Upload single image
                        uploaded_file = st.file_uploader(
                            "Choose an image...", 
                            type=["jpg", "jpeg", "png"], 
                            key="single_uploader",
                            help="Drag and drop your image here or click to browse"
                        )
                        
                        # Reset uploaded file in session state when file uploader is cleared
                        if uploaded_file is None:
                            st.session_state.uploaded_file = None
                        
                        use_camera = st.checkbox("Use Camera Instead", key="use_camera_checkbox")
                        
                        if use_camera:
                            camera_image = st.camera_input("Take a picture", key="camera_input")
                            if camera_image is not None:
                                st.session_state.camera_image = camera_image
                                st.session_state.uploaded_file = None
                        else:
                            st.session_state.camera_image = None
                            if uploaded_file is not None:
                                st.session_state.uploaded_file = uploaded_file
                    
                    # Only show preview if we have a valid file in session state
                    if (st.session_state.get("uploaded_file") is not None or 
                        st.session_state.get("camera_image") is not None):
                        
                        preview_container = st.container()
                        with preview_container:
                            st.subheader("Image Preview")
                            
                            prev_col1, prev_col2, prev_col3 = st.columns([1, 2, 1])
                            
                            with prev_col2:
                                image_input = st.session_state.camera_image if st.session_state.camera_image is not None else st.session_state.uploaded_file
                                st.image(image_input, use_container_width=True)
                    
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                    with col2:
                        # Check if we have either uploaded file or camera image
                        has_valid_image = (st.session_state.get("uploaded_file") is not None or 
                                        st.session_state.get("camera_image") is not None)
                        
                        # Only enable button if we have an image
                        if st.button("🔍 Start Analysis", 
                                type="primary", 
                                use_container_width=True,
                                disabled=not has_valid_image):
                            st.session_state.analysis_step = "analyze"
                            st.rerun()
                        
                        if not has_valid_image:
                            st.info("Please upload an image or take a photo to proceed")
            
            elif analysis_type == "Patch-Based (Multiple Angles)":
                st.write(f"Upload images of the {st.session_state.selected_fruit} from different angles:")
                
                # Initialize patch-based variables
                front_file = None
                back_file = None
                top_file = None
                bottom_file = None
                
                # Create a more attractive layout for the uploaders
                st.write("### Required Views")
                
                # Front and back views (required)
                front_col, back_col = st.columns(2)
                
                with front_col:
                    st.markdown("**🔄 Front View**", unsafe_allow_html=True)
                    front_file = st.file_uploader("Upload front view", type=["jpg", "jpeg", "png"], key="front")
                    if front_file:
                        # Create a smaller preview
                        st.image(front_file, width=250)
                    
                with back_col:
                    st.markdown("**🔄 Back View**", unsafe_allow_html=True)
                    back_file = st.file_uploader("Upload back view", type=["jpg", "jpeg", "png"], key="back")
                    if back_file:
                        # Create a smaller preview
                        st.image(back_file, width=250)
                
                # Add a divider before optional views
                if front_file is not None and back_file is not None:
                    st.write("---")
                    st.write("### Optional Views")
                
                # Option to add top view
                if front_file is not None and back_file is not None and not st.session_state.show_top:
                    show_top_view = st.button("+ Add Top View")
                    if show_top_view:
                        st.session_state.show_top = True
                
                # Show top view uploader if requested
                if st.session_state.show_top:
                    top_col, add_bottom_col = st.columns([3, 1])
                    
                    with top_col:
                        st.markdown("**⬆️ Top View**", unsafe_allow_html=True)
                        top_file = st.file_uploader("Upload top view", type=["jpg", "jpeg", "png"], key="top")
                        if top_file:
                            # Create a smaller preview
                            st.image(top_file, width=250)
                    
                    with add_bottom_col:
                        if not st.session_state.show_bottom and top_file is not None:
                            show_bottom_view = st.button("+ Add Bottom View")
                            if show_bottom_view:
                                st.session_state.show_bottom = True
                
                # Show bottom view uploader if requested
                if st.session_state.show_bottom:
                    st.markdown("**⬇️ Bottom View**", unsafe_allow_html=True)
                    bottom_file = st.file_uploader("Upload bottom view", type=["jpg", "jpeg", "png"], key="bottom")
                    if bottom_file:
                        # Create a smaller preview
                        st.image(bottom_file, width=250)
                
                if front_file is not None and back_file is not None:
                    st.session_state.front_file = front_file
                    st.session_state.back_file = back_file
                    st.session_state.top_file = top_file
                    st.session_state.bottom_file = bottom_file
                    
                    # Add start analysis button
                    st.write("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔍 Start Multi-Angle Analysis", type="primary", use_container_width=True):
                            st.session_state.analysis_step = "analyze"
                            st.session_state.start_analysis = True
                            st.rerun()
                else:
                    # Show message that both front and back images are required
                    if (front_file is not None or back_file is not None) and not (front_file is not None and back_file is not None):
                        st.warning("Both front and back images are required for multi-angle analysis")
        
        elif st.session_state.analysis_step == "analyze":
            st.header(f"Analyzing {st.session_state.selected_fruit}")
        
            if st.button("← Start New Analysis", key="new_analysis"):
                st.session_state.selected_fruit = None
                st.session_state.analysis_step = "select_fruit"
                st.session_state.uploaded_file = None
                st.session_state.camera_image = None
                st.session_state.start_analysis = False
                st.session_state.show_top = False
                st.session_state.show_bottom = False
                
                for key in ["verification_results", "verified_fruit_type", "ignore_verification"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            
            username = get_current_user()

            if "verification_results" in st.session_state:
                from fruit_verification import display_verification_warning
                display_verification_warning(st.session_state.verification_results)
                return 
            
            if st.session_state.uploaded_file is not None or st.session_state.camera_image is not None:
                # Single image analysis
                image_input = st.session_state.camera_image if st.session_state.camera_image is not None else st.session_state.uploaded_file
                
                if image_input is not None:
                    use_segmentation = st.session_state.use_segmentation
                    refine_segmentation = st.session_state.refine_segmentation
                    refinement_method = st.session_state.refinement_method
                    use_enhanced_analysis = st.session_state.use_enhanced_analysis
                    verify_fruit_type = st.session_state.get("verify_fruit_type", True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Check if we need to verify fruit type
                    skip_verification = (not verify_fruit_type or 
                                         'verified_fruit_type' in st.session_state or 
                                         'ignore_verification' in st.session_state)
                    
                    if not skip_verification and use_segmentation and hasattr(system, 'verify_fruit_type'):
                        # Perform fruit type verification
                        status_text.text("Performing initial segmentation for fruit verification...")
                        progress_bar.progress(15)
                        
                        # Load and preprocess image
                        img = Image.open(image_input).convert('RGB')
                        timestamp = int(time.time())
                        temp_path = f"results/verification_image_{timestamp}.png"
                        img.save(temp_path)
                        
                        # Segment the image
                        try:
                            seg_results = system.segment_fruit_with_metrics(temp_path)
                            segmented_img = seg_results["segmented_image"]
                            
                            progress_bar.progress(30)
                            status_text.text("Verifying fruit type...")
                            
                            # Verify fruit type
                            selected_type = st.session_state.selected_fruit.lower()
                            verification_results = system.verify_fruit_type(segmented_img, selected_type)
                            
                            # If mismatch detected, show warning
                            if not verification_results["is_match"]:
                                progress_bar.progress(40)
                                status_text.text("Fruit type mismatch detected. Waiting for user input...")
                                
                                # Prepare results for UI
                                verification_results.update({
                                    "fruit_type_selected": selected_type,
                                    "original_image": img,
                                    "segmented_image": segmented_img
                                })
                                
                                # Store in session state
                                st.session_state.verification_results = verification_results
                                st.rerun()  # Show verification UI
                        except Exception as e:
                            # Log error but continue
                            print(f"Error during fruit verification: {str(e)}")
                            # Continue with normal processing
                    
                    # Continue with normal analysis
                    if use_enhanced_analysis:
                        status_text.text(f"Processing {st.session_state.selected_fruit} using two-stage analysis...")
                        progress_bar.progress(25)

                        results = system.analyze_ripeness_enhanced(
                            image_input,
                            fruit_type=st.session_state.selected_fruit.lower(),
                            use_segmentation=use_segmentation
                        )
                        
                        if "warning" in results and results["warning"] == "fruit_type_mismatch":
                            st.session_state.verification_results = {
                                "fruit_type_selected": results["fruit_type_selected"],
                                "detected_type": results["fruit_type_detected"],
                                "confidence": results["detection_confidence"],
                                "all_probabilities": results["all_probabilities"],
                                "original_image": results["original_image"],
                                "segmented_image": results["segmented_image"],
                                "message": results["suggestion"]
                            }
                            st.rerun()
                            
                        if "warning" in results and results["warning"] == "fruit_type_mismatch":
                            # Store verification results in session state
                            st.session_state.verification_results = {
                                "fruit_type_selected": results["fruit_type_selected"],
                                "detected_type": results["fruit_type_detected"],
                                "confidence": results["detection_confidence"],
                                "all_probabilities": results["all_probabilities"],
                                "original_image": results["original_image"],
                                "segmented_image": results["segmented_image"],
                                "message": results["suggestion"]
                            }
                            st.rerun()
                        
                        progress_bar.progress(100)
                        status_text.text("Enhanced analysis complete!")
                        
                        display_card_ripeness_results(results, system, username)
                    elif use_segmentation:
                        status_text.text(f"Processing {st.session_state.selected_fruit} image with segmentation...")
                        progress_bar.progress(25)
                        
                        results = system.process_image_with_visualization(
                            image_input,
                            fruit_type=st.session_state.selected_fruit.lower(),
                            refine_segmentation=refine_segmentation,
                            refinement_method=refinement_method
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                        
                        display_results(results, system, use_segmentation, username)
                    else:
                        status_text.text(f"Processing {st.session_state.selected_fruit} image without segmentation...")
                        progress_bar.progress(25)
                        
                        results = system.process_image_without_segmentation(
                            image_input,
                            fruit_type=st.session_state.selected_fruit.lower()
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                        
                        display_results(results, system, use_segmentation, username)
            
            elif st.session_state.start_analysis and "front_file" in st.session_state and "back_file" in st.session_state:
                # Multi-angle analysis
                front_file = st.session_state.front_file
                back_file = st.session_state.back_file
                top_file = st.session_state.top_file if "top_file" in st.session_state else None
                bottom_file = st.session_state.bottom_file if "bottom_file" in st.session_state else None
                
                use_segmentation = st.session_state.use_segmentation
                refine_segmentation = st.session_state.refine_segmentation
                refinement_method = st.session_state.refinement_method
                use_enhanced_analysis = st.session_state.use_enhanced_analysis
                verify_fruit_type = st.session_state.get("verify_fruit_type", True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Check if we need to verify fruit type for patch-based analysis
                skip_verification = (not verify_fruit_type or 
                                     'verified_fruit_type' in st.session_state or 
                                     'ignore_verification' in st.session_state)
                
                # Verify front image only if verification is enabled
                if not skip_verification and use_segmentation and hasattr(system, 'verify_fruit_type'):
                    # Verify fruit type from front view
                    status_text.text("Verifying fruit type from front view...")
                    progress_bar.progress(10)
                    
                    # Load and preprocess image
                    img = Image.open(front_file).convert('RGB')
                    timestamp = int(time.time())
                    temp_path = f"results/verification_image_{timestamp}.png"
                    img.save(temp_path)
                    
                    # Segment the image
                    try:
                        seg_results = system.segment_fruit_with_metrics(temp_path)
                        segmented_img = seg_results["segmented_image"]
                        
                        progress_bar.progress(20)
                        status_text.text("Verifying fruit type...")
                        
                        # Verify fruit type
                        selected_type = st.session_state.selected_fruit.lower()
                        verification_results = system.verify_fruit_type(segmented_img, selected_type)
                        
                        # If mismatch detected, show warning
                        if not verification_results["is_match"]:
                            progress_bar.progress(30)
                            status_text.text("Fruit type mismatch detected in front view. Waiting for user input...")
                            
                            # Prepare results for UI
                            verification_results.update({
                                "fruit_type_selected": selected_type,
                                "original_image": img,
                                "segmented_image": segmented_img
                            })
                            
                            # Store in session state
                            st.session_state.verification_results = verification_results
                            st.rerun()  # Show verification UI
                    except Exception as e:
                        # Log error but continue
                        print(f"Error during fruit verification: {str(e)}")
                
                # Continue with normal patch-based analysis
                status_text.text(f"Processing {st.session_state.selected_fruit} from multiple angles...")
                
                results_front = process_angle_image(
                    system, front_file, st.session_state.selected_fruit.lower(), "Front", 
                    use_segmentation, refine_segmentation, refinement_method, use_enhanced_analysis
                )
                progress_bar.progress(25)
                
                results_back = process_angle_image(
                    system, back_file, st.session_state.selected_fruit.lower(), "Back", 
                    use_segmentation, refine_segmentation, refinement_method, use_enhanced_analysis
                )
                progress_bar.progress(50)
                
                results_top = None
                results_bottom = None
                
                if st.session_state.show_top and top_file is not None:
                    results_top = process_angle_image(
                        system, top_file, st.session_state.selected_fruit.lower(), "Top", 
                        use_segmentation, refine_segmentation, refinement_method, use_enhanced_analysis
                    )
                    progress_bar.progress(75)
                
                if st.session_state.show_bottom and bottom_file is not None:
                    results_bottom = process_angle_image(
                        system, bottom_file, st.session_state.selected_fruit.lower(), "Bottom", 
                        use_segmentation, refine_segmentation, refinement_method, use_enhanced_analysis
                    )
                    progress_bar.progress(90)
                
                angle_results = [r for r in [results_front, results_back, results_top, results_bottom] if r is not None]
                combined_results = combine_multi_angle_results(angle_results)
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")

                if use_enhanced_analysis:
                    display_enhanced_patch_based_results(combined_results, system, username)
                else:
                    display_patch_based_results(combined_results, system, use_segmentation, username)
            else:
                st.error("No image data found. Please go back and upload images.")

if __name__ == "__main__":
    main()