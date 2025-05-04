
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import torch
import io

from system import FruitRipenessSystem
from utils.helpers import get_image_download_link, seed_everything
from user_management import initialize_database, save_user_result
from authentication import show_login_page, get_current_user
from user_history import show_history_page, show_result_details
from huggingface_hub import hf_hub_download
from evaluation_page import add_evaluation_page_to_app

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
    seg_model_filename="best_model.pth",
    classifier_model_repo="TentenPolllo/fruitripeness",  # Use your actual HF repo name
    classifier_model_filename="fruit_classifier_full.pth"
):
    """Load both segmentation and classifier models from HF Hub"""
    seg_model_path = hf_hub_download(
        repo_id=seg_model_repo,
        filename=seg_model_filename,
    )
    
    classifier_model_path = hf_hub_download(
        repo_id=classifier_model_repo,
        filename=classifier_model_filename,
    )
    
    return FruitRipenessSystem(
        seg_model_path=seg_model_path,
        classifier_model_path=classifier_model_path
    )
    
def combine_multi_angle_results(results_list):
    """
    Combine results from multiple angles of the same fruit
    
    Args:
        results_list: List of results dictionaries from different angles
        
    Returns:
        Combined results dictionary
    """
    
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        return {"error": "No valid results to combine"}
    
    combined = valid_results[0].copy()

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
    
    
    combined["multi_angle"] = True
    combined["num_angles"] = len(valid_results)
    combined["angle_results"] = valid_results
    combined["angle_names"] = [r.get("angle_name", "Unknown") for r in valid_results]
    
    return combined

def process_angle_image(system, image_file, fruit_type, angle_name, use_segmentation, refine_segmentation, refinement_method):
    """Process an image for a specific angle in patch-based analysis"""
    try:
        if use_segmentation:
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
                        bbox_img = Image.open(bbox_path)
                        st.image(bbox_img, use_container_width=True)
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
                
                
                if "combined_visualization" in results["visualizations"]:
                    combined_path = results["visualizations"]["combined_visualization"]
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
                
                
                viz_tab1, viz_tab2 = st.tabs(["Bounding Box Detection", "Comparison View"])
                
                with viz_tab1:
                    bbox_path = results["visualizations"]["bounding_box_visualization"]
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
                
                with viz_tab2:
                    comparison_path = results["visualizations"]["comparison_visualization"]
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
                    st.subheader("🔍 Base U-Net vs Enhanced U-Net Comparison")
                    
                    comparison = results["comparison_metrics"]
                    
                    # Performance Metrics
                    st.write("**Performance Comparison:**")
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.metric("Base U-Net Time", f"{comparison['baseline_time']*1000:.1f} ms")
                    
                    with perf_col2:
                        st.metric("Enhanced U-Net Time", f"{comparison['enhanced_time']*1000:.1f} ms")
                    
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
                        st.write("Enhanced U-Net Result")
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
                                st.write(f"**{layer_name} Layer Comparison:**")
                                st.image(viz, use_container_width=True)
                                
                                # Add metrics table for this layer
                                if "layer_metrics" in comparison and layer_name in comparison["layer_metrics"]:
                                    layer_metric = comparison["layer_metrics"][layer_name]
                                    
                                    metric_data = {
                                        "Metric": ["Mean Activation", "Standard Deviation", "Feature Entropy"],
                                        "Base U-Net": [
                                            f"{layer_metric['baseline']['mean_activation']:.4f}",
                                            f"{layer_metric['baseline']['std_activation']:.4f}",
                                            f"{layer_metric['baseline']['entropy']:.4f}"
                                        ],
                                        "Enhanced U-Net": [
                                            f"{layer_metric['enhanced']['mean_activation']:.4f}",
                                            f"{layer_metric['enhanced']['std_activation']:.4f}",
                                            f"{layer_metric['enhanced']['entropy']:.4f}"
                                        ],
                                        "Improvement": [
                                            f"{(layer_metric['enhanced']['mean_activation'] - layer_metric['baseline']['mean_activation']) / abs(layer_metric['baseline']['mean_activation'] + 1e-8) * 100:.1f}%",
                                            f"{(layer_metric['enhanced']['std_activation'] - layer_metric['baseline']['std_activation']) / abs(layer_metric['baseline']['std_activation'] + 1e-8) * 100:.1f}%",
                                            f"{layer_metric['entropy_improvement']:.1f}%"
                                        ]
                                    }
                                    
                                    st.table(metric_data)
                    
                    with tab3:
                        if "regularization_viz" in results and results["regularization_viz"] is not None:
                            st.write("**Impact of Dynamic Regularization:**")
                            st.image(results["regularization_viz"], use_container_width=True)
                            
                            st.write("""
                            The chart above shows the L1 regularization values applied by the dynamic regularization modules.
                            Higher values indicate stronger regularization, which helps prevent overfitting.
                            """)
                    
                    # Explanation of architectural improvements
                    st.write("---")
                    st.subheader("📚 Enhanced U-Net Architecture Improvements")
                    
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
                        image_paths[viz_key] = viz_path
                
                # Save the results
                result_id = save_user_result(username, save_results, image_paths)
                
                st.success(f"✅ Analysis results saved successfully! (ID: {result_id})")
                
                # Add button to view history
                if st.button("View Saved Results"):
                    st.session_state.page = "history"
                    st.rerun()
    else:
        st.info("💡 Log in with a user account to save your analysis results for future reference.")
            

def display_patch_based_results(combined_results, system, use_segmentation, username):
    """Display results for patch-based analysis"""
    if "error" in combined_results and "angle_results" not in combined_results:
        st.error(f"Error: {combined_results['error']}")
        return
    
    
    st.subheader("📊 Patch-Based Analysis Results")
    st.info(f"Analyzed {combined_results['num_angles']} angles of the fruit")
    
    
    st.subheader("🔍 Fruit Analysis")
    st.write(f"**Selected Fruit:** {combined_results['fruit_type'].title()}")
    
    
    st.subheader("🍌 Combined Ripeness Detection")
    
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
    
    
    angle_results = combined_results["angle_results"]
    angle_names = combined_results["angle_names"]
    
    
    angle_tabs = st.tabs(angle_names)
    
    
    for i, (tab, angle_result) in enumerate(zip(angle_tabs, angle_results)):
        with tab:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Original {angle_names[i]} View**")
                st.image(angle_result["original_image"], use_container_width=True)
            
            with col2:
                if use_segmentation:
                    st.write(f"**Segmented {angle_names[i]} View**")
                    st.image(angle_result["segmented_image"], use_container_width=True)
                else:
                    if "visualizations" in angle_result and "bounding_box_visualization" in angle_result["visualizations"]:
                        st.write(f"**Detection Results for {angle_names[i]} View**")
                        bbox_path = angle_result["visualizations"]["bounding_box_visualization"]
                        bbox_img = Image.open(bbox_path)
                        st.image(bbox_img, use_container_width=True)
                    else:
                        st.write(f"**Processed {angle_names[i]} View**")
                        st.image(angle_result["original_image"], use_container_width=True)
            
            
            st.write(f"**Ripeness Predictions for {angle_names[i]} View:**")
            if "ripeness_predictions" in angle_result and angle_result["ripeness_predictions"]:
                
                angle_ripeness_data = {
                    "Ripeness Level": [],
                    "Confidence": []
                }
                
                for pred in angle_result["ripeness_predictions"]:
                    angle_ripeness_data["Ripeness Level"].append(pred["ripeness"])
                    angle_ripeness_data["Confidence"].append(f"{pred['confidence']:.2f}")
                
                
                st.table(angle_ripeness_data)
            else:
                st.write("No ripeness predictions for this angle")
            
            
            if "visualizations" in angle_result and "combined_visualization" in angle_result["visualizations"]:
                st.write(f"**Visualization for {angle_names[i]} View:**")
                combined_path = angle_result["visualizations"]["combined_visualization"]
                combined_img = Image.open(combined_path)
                st.image(combined_img, use_container_width=True)
    
    
    st.subheader("📊 Angle Comparison")
    
    
    all_ripeness_levels = set()
    for result in angle_results:
        if "ripeness_predictions" in result and result["ripeness_predictions"]:
            for pred in result["ripeness_predictions"]:
                all_ripeness_levels.add(pred["ripeness"])
    
    if all_ripeness_levels:
        ripeness_levels = list(all_ripeness_levels)
        ripeness_data = []
        
        for level in ripeness_levels:
            level_data = {"ripeness": level}
            for i, result in enumerate(angle_results):
                confidence = 0
                if "ripeness_predictions" in result and result["ripeness_predictions"]:
                    for pred in result["ripeness_predictions"]:
                        if pred["ripeness"] == level:
                            confidence = pred["confidence"]
                            break
                level_data[angle_names[i]] = confidence
            ripeness_data.append(level_data)
        
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.2
        index = np.arange(len(ripeness_levels))
        
        for i, angle in enumerate(angle_names):
            values = [data[angle] for data in ripeness_data]
            ax.bar(index + i*bar_width, values, bar_width, label=angle)
        
        ax.set_xlabel('Ripeness Level')
        ax.set_ylabel('Confidence')
        ax.set_title('Ripeness Predictions by Angle')
        ax.set_xticks(index + bar_width * (len(angle_names) - 1) / 2)
        ax.set_xticklabels(ripeness_levels)
        ax.legend()
        
        st.pyplot(fig)
    else:
        st.write("No ripeness predictions available for comparison")
    st.subheader("Save Results")

    if username and username != "guest":
        save_col1, save_col2 = st.columns([3, 1])
        
        with save_col1:
            save_note = st.text_input("Add a note about this multi-angle analysis (optional)")
        
        with save_col2:
            save_button = st.button("💾 Save Multi-Angle Results", type="primary")
            
            if save_button:
                # Prepare a copy of results for saving
                save_results = combined_results.copy()
                
                # Add user's note
                save_results["user_note"] = save_note
                
                # Add segmentation info
                save_results["segmentation"] = use_segmentation
                
                # Save image paths for storage
                image_paths = {}
                
                # Get paths from the first angle result (for overall representation)
                first_angle = combined_results["angle_results"][0]
                
                # Original image
                if "original_image_path" in first_angle:
                    image_paths["original"] = first_angle["original_image_path"]
                
                # Segmented image
                if "segmented_image_path" in first_angle:
                    image_paths["segmented"] = first_angle["segmented_image_path"]
                
                # Visualizations
                if "visualizations" in first_angle:
                    for viz_key, viz_path in first_angle["visualizations"].items():
                        image_paths[viz_key] = viz_path
                
                # Save the results
                result_id = save_user_result(username, save_results, image_paths)
                
                st.success(f"✅ Multi-angle analysis results saved successfully! (ID: {result_id})")
                
                # Add button to view history
                if st.button("View Saved Results"):
                    st.session_state.page = "history"
                    st.rerun()
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
            nav_options = ["Home", "Analysis History", "About", "Evaluation"]
            
            nav_map = {
                "Home": "home",
                "Analysis History": "history",
                "Evaluation": "evaluation",
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
                    
                    st.rerun()
            
            if st.button("Logout", use_container_width=True):
                for key in ["logged_in", "username", "page"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        st.divider()    
        st.caption("Fruit Ripeness Detection System v1.1")
        st.caption("© 2025 FruitThesis Industries")
    
    if not st.session_state.logged_in:
        if show_login_page():
            st.session_state.page = "home"
            st.rerun()
        return
    
    system = load_models()
    
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
                    st.markdown(f"### {fruit_icons['Tomato']} Tomato")
                    if st.button("Select Tomato", key="tomato_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Tomato"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
            
            with col2:
                pineapple_card = st.container()
                with pineapple_card:
                    st.markdown(f"### {fruit_icons['Pineapple']} Pineapple")
                    if st.button("Select Pineapple", key="pineapple_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Pineapple"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
            
            with col3:
                banana_card = st.container()
                with banana_card:
                    st.markdown(f"### {fruit_icons['Banana']} Banana")
                    if st.button("Select Banana", key="banana_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Banana"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
            
            with col4:
                strawberry_card = st.container()
                with strawberry_card:
                    st.markdown(f"### {fruit_icons['Strawberry']} Strawberry")
                    if st.button("Select Strawberry", key="strawberry_btn", use_container_width=True):
                        st.session_state.selected_fruit = "Strawberry"
                        st.session_state.analysis_step = "upload_image"
                        st.rerun()
            
            with col5:
                mango_card = st.container()
                with mango_card:
                    st.markdown(f"### {fruit_icons['Mango']} Mango")
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
                
                if (st.session_state.uploaded_file is not None or 
                    st.session_state.camera_image is not None):
                    
                    preview_container = st.container()
                    with preview_container:
                        st.subheader("Image Preview")
                        
                        prev_col1, prev_col2, prev_col3 = st.columns([1, 2, 1])
                        
                        with prev_col2:
                            image_input = st.session_state.camera_image if st.session_state.camera_image is not None else st.session_state.uploaded_file
                            st.image(image_input, use_container_width=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
                            st.session_state.analysis_step = "analyze"
                            st.rerun()
            
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
                    st.markdown("**🔄 Front View**")
                    front_file = st.file_uploader("Upload front view", type=["jpg", "jpeg", "png"], key="front")
                    if front_file:
                        # Create a smaller preview
                        st.image(front_file, width=250)
                    
                with back_col:
                    st.markdown("**🔄 Back View**")
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
                        st.markdown("**⬆️ Top View**")
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
                    st.markdown("**⬇️ Bottom View**")
                    bottom_file = st.file_uploader("Upload bottom view", type=["jpg", "jpeg", "png"], key="bottom")
                    if bottom_file:
                        # Create a smaller preview
                        st.image(bottom_file, width=250)
                
                # Show analysis button when required images are uploaded
                if front_file is not None and back_file is not None:
                    # Store files in session state
                    st.session_state.front_file = front_file
                    st.session_state.back_file = back_file
                    st.session_state.top_file = top_file
                    st.session_state.bottom_file = bottom_file
                    
                    # Add start analysis button
                    st.write("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(f"🔍 Start {st.session_state.selected_fruit} Analysis", type="primary", use_container_width=True):
                            st.session_state.analysis_step = "analyze"
                            st.session_state.start_analysis = True
                            st.rerun()
                else:
                    st.warning("Please upload both front and back images for patch-based analysis")
        
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
                st.rerun()
            
            username = get_current_user()
            
            if st.session_state.uploaded_file is not None or st.session_state.camera_image is not None:
                image_input = st.session_state.camera_image if st.session_state.camera_image is not None else st.session_state.uploaded_file
                
                if image_input is not None:
                    use_segmentation = st.session_state.use_segmentation
                    refine_segmentation = st.session_state.refine_segmentation
                    refinement_method = st.session_state.refinement_method
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    if use_segmentation:
                        status_text.text(f"Processing {st.session_state.selected_fruit} image with segmentation...")
                        progress_bar.progress(25)
                        
                        results = system.process_image_with_visualization(
                            image_input,
                            fruit_type=st.session_state.selected_fruit.lower(),
                            refine_segmentation=refine_segmentation,
                            refinement_method=refinement_method
                        )
                        
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
                front_file = st.session_state.front_file
                back_file = st.session_state.back_file
                top_file = st.session_state.top_file if "top_file" in st.session_state else None
                bottom_file = st.session_state.bottom_file if "bottom_file" in st.session_state else None
                
                use_segmentation = st.session_state.use_segmentation
                refine_segmentation = st.session_state.refine_segmentation
                refinement_method = st.session_state.refinement_method
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"Processing {st.session_state.selected_fruit} from multiple angles...")
                
                results_front = process_angle_image(
                    system, front_file, st.session_state.selected_fruit.lower(), "Front", 
                    use_segmentation, refine_segmentation, refinement_method
                )
                progress_bar.progress(25)
                
                results_back = process_angle_image(
                    system, back_file, st.session_state.selected_fruit.lower(), "Back", 
                    use_segmentation, refine_segmentation, refinement_method
                )
                progress_bar.progress(50)
                
                results_top = None
                results_bottom = None
                
                if st.session_state.show_top and top_file is not None:
                    results_top = process_angle_image(
                        system, top_file, st.session_state.selected_fruit.lower(), "Top", 
                        use_segmentation, refine_segmentation, refinement_method
                    )
                    progress_bar.progress(75)
                
                if st.session_state.show_bottom and bottom_file is not None:
                    results_bottom = process_angle_image(
                        system, bottom_file, st.session_state.selected_fruit.lower(), "Bottom", 
                        use_segmentation, refine_segmentation, refinement_method
                    )
                    progress_bar.progress(90)
                
                angle_results = [r for r in [results_front, results_back, results_top, results_bottom] if r is not None]
                combined_results = combine_multi_angle_results(angle_results)
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")

                display_patch_based_results(combined_results, system, use_segmentation, username)
            else:
                st.error("No image data found. Please go back and upload images.")

if __name__ == "__main__":
    main()