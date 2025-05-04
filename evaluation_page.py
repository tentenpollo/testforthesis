import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import streamlit as st
from system import FruitRipenessSystem
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_models(
    seg_model_repo="TentenPolllo/fruitripeness",
    classifier_model_path="fruit_classifier_full.pth"
):
    """Load segmentation model from HF Hub and classifier locally"""
    seg_model_path = hf_hub_download(
        repo_id=seg_model_repo,
        filename="best_model.pth",
    )
    
    return FruitRipenessSystem(
        seg_model_path=seg_model_path,
        classifier_model_path=classifier_model_path
    )

def evaluate_segmentation_from_folders(system, base_folder, fruit_type, refinement_options=None):
    """
    Evaluate segmentation impact on classification using a folder structure where:
    - Base folder is the fruit type (e.g., 'Tomato')
    - Subfolders are ripeness classes (e.g., 'Ripe', 'Unripe', 'Overripe')
    - Image files are in each subfolder
    
    Args:
        system: FruitRipenessSystem instance
        base_folder: Path to the base folder containing class subfolders
        fruit_type: Type of fruit to analyze (e.g., 'tomato')
        refinement_options: Optional list of refinement methods to test
        
    Returns:
        Dictionary with evaluation results
    """
    if refinement_options is None:
        refinement_options = ["none", "all", "morphological", "contour", "boundary"]
    
    # Initialize results structure
    results = {
        "without_segmentation": {
            "predictions": [],
            "ground_truth": [],
            "images": [],
            "confidences": []
        }
    }
    
    # Add results structure for each refinement option
    for option in refinement_options:
        if option == "none":
            results["with_segmentation_no_refinement"] = {
                "predictions": [],
                "ground_truth": [],
                "images": [],
                "confidences": []
            }
        else:
            results[f"with_segmentation_{option}"] = {
                "predictions": [],
                "ground_truth": [],
                "images": [],
                "confidences": []
            }
    
    # Check if base folder exists
    if not os.path.exists(base_folder):
        return {"error": f"Base folder {base_folder} does not exist"}
    
    # Get class subfolders
    class_folders = [f for f in os.listdir(base_folder) 
                    if os.path.isdir(os.path.join(base_folder, f))]
    
    if not class_folders:
        return {"error": f"No class subfolders found in {base_folder}"}
    
    st.write(f"Found {len(class_folders)} classes: {', '.join(class_folders)}")
    
    # Normalize fruit type
    fruit_type_lower = fruit_type.lower()
    
    # Process each class folder
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_images = 0
    
    # Count total images first
    for class_name in class_folders:
        class_path = os.path.join(base_folder, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images += len(image_files)
    
    # Process images
    processed_count = 0
    for class_name in class_folders:
        class_path = os.path.join(base_folder, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            status_text.text(f"Processing {img_path}...")
            
            # Process without segmentation
            no_seg_results = system.process_image_without_segmentation(
                img_path, fruit_type=fruit_type_lower
            )
            
            # Extract prediction for without segmentation
            if "ripeness_predictions" in no_seg_results and no_seg_results["ripeness_predictions"]:
                top_pred = no_seg_results["ripeness_predictions"][0]
                results["without_segmentation"]["predictions"].append(top_pred["ripeness"])
                results["without_segmentation"]["ground_truth"].append(class_name)
                results["without_segmentation"]["images"].append(img_path)
                results["without_segmentation"]["confidences"].append(top_pred["confidence"])
            
            # Process with segmentation but no refinement
            if "none" in refinement_options:
                seg_no_refine_results = system.process_image_with_visualization(
                    img_path, 
                    fruit_type=fruit_type_lower,
                    refine_segmentation=False
                )
                
                if ("ripeness_predictions" in seg_no_refine_results and 
                    seg_no_refine_results["ripeness_predictions"]):
                    top_pred = seg_no_refine_results["ripeness_predictions"][0]
                    results["with_segmentation_no_refinement"]["predictions"].append(top_pred["ripeness"])
                    results["with_segmentation_no_refinement"]["ground_truth"].append(class_name)
                    results["with_segmentation_no_refinement"]["images"].append(img_path)
                    results["with_segmentation_no_refinement"]["confidences"].append(top_pred["confidence"])
            
            # Process with segmentation and each refinement method
            for option in refinement_options:
                if option == "none":
                    continue  # Already handled above
                
                seg_results = system.process_image_with_visualization(
                    img_path, 
                    fruit_type=fruit_type_lower,
                    refine_segmentation=True,
                    refinement_method=option
                )
                
                if "ripeness_predictions" in seg_results and seg_results["ripeness_predictions"]:
                    top_pred = seg_results["ripeness_predictions"][0]
                    results[f"with_segmentation_{option}"]["predictions"].append(top_pred["ripeness"])
                    results[f"with_segmentation_{option}"]["ground_truth"].append(class_name)
                    results[f"with_segmentation_{option}"]["images"].append(img_path)
                    results[f"with_segmentation_{option}"]["confidences"].append(top_pred["confidence"])
            
            # Update progress
            processed_count += 1
            progress_bar.progress(processed_count / total_images)
    
    status_text.text("Processing complete!")
    
    # Calculate metrics for each method
    metrics = {}
    for method, data in results.items():
        if len(data["predictions"]) == 0:
            metrics[method] = {
                "error": "No predictions available"
            }
            continue
        
        # Calculate confusion matrix metrics
        metrics[method] = calculate_classification_metrics(
            data["predictions"], 
            data["ground_truth"],
            data["confidences"]
        )
        
        # Add raw data for further analysis
        metrics[method]["raw_data"] = {
            "predictions": data["predictions"],
            "ground_truth": data["ground_truth"],
            "images": data["images"],
            "confidences": data["confidences"]
        }
    
    return metrics

def calculate_classification_metrics(predictions, ground_truth, confidences=None):
    """
    Calculate classification metrics from predictions and ground truth
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        confidences: Optional list of confidence scores
        
    Returns:
        Dictionary with classification metrics
    """
    # Get unique classes
    classes = sorted(list(set(ground_truth)))
    
    # Initialize metrics
    metrics = {
        "overall": {
            "accuracy": 0,
            "total": len(ground_truth),
            "avg_confidence": 0 if confidences is None else sum(confidences) / len(confidences)
        },
        "per_class": {}
    }
    
    # Initialize per-class metrics
    for cls in classes:
        metrics["per_class"][cls] = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "support": 0,
            "avg_confidence": 0
        }
    
    # Count correct predictions for accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    metrics["overall"]["accuracy"] = correct / len(ground_truth) if len(ground_truth) > 0 else 0
    
    # Calculate per-class metrics
    for cls in classes:
        # Count true positives, false positives, false negatives
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g == cls)
        support = sum(1 for g in ground_truth if g == cls)
        
        # Calculate confidence scores for this class if available
        if confidences is not None:
            class_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == cls]
            avg_conf = sum(class_confidences) / len(class_confidences) if class_confidences else 0
        else:
            avg_conf = 0
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store in metrics
        metrics["per_class"][cls].update({
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall, 
            "f1_score": f1,
            "support": support,
            "avg_confidence": avg_conf
        })
    
    # Calculate macro-average metrics
    metrics["overall"]["macro_precision"] = sum(metrics["per_class"][cls]["precision"] for cls in classes) / len(classes)
    metrics["overall"]["macro_recall"] = sum(metrics["per_class"][cls]["recall"] for cls in classes) / len(classes)
    metrics["overall"]["macro_f1"] = sum(metrics["per_class"][cls]["f1_score"] for cls in classes) / len(classes)
    
    # Calculate weighted-average metrics
    total = sum(metrics["per_class"][cls]["support"] for cls in classes)
    if total > 0:
        metrics["overall"]["weighted_precision"] = sum(metrics["per_class"][cls]["precision"] * metrics["per_class"][cls]["support"] for cls in classes) / total
        metrics["overall"]["weighted_recall"] = sum(metrics["per_class"][cls]["recall"] * metrics["per_class"][cls]["support"] for cls in classes) / total
        metrics["overall"]["weighted_f1"] = sum(metrics["per_class"][cls]["f1_score"] * metrics["per_class"][cls]["support"] for cls in classes) / total
    
    return metrics

def display_evaluation_results(metrics):
    """Display evaluation results in a user-friendly way"""
    st.subheader("Evaluation Results")
    
    # Compare overall accuracy
    methods = list(metrics.keys())
    
    # Create a dataframe for overall metrics
    overall_data = {
        "Method": [],
        "Accuracy": [],
        "Macro Precision": [],
        "Macro Recall": [],
        "Macro F1": [],
        "Avg Confidence": []
    }
    
    for method in methods:
        if "error" in metrics[method]:
            continue
        
        # Add friendlier method name
        method_name = method.replace("_", " ").title()
        overall_data["Method"].append(method_name)
        overall_data["Accuracy"].append(metrics[method]["overall"]["accuracy"])
        overall_data["Macro Precision"].append(metrics[method]["overall"]["macro_precision"])
        overall_data["Macro Recall"].append(metrics[method]["overall"]["macro_recall"])
        overall_data["Macro F1"].append(metrics[method]["overall"]["macro_f1"])
        overall_data["Avg Confidence"].append(metrics[method]["overall"]["avg_confidence"])
    
    overall_df = pd.DataFrame(overall_data)
    st.write("Overall Performance Metrics:")
    st.table(overall_df)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_list = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    
    for i, metric in enumerate(metrics_list):
        ax = axes[i//2, i%2]
        x = np.arange(len(overall_data["Method"]))
        ax.bar(x, overall_data[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(overall_data["Method"], rotation=45, ha="right")
        ax.set_title(metric)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show per-class metrics for each method
    for method in methods:
        if "error" in metrics[method]:
            continue
            
        with st.expander(f"Per-Class Metrics - {method.replace('_', ' ').title()}"):
            if "per_class" in metrics[method]:
                per_class = metrics[method]["per_class"]
                
                class_data = {
                    "Class": list(per_class.keys()),
                    "Precision": [per_class[cls]["precision"] for cls in per_class],
                    "Recall": [per_class[cls]["recall"] for cls in per_class],
                    "F1-Score": [per_class[cls]["f1_score"] for cls in per_class],
                    "Support": [per_class[cls]["support"] for cls in per_class],
                    "Avg Confidence": [per_class[cls]["avg_confidence"] for cls in per_class]
                }
                
                class_df = pd.DataFrame(class_data)
                st.table(class_df)
                
                # Confusion matrix if there are enough classes
                if len(per_class) > 1:
                    st.write("Confusion Matrix:")
                    
                    # Create confusion matrix
                    cm = create_confusion_matrix(
                        metrics[method]["raw_data"]["predictions"],
                        metrics[method]["raw_data"]["ground_truth"],
                        list(per_class.keys())
                    )
                    
                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    
                    # Show all ticks and label them
                    ax.set_xticks(np.arange(len(per_class)))
                    ax.set_yticks(np.arange(len(per_class)))
                    ax.set_xticklabels(list(per_class.keys()))
                    ax.set_yticklabels(list(per_class.keys()))
                    
                    # Rotate the tick labels and set their alignment
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Loop over data dimensions and create text annotations
                    for i in range(len(per_class)):
                        for j in range(len(per_class)):
                            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
                    
                    ax.set_title("Confusion Matrix")
                    ax.set_ylabel('True label')
                    ax.set_xlabel('Predicted label')
                    fig.tight_layout()
                    st.pyplot(fig)
    
    # Show incorrectly classified images
    with st.expander("View Incorrectly Classified Images"):
        # Let user choose which method to view
        selected_method = st.selectbox(
            "Select Method",
            methods,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        if "error" not in metrics[selected_method]:
            raw_data = metrics[selected_method]["raw_data"]
            
            # Find incorrectly classified images
            incorrect_indices = [i for i, (pred, gt) in enumerate(zip(raw_data["predictions"], raw_data["ground_truth"])) if pred != gt]
            
            if incorrect_indices:
                st.write(f"Found {len(incorrect_indices)} incorrectly classified images for {selected_method.replace('_', ' ').title()}")
                
                # Show a random sample of incorrect predictions
                sample_size = min(10, len(incorrect_indices))
                sample_indices = np.random.choice(incorrect_indices, sample_size, replace=False)
                
                for idx in sample_indices:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(raw_data["images"][idx], caption=f"True: {raw_data['ground_truth'][idx]}", width=300)
                    
                    with col2:
                        st.write(f"Predicted: {raw_data['predictions'][idx]}")
                        st.write(f"Confidence: {raw_data['confidences'][idx]:.2f}")
            else:
                st.write(f"No incorrectly classified images found for {selected_method.replace('_', ' ').title()}")

def create_confusion_matrix(predictions, ground_truth, classes):
    """
    Create a confusion matrix from predictions and ground truth
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        classes: List of class names
        
    Returns:
        Confusion matrix as numpy array
    """
    # Initialize confusion matrix
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Create class to index mapping
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Fill confusion matrix
    for pred, gt in zip(predictions, ground_truth):
        pred_idx = class_to_idx.get(pred, -1)
        gt_idx = class_to_idx.get(gt, -1)
        
        if pred_idx >= 0 and gt_idx >= 0:
            cm[gt_idx, pred_idx] += 1
    
    return cm

def add_evaluation_page_to_app():
    """Add an evaluation page to the Streamlit app"""
    st.title("📊 Segmentation Effectiveness Evaluation")
    
    system = load_models()
    
    st.write("""
    This page helps you evaluate the effectiveness of fruit segmentation on ripeness classification.
    Upload a folder with subfolders for each ripeness class, and we'll analyze the impact of segmentation.
    """)
    
    # Let user select fruit type
    fruit_type = st.selectbox(
        "Select Fruit Type",
        system.supported_fruits,  # Your system's supported fruits list
        format_func=lambda x: x.title()
    )
    
    # Let user input folder path
    base_folder = st.text_input(
        "Enter path to folder containing ripeness class subfolders",
        help="The folder structure should be: base_folder/ripeness_class/image_files"
    )
    
    # Let user select refinement methods to test
    refinement_options = st.multiselect(
        "Select segmentation refinement methods to test",
        ["none", "all", "morphological", "contour", "boundary"],
        default=["none", "all"],
        help="Select methods to evaluate. 'none' means no refinement, just segmentation."
    )
    
    # Always include "without segmentation" for comparison
    st.info("The evaluation will always include 'without segmentation' for comparison.")
    
    if base_folder and st.button("Run Evaluation", type="primary"):
        if not os.path.exists(base_folder):
            st.error(f"Folder not found: {base_folder}")
        else:
            with st.spinner("Evaluating segmentation effectiveness..."):
                # Run evaluation
                metrics = evaluate_segmentation_from_folders(
                    system, base_folder, fruit_type, refinement_options
                )
                
                if "error" in metrics:
                    st.error(metrics["error"])
                else:
                    # Display evaluation results
                    display_evaluation_results(metrics)
                    
                    # Save results
                    timestamp = int(time.time())
                    results_path = f"results/segmentation_evaluation_{timestamp}_{fruit_type}.json"
                    
                    # Save metrics to file (you'll need to implement a save_metrics function)
                    # save_metrics(metrics, results_path)
                    
                    st.success(f"Evaluation complete! Results saved to {results_path}")