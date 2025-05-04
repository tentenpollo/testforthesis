import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import os
from PIL import ImageDraw

def feature_map_visualization(activation, title, max_images=8):
    """Generate a visualization of feature maps"""
    if activation is None or not isinstance(activation, torch.Tensor):
        # Create a placeholder image with error message
        placeholder = Image.new('RGB', (400, 100), color=(240, 240, 240))
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 40), f"No activation data for {title}", fill=(0, 0, 0))
        return placeholder
        
    features = activation.cpu().numpy()
    
    # Get the first n channels for visualization
    n_features = min(features.shape[1], max_images)
    
    # Create a grid
    rows = int(np.ceil(n_features / 4))
    cols = min(n_features, 4)
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n_features:
                feature = features[0, idx]
                
                # Normalize for visualization
                feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
                
                axes[i, j].imshow(feature, cmap='viridis')
                axes[i, j].set_title(f'F{idx}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    
    # Convert to PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    
    # Close figure to avoid memory leak
    plt.close(fig)
    
    return image

def calculate_feature_stats(activation):
    """Calculate statistics about feature maps"""
    if activation is None:
        return {
            "mean_activation": 0.0,
            "std_activation": 0.0,
            "entropy": 0.0
        }
        
    features = activation.cpu().numpy()
    
    # Calculate metrics
    mean_activation = np.mean(features)
    std_activation = np.std(features)
    channel_entropy = []
    
    for c in range(features.shape[1]):
        feature = features[0, c]
        # Normalize to 0-255 for entropy calculation
        feature_norm = ((feature - feature.min()) / (feature.max() - feature.min() + 1e-8) * 255).astype(np.uint8)
        hist = cv2.calcHist([feature_norm], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        channel_entropy.append(entropy)
    
    # Average entropy across channels
    avg_entropy = np.mean(channel_entropy)
    
    return {
        "mean_activation": float(mean_activation),
        "std_activation": float(std_activation),
        "entropy": float(avg_entropy)
    }

def compare_model_metrics(baseline_model, enhanced_model, key_layers):
    """Compare metrics between baseline and enhanced models"""
    metrics = {}
    
    for layer_name in key_layers:
        baseline_key, enhanced_key = key_layers[layer_name]
        baseline_activation = baseline_model.activations.get(baseline_key) if hasattr(baseline_model, 'activations') else None
        enhanced_activation = enhanced_model.activations.get(enhanced_key) if hasattr(enhanced_model, 'activations') else None
        
        if baseline_activation is not None or enhanced_activation is not None:
            baseline_stats = calculate_feature_stats(baseline_activation)
            enhanced_stats = calculate_feature_stats(enhanced_activation)
            
            # Calculate improvement percentages (handle division by zero)
            if baseline_stats["entropy"] != 0:
                entropy_improvement = ((enhanced_stats["entropy"] - baseline_stats["entropy"]) / 
                                    baseline_stats["entropy"] * 100)
            else:
                entropy_improvement = 0
            
            metrics[layer_name] = {
                "baseline": baseline_stats,
                "enhanced": enhanced_stats,
                "entropy_improvement": entropy_improvement
            }
    
    return metrics

# In visualization_metrics.py, update generate_comparison_visualization:

def generate_comparison_visualization(baseline_model, enhanced_model, key_layers):
    """Generate visualizations comparing baseline and enhanced models"""
    visualizations = {}
    
    for layer_name, (baseline_key, enhanced_key) in key_layers.items():
        # Safely get activations
        baseline_activation = None
        enhanced_activation = None
        
        if hasattr(baseline_model, 'activations'):
            baseline_activation = baseline_model.activations.get(baseline_key)
        
        if hasattr(enhanced_model, 'activations'):
            enhanced_activation = enhanced_model.activations.get(enhanced_key)
        
        # Only create visualization if we have at least one of the activations
        if baseline_activation is not None or enhanced_activation is not None:
            # Create side-by-side visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Handle case where one activation might be None
            if baseline_activation is not None:
                feature_idx = min(4, baseline_activation.shape[1]-1) if baseline_activation.shape[1] > 4 else 0
                baseline_feature = baseline_activation[0, feature_idx].cpu().numpy()
                baseline_feature = (baseline_feature - baseline_feature.min()) / (baseline_feature.max() - baseline_feature.min() + 1e-8)
                im1 = ax1.imshow(baseline_feature, cmap='viridis')
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            else:
                ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax1.set_facecolor('#f0f0f0')
            
            ax1.set_title(f'Baseline U-Net\n{layer_name}')
            ax1.axis('off')
            
            if enhanced_activation is not None:
                feature_idx = min(4, enhanced_activation.shape[1]-1) if enhanced_activation.shape[1] > 4 else 0
                enhanced_feature = enhanced_activation[0, feature_idx].cpu().numpy()
                enhanced_feature = (enhanced_feature - enhanced_feature.min()) / (enhanced_feature.max() - enhanced_feature.min() + 1e-8)
                im2 = ax2.imshow(enhanced_feature, cmap='viridis')
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            else:
                ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax2.set_facecolor('#f0f0f0')
            
            ax2.set_title(f'Enhanced U-Net\n{layer_name}')
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Convert to PIL image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            visualizations[layer_name] = Image.open(buf)
            
            # Close figure to avoid memory leak
            plt.close(fig)
    
    return visualizations   

def visualize_regularization_impact(enhanced_model):
    """Visualize the impact of dynamic regularization"""
    try:
        # Check if the model has the method
        if not hasattr(enhanced_model, 'get_regularization_metrics'):
            print("Model does not have get_regularization_metrics method")
            return None
        
        try:
            reg_metrics = enhanced_model.get_regularization_metrics()
        except Exception as e:
            print(f"Error getting regularization metrics: {e}")
            return None
        
        if not reg_metrics:
            print("No regularization metrics found")
            return None
        
        # Create horizontal bar chart instead of vertical for better label clarity
        fig, ax = plt.subplots(figsize=(8, 5))
        
        modules = list(reg_metrics.keys())
        values = list(reg_metrics.values())
        
        # Use shorter module names for better display
        short_names = [name.replace('reg', 'R') for name in modules]
        
        # Create horizontal bar chart 
        bars = ax.barh(short_names, values, color='skyblue')
        
        # Add value labels inside the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width * 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{width:.6f}',
                   ha='center', va='center', 
                   color='black', fontweight='bold')
        
        ax.set_xlabel('L1 Regularization Value')
        ax.set_title('Dynamic Regularization Impact')
        plt.tight_layout()
        
        # Convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        reg_viz = Image.open(buf)
        
        # Close figure to avoid memory leak
        plt.close(fig)
        
        return reg_viz
    except Exception as e:
        print(f"Error in visualization: {e}")
        return None