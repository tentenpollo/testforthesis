"""
Helper functions for Fruit Ripeness Detection System
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import base64
from io import BytesIO
import json
def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_mask_to_image(image, mask):
    """
    Apply binary mask to an image
    
    Args:
        image: PIL Image or numpy array
        mask: Binary mask as numpy array (single channel)
    
    Returns:
        PIL Image with mask applied
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    # Ensure mask is right size
    if img_array.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Expand mask to 3 channels if needed
    if len(mask.shape) == 2:
        mask_array = np.expand_dims(mask, axis=2) * 255
        mask_array = np.repeat(mask_array, 3, axis=2) // 255
    else:
        mask_array = mask
    
    # Apply mask
    segmented_array = img_array * mask_array
    return Image.fromarray(segmented_array.astype(np.uint8))

def get_image_download_link(img, filename="segmented_image.png", text="Download Segmented Image"):
    """
    Generate a link to download a PIL image
    
    Args:
        img: PIL Image
        filename: Name of the file to download
        text: Link text
    
    Returns:
        HTML hyperlink for downloading the image
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def visualize_results(original_image, segmented_image, fruit_type, classification_confidence, ripeness_predictions):
    """
    Create a visualization of the processing results
    
    Args:
        original_image: PIL Image of original input
        segmented_image: PIL Image of segmented fruit
        fruit_type: Detected fruit type (string)
        classification_confidence: Confidence score for fruit type
        ripeness_predictions: List of ripeness predictions
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show segmented image
    axes[1].imshow(segmented_image)
    
    # Construct title
    title = f"Segmented Image: {fruit_type.title()}\n"
    title += f"Confidence: {classification_confidence:.2f}\n"
    
    if ripeness_predictions:
        for i, pred in enumerate(ripeness_predictions[:2]):  # Show top 2 predictions
            title += f"{pred['ripeness']}: {pred['confidence']:.2f}\n"
            
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def convert_image_for_model(img, transform):
    """
    Convert PIL image to tensor for model input
    
    Args:
        img: PIL Image
        transform: torchvision transforms to apply
        
    Returns:
        Tensor ready for model input (batch dimension added)
    """
    return transform(img).unsqueeze(0)

def load_mask_from_output(output, threshold=0.5):
    """
    Convert model output to binary mask
    
    Args:
        output: Model output tensor
        threshold: Threshold for binary segmentation
        
    Returns:
        Binary mask as numpy array
    """
    with torch.no_grad():
        mask = torch.sigmoid(output) > threshold
        mask = mask.squeeze().cpu().numpy().astype(np.uint8)
    return mask

def make_serializable(obj):
    """
    Improved function to convert complex objects to JSON-serializable types
    
    Args:
        obj: The object to convert
        
    Returns:
        A JSON-serializable version of the object
    """
    import numpy as np
    from PIL import Image
    
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items() if k not in ["original_image", "segmented_image"]}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        if obj.size > 1000000:  # Large arrays (like masks)
            # For large arrays, just return shape and data type info
            return {"__numpy_array__": True, "shape": obj.shape, "dtype": str(obj.dtype)}
        else:
            return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, Image.Image):
        # Return dictionary with image info instead of the image itself
        return {"__pil_image__": True, "size": obj.size, "mode": obj.mode}
    else:
        # For other objects, convert to string representation
        try:
            return str(obj)
        except:
            return "Unserializable object"