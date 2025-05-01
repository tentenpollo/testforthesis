import cv2
import numpy as np

def refine_mask(mask, refinement_method="all", kernel_size=5, iterations=1):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    if mask.max() == 255:
        mask = mask // 255
    refined_mask = mask.copy()

    if refinement_method in ["morphological", "all"]:
        refined_mask = apply_morphological_operations(refined_mask, kernel_size, iterations)
    
    if refinement_method in ["contour", "all"]:
        refined_mask = refine_with_contours(refined_mask)
    
    if refinement_method in ["boundary", "all"]:
        refined_mask = smooth_boundaries(refined_mask, kernel_size)
    
    return refined_mask

def apply_morphological_operations(mask, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    temp_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    temp_mask = cv2.erode(temp_mask, kernel, iterations=1)
    temp_mask = cv2.dilate(temp_mask, kernel, iterations=1)
    
    return temp_mask

def refine_with_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    refined_mask = np.zeros_like(mask)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    
        min_area = cv2.contourArea(largest_contour) * 0.05  
        
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(refined_mask, [contour], 0, 1, -1)  
    else:
        
        refined_mask = mask
        
    return refined_mask

def smooth_boundaries(mask, kernel_size=5):
    """
    Smooth boundaries of the mask
    
    Args:
        mask: Binary mask as numpy array
        kernel_size: Size of smoothing kernel
        
    Returns:
        Mask with smoothed boundaries
    """
    
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    
    
    smoothed = np.where(blurred > 0.5, 1, 0).astype(np.uint8)
    
    return smoothed

def get_mask_quality_metrics(mask):
    """
    Calculate quality metrics for a segmentation mask
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Dictionary of quality metrics
    """
    
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    
    if mask.max() == 255:
        binary_mask = mask // 255
    else:
        binary_mask = mask
    
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    metrics = {}
    
    
    metrics["num_objects"] = len(contours)
    
    
    metrics["mask_area"] = np.sum(binary_mask)
    
    
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    metrics["coverage_ratio"] = metrics["mask_area"] / total_pixels
    
    
    if contours:
        total_perimeter = sum([cv2.arcLength(contour, True) for contour in contours])
        metrics["boundary_complexity"] = total_perimeter / metrics["mask_area"] if metrics["mask_area"] > 0 else float('inf')
    else:
        metrics["boundary_complexity"] = 0
    
    return metrics