import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.font_manager as fm
import time

def standardize_ripeness_label(fruit_type, class_name):
    fruit_type_normalized = fruit_type.lower() if fruit_type else ""
    class_name_lower = class_name.lower() if class_name else ""
    
    if class_name_lower in ["flower", "null"]:
        return None
    
    if fruit_type_normalized in ["banana", "bananas"]:
        if class_name_lower in ["freshunripe"]:
            return "Unripe"
        elif class_name_lower in ["freshripe"]:
            return "Ripe"
        elif class_name_lower in ["rotten", "overripe"]:
            return "Overripe"
        else:
            return class_name.capitalize()
    elif fruit_type_normalized in ["strawberry", "strawberries"]:
        if class_name_lower in ["strawberryripe", "ripe"]:
            return "Ripe"
        elif class_name_lower in ["strawberryunripe", "unripe"]:    
            return "Unripe"
        elif class_name_lower in ["strawberryrotten", "rotten"]:
            return "Overripe"
        else:
            return class_name.capitalize()
    elif fruit_type_normalized in ["mango", "mangoes", "mangos"]:
        if class_name_lower in ["early_ripe-21-40-", "unripe-1-20-"]:
            return "Unripe"
        elif class_name_lower == "partially_ripe-41-60-":
            return "Underripe"
        elif class_name_lower == "ripe-61-80-":
            return "Ripe"
        elif class_name_lower == "over_ripe-81-100-":
            return "Overripe"
        else:
            return class_name.capitalize()
    elif fruit_type_normalized in ["tomato", "tomatoes", "pineapple", "pineapples"]:
        return class_name.replace("_", " ").title()
    else:
        return class_name.capitalize()
    
def create_enhanced_visualization(results, original_img, segmented_img, save_path):
    """
    Create an enhanced visualization with bounding boxes for the ripeness detection results
    Selectively displays confidence scores based on analysis mode
    
    Args:
        results: Dictionary containing the processing results
        original_img: Original PIL image
        segmented_img: Segmented PIL image
        save_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    # Convert PIL images to numpy arrays for OpenCV
    original_arr = np.array(original_img)
    
    # Make a copy for drawing on
    visualization = original_arr.copy()
    
    # Get predictions from raw results
    raw_results = results.get("raw_results", {})
    predictions = raw_results.get("predictions", [])
    
    # Determine if this is enhanced analysis mode
    is_enhanced = results.get("analysis_type") == "enhanced_two_stage" or "confidence_distributions" in results
    
    # Draw bounding boxes on the visualization
    has_predictions = False
    
    # Check if we have predictions in the raw_results
    if predictions:
        has_predictions = True
        # Draw bounding boxes from Roboflow predictions
        for pred in predictions:
            # Get bounding box coordinates (center format)
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            width = pred.get("width", 0)
            height = pred.get("height", 0)
            confidence = pred.get("confidence", 0)
            class_name = pred.get("class", "unknown")
            
            # Convert to top-left, bottom-right format
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(visualization.shape[1], x2)
            y2 = min(visualization.shape[0], y2)
            
            # Determine color based on class (e.g., different ripeness levels)
            color_map = {
                "tomato_green": (0, 255, 0),      # Green
                "tomato_turning": (0, 255, 255),  # Yellow
                "tomato_pink": (0, 165, 255),     # Orange
                "tomato_light_red": (0, 0, 255),  # Red
                "tomato_red": (0, 0, 255),        # Red
                "tomato_ripe": (0, 0, 255),       # Red
                "tomato_breaker": (0, 255, 255),  # Yellow
                "tomato_half_ripe": (0, 165, 255), # Orange
                "tomato_rotten": (128, 0, 128),   # Purple
                "tomato_damaged": (128, 0, 128),  # Purple
                "default": (255, 255, 255)        # White
            }
            
            # Get color based on class
            color = color_map.get(class_name.lower(), color_map["default"])
            
            # Draw bounding box
            cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
            
            # Create label - different for enhanced vs standard mode
            if is_enhanced:
                # Enhanced mode: Class name only, without confidence
                display_name = class_name.replace("_", " ").title()
                label = display_name
            else:
                # Standard mode: Include confidence score
                display_name = class_name.replace("_", " ").title()
                label = f"{display_name}: {confidence:.2f}"
            
            # Get text size
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Ensure text background doesn't go outside image
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1]
            
            # Draw text background
            cv2.rectangle(visualization, (text_x, text_y - text_size[1]), 
                        (text_x + text_size[0], text_y), color, -1)
            
            # Draw text
            cv2.putText(visualization, label, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Check if we need to look in other places for bounding boxes (fallback approach)
    if not has_predictions:
        # Try getting prediction information from fruits_data if available
        if "fruits_data" in results:
            has_predictions = True
            for fruit_data in results["fruits_data"]:
                if "bbox" in fruit_data:
                    bbox = fruit_data["bbox"]
                    
                    # Get bounding box coordinates
                    x = bbox.get("x", 0)
                    y = bbox.get("y", 0)
                    width = bbox.get("width", 0)
                    height = bbox.get("height", 0)
                    confidence = bbox.get("confidence", 0)
                    class_name = bbox.get("class", "unknown")
                    
                    # Convert to top-left, bottom-right format
                    x1 = int(x - width / 2)
                    y1 = int(y - height / 2)
                    x2 = int(x + width / 2)
                    y2 = int(y + height / 2)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(visualization.shape[1], x2)
                    y2 = min(visualization.shape[0], y2)
                    
                    # Use a default color (blue) for fruits_data bounding boxes
                    color = (255, 0, 0)  # Blue for fruits_data
                    
                    # Draw bounding box
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label - different for enhanced vs standard mode
                    if is_enhanced:
                        # Enhanced mode: Class name only, without confidence
                        display_name = class_name.replace("_", " ").title()
                        label = display_name
                    else:
                        # Standard mode: Include confidence score
                        display_name = class_name.replace("_", " ").title()
                        label = f"{display_name}: {confidence:.2f}"
                    
                    # Get text size
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Ensure text background doesn't go outside image
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1]
                    
                    # Draw text background
                    cv2.rectangle(visualization, (text_x, text_y - text_size[1]), 
                                (text_x + text_size[0], text_y), color, -1)
                    
                    # Draw text
                    cv2.putText(visualization, label, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add a title to the visualization
    fruit_type = results.get("fruit_type", "Unknown").title()
    title = f"{fruit_type} Ripeness Detection"
    
    # Draw title at the top of the image
    cv2.putText(visualization, title, (10, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Save the visualization
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    # Return the path to the saved visualization
    return save_path

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
    try:
        bbox_visualization = create_enhanced_visualization(
            results, original_img, segmented_img, box_vis_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error creating bounding box visualization: {str(e)}")
        bbox_visualization = None
    
    # Create side-by-side comparison
    comparison_path = f"{save_dir}/comparison_{timestamp}.png"
    try:
        comparison_visualization = create_side_by_side_comparison(
            results, original_img, segmented_img, comparison_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error creating comparison visualization: {str(e)}")
        comparison_visualization = None
    
    # Create combined visualization
    combined_path = f"{save_dir}/combined_{timestamp}.png"
    try:
        combined_visualization = create_combined_visualization(
            results, original_img, segmented_img, combined_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error creating combined visualization: {str(e)}")
        combined_visualization = None
    
    # Collect all successful visualizations
    result_paths = {}
    
    if bbox_visualization:
        result_paths["bounding_box_visualization"] = bbox_visualization
    
    if comparison_visualization:
        result_paths["comparison_visualization"] = comparison_visualization
    
    if combined_visualization:
        result_paths["combined_visualization"] = combined_visualization
    
    return result_paths

def create_side_by_side_comparison(results, original_image, segmented_image, save_path=None):
    segmentation_disabled = original_image is segmented_image
    
    width, height = original_image.size
    
    padding = 30
    if segmentation_disabled:
        canvas_width = width + (padding * 2)  
    else:
        canvas_width = (width * 2) + (padding * 3)  
    
    text_height = 150  
    canvas_height = height + padding * 2 + text_height

    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    
    try:
        
        fonts = [f for f in fm.findSystemFonts() if 'arial' in f.lower() or 'helvetica' in f.lower()]
        if fonts:
            font_path = fonts[0]
        else:
            
            font_path = fm.findfont(fm.FontProperties())
            
        title_font = ImageFont.truetype(font_path, 28)
        header_font = ImageFont.truetype(font_path, 22)
        info_font = ImageFont.truetype(font_path, 18)
    except:
        
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        info_font = ImageFont.load_default()
    
    
    fruit_type = results.get("fruit_type", "Unknown")
    confidence = results.get("classification_confidence", 0)
    ripeness_predictions = results.get("ripeness_predictions", [])
    
    
    if segmentation_disabled:
        title = f"Fruit Ripeness Detection: {fruit_type} (No Segmentation)"
    else:
        title = f"Fruit Ripeness Detection: {fruit_type}"
    
    title_width = draw.textlength(title, font=title_font)
    draw.text(((canvas_width - title_width) // 2, padding // 2), title, fill=(0, 0, 0), font=title_font)
    
    if segmentation_disabled:
        
        header_text = "Original Image"
        header_width = draw.textlength(header_text, font=header_font)
        draw.text((padding + width//2 - header_width//2, padding * 2), header_text, fill=(0, 0, 0), font=header_font)
        
        
        canvas.paste(original_image, (padding, padding * 3))
    else:
        
        header_text = "Original Image"
        header_width = draw.textlength(header_text, font=header_font)
        draw.text((padding + width//2 - header_width//2, padding * 2), header_text, fill=(0, 0, 0), font=header_font)
        
        
        header_text = "Segmented Image"
        header_width = draw.textlength(header_text, font=header_font) 
        draw.text((padding * 2 + width + width//2 - header_width//2, padding * 2), header_text, fill=(0, 0, 0), font=header_font)
        
        
        canvas.paste(original_image, (padding, padding * 3))
        
        
        canvas.paste(segmented_image, (padding * 2 + width, padding * 3))
    
    
    line_y = padding * 3 + height + padding // 2
    draw.line([(padding, line_y), (canvas_width - padding, line_y)], fill=(200, 200, 200), width=2)
    
    
    text_y = line_y + padding
    draw.text((padding, text_y), f"Fruit Type: {fruit_type} (Confidence: {confidence:.2f})", fill=(0, 0, 0), font=info_font)
    
    
    text_y += 30
    draw.text((padding, text_y), "Ripeness Predictions:", fill=(0, 0, 0), font=info_font)
    
    for i, pred in enumerate(ripeness_predictions):
        ripeness = pred.get("ripeness", "Unknown")
        pred_confidence = pred.get("confidence", 0)
        text_y += 25
        draw.text((padding * 2, text_y), f"{ripeness}: {pred_confidence:.2f}", fill=(0, 0, 0), font=info_font)
    
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
        return save_path
    else:
        
        timestamp = int(time.time())
        default_save_path = f"results/comparison_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        canvas.save(default_save_path)
        return default_save_path

def create_combined_visualization(results, original_image, segmented_image, save_path=None):
    """
    Create a complete visualization with both bounding boxes and side-by-side comparison
    
    Args:
        results: Dictionary containing ripeness prediction results
        original_image: PIL Image of the original image
        segmented_image: PIL Image of the segmented image
        save_path: Optional path to save the visualization
        
    Returns:
        Path to the saved visualization image
    """
    
    segmentation_disabled = original_image is segmented_image
    
    
    
    original_cv = np.array(original_image)
    original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
    
    
    height, width = original_cv.shape[:2]
    
    
    raw_predictions = results.get("raw_results", {}).get("predictions", [])
    fruit_type = results.get("fruit_type", "Unknown")
    
    
    colors = [
        (0, 0, 255),      
        (255, 0, 0),      
        (0, 255, 0),      
        (255, 0, 255),    
        (0, 255, 255),    
    ]
    
    
    bbox_img = original_cv.copy()
    
    
    box_count = 0
    for i, pred in enumerate(raw_predictions):
        if "x" in pred and "y" in pred and "width" in pred and "height" in pred:
            
            class_name = pred.get("class", "unknown")
            confidence_score = pred.get("confidence", 0)
            
            
            standardized_label = standardize_ripeness_label(fruit_type, class_name)
            
            
            if standardized_label is None:
                print(f"Skipping drawing bounding box for '{class_name}' class in combined visualization")
                continue
                
            
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            width_box = pred.get("width", 0)
            height_box = pred.get("height", 0)
            
            
            x1 = int(x - width_box/2)
            y1 = int(y - height_box/2)
            x2 = int(x + width_box/2)
            y2 = int(y + height_box/2)
            
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width-1, x2)
            y2 = min(height-1, y2)
            
            
            color = colors[box_count % len(colors)]
            box_count += 1
            
            
            cv2.rectangle(bbox_img, (x1, y1), (x2, y2), color, 3)
            
            
            label = f"{standardized_label} {int(confidence_score * 100)}%"
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            
            if y1 - text_size[1] - 10 >= 0:
                
                cv2.rectangle(bbox_img, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
                
                cv2.putText(bbox_img, label, (x1, y1-5), font, font_scale, (255, 255, 255), thickness)
            else:
                
                cv2.rectangle(bbox_img, (x1, y2), (x1+text_size[0], y2+text_size[1]+10), color, -1)
                cv2.putText(bbox_img, label, (x1, y2+text_size[1]+5), font, font_scale, (255, 255, 255), thickness)
    
    
    bbox_img_pil = Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
    
    
    fruit_type = results.get("fruit_type", "Unknown")
    confidence = results.get("classification_confidence", 0)
    ripeness_predictions = results.get("ripeness_predictions", [])
    
    
    padding = 30
    
    if segmentation_disabled:
        
        canvas_width = width * 2 + padding * 3  
    else:
        canvas_width = width * 3 + padding * 4  
    
    
    text_height = 150
    canvas_height = height + text_height + padding * 3
    
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
    
    draw = ImageDraw.Draw(canvas)
    
    
    try:
        fonts = [f for f in fm.findSystemFonts() if 'arial' in f.lower() or 'helvetica' in f.lower()]
        if fonts:
            font_path = fonts[0]
        else:
            font_path = fm.findfont(fm.FontProperties())
            
        title_font = ImageFont.truetype(font_path, 28)
        header_font = ImageFont.truetype(font_path, 22)
        info_font = ImageFont.truetype(font_path, 18)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        info_font = ImageFont.load_default()
    
    
    if segmentation_disabled:
        title = f"Fruit Ripeness Detection: {fruit_type} (No Segmentation)"
    else:
        title = f"Fruit Ripeness Detection: {fruit_type}"
        
    title_width = draw.textlength(title, font=title_font)
    draw.text(((canvas_width - title_width) // 2, padding // 2), title, fill=(0, 0, 0), font=title_font)
    
    
    if segmentation_disabled:
        headers = ["Original Image", "Bounding Box Detection"]
    else:
        headers = ["Original Image", "Bounding Box Detection", "Segmented Image"]
    
    for i, header in enumerate(headers):
        header_width = draw.textlength(header, font=header_font)
        x_pos = padding + (width + padding) * i + width // 2 - header_width // 2
        draw.text((x_pos, padding * 2), header, fill=(0, 0, 0), font=header_font)
    
    
    canvas.paste(original_image, (padding, padding * 3))
    canvas.paste(bbox_img_pil, (padding * 2 + width, padding * 3))
    if not segmentation_disabled:
        canvas.paste(segmented_image, (padding * 3 + width * 2, padding * 3))
    
    
    line_y = padding * 3 + height + padding // 2
    draw.line([(padding, line_y), (canvas_width - padding, line_y)], fill=(200, 200, 200), width=2)
    
    
    text_y = line_y + padding
    draw.text((padding, text_y), f"Fruit Type: {fruit_type} (Confidence: {confidence:.2f})", fill=(0, 0, 0), font=info_font)
    
    
    text_y += 30
    draw.text((padding, text_y), "Ripeness Predictions:", fill=(0, 0, 0), font=info_font)
    
    for i, pred in enumerate(ripeness_predictions):
        ripeness = pred.get("ripeness", "Unknown")
        pred_confidence = pred.get("confidence", 0)
        text_y += 25
        draw.text((padding * 2, text_y), f"{ripeness}: {pred_confidence:.2f}", fill=(0, 0, 0), font=info_font)
    
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
        return save_path
    else:
        
        timestamp = int(time.time())
        default_save_path = f"results/complete_vis_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        canvas.save(default_save_path)
        return default_save_path