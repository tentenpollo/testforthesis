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
    elif fruit_type_normalized in ["tomato", "tomatoes", "pineapple", "pineapples"]:
        return class_name.replace("_", " ").title()
    else:
        return class_name.capitalize()

def create_enhanced_visualization(results, original_image, segmented_image, save_path=None):
    original_cv = np.array(original_image)
    original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
    
    h, w = original_cv.shape[:2]
    display_img = cv2.copyMakeBorder(original_cv, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    fruit_type = results.get("fruit_type", "Unknown")
    raw_predictions = results.get("raw_results", {}).get("predictions", [])
    
    print(f"Raw predictions for visualization: {raw_predictions}")
    
    colors = [
        (0, 0, 255),      
        (255, 0, 0),      
        (0, 255, 0),      
        (255, 0, 255),    
        (0, 255, 255),    
    ]
    
    box_count = 0
    for i, pred in enumerate(raw_predictions):
        if "x" in pred and "y" in pred and "width" in pred and "height" in pred:
            
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            width_box = pred.get("width", 0)
            height_box = pred.get("height", 0)
            
            class_name = pred.get("class", "unknown")
            confidence_score = pred.get("confidence", 0)
            
            standardized_label = standardize_ripeness_label(fruit_type, class_name)
            
            if standardized_label is None:
                print(f"Skipping drawing bounding box for '{class_name}' class")
                continue
            
            x1 = int(x - width_box/2)
            y1 = int(y - height_box/2)
            x2 = int(x + width_box/2)
            y2 = int(y + height_box/2)
        
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(display_img.shape[1]-1, x2)
            y2 = min(display_img.shape[0]-1, y2)
            
            color = colors[box_count % len(colors)]
            box_count += 1
            
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 3)
            
            label = f"{standardized_label} {int(confidence_score * 100)}%"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            if y1 - text_size[1] - 10 >= 0:
                
                cv2.rectangle(display_img, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
                
                cv2.putText(display_img, label, (x1, y1-5), font, font_scale, (255, 255, 255), thickness)
            else:
                
                cv2.rectangle(display_img, (x1, y2), (x1+text_size[0], y2+text_size[1]+10), color, -1)
                cv2.putText(display_img, label, (x1, y2+text_size[1]+5), font, font_scale, (255, 255, 255), thickness)
    
    result_img = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_img.save(save_path)
        return save_path
    else:
        
        timestamp = int(time.time())
        default_save_path = f"results/boxes_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        result_img.save(default_save_path)
        return default_save_path

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