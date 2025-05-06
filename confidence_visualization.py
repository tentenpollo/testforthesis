import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import os
import time

def visualize_confidence_distribution(fruit_data, confidence_distribution, fruit_type, save_path=None):
    """
    Create visualization of ripeness confidence distribution for a single fruit
    Handles different class structures for various fruit classification models
    
    Args:
        fruit_data: Dictionary with fruit data
        confidence_distribution: Dictionary with confidence values by ripeness level
        fruit_type: Type of fruit
        save_path: Optional path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    try:
        # Check if we have valid data
        if not confidence_distribution or "error" in confidence_distribution:
            fig, ax = plt.subplots(figsize=(8, 4))
            error_msg = confidence_distribution.get("error", "No confidence distribution available")
            ax.text(0.5, 0.5, f"Error: {error_msg}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            
            # Convert to PIL image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            error_img = Image.open(buf)
            plt.close(fig)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                error_img.save(save_path)
                return save_path
            else:
                timestamp = int(time.time())
                default_save_path = f"results/confidence_error_{timestamp}.png"
                os.makedirs("results", exist_ok=True)
                error_img.save(default_save_path)
                return default_save_path
        
        # Load the fruit image
        fruit_img = None
        if "masked_crop_path" in fruit_data and os.path.exists(fruit_data["masked_crop_path"]):
            fruit_img = Image.open(fruit_data["masked_crop_path"])
        elif "original_crop_path" in fruit_data and os.path.exists(fruit_data["original_crop_path"]):
            fruit_img = Image.open(fruit_data["original_crop_path"])
        
        # Create a figure with two parts: image and confidence chart
        fig = plt.figure(figsize=(12, 5))
        
        # Define generic color maps for ripeness levels
        generic_colors = {
            "Unripe": '#A5D6A7',       # Green
            "Partially Ripe": '#FFF59D', # Yellow
            "Ripe": '#FFCC80',          # Orange
            "Overripe": '#EF9A9A',      # Red
            "Green": '#A5D6A7',         # Green (alias)
            "Yellow": '#FFF59D',        # Yellow (alias)
            "Red": '#FFCC80',           # Orange/Red (alias)
            "Rotten": '#EF9A9A'         # Red (alias)
        }
        
        # Color palette for specific classes based on fruit type
        fruit_specific_colors = {
            "banana": {
                "FreshUnripe": '#A5D6A7',   # Green
                "FreshRipe": '#FFCC80',     # Orange
                "Overripe": '#EF9A9A'       # Red
            },
            "mango": {
                "Unripe-1-20-": '#A5D6A7',             # Green
                "Early_Ripe-21-40-": '#C5E1A5',       # Light green
                "Partially_Ripe-41-60-": '#FFF59D',   # Yellow
                "Ripe-61-80-": '#FFCC80',             # Orange
                "Over_Ripe-81-100-": '#EF9A9A'        # Red
            },
            "tomato": {
                "Green": '#A5D6A7',         # Green
                "Breaker": '#C5E1A5',       # Light green
                "Turning": '#FFF59D',       # Yellow
                "Pink": '#FFAB91',          # Light orange
                "Light_Red": '#FF8A65',     # Orange
                "Red": '#EF5350'            # Red
            },
            "strawberry": {
                "Unripe": '#A5D6A7',     # Green
                "Ripe": '#EF5350',      # Red
                "Overripe": '#C62828'   # Dark red/brown
            },
            "pineapple": {
                "Unripe": '#A5D6A7',        # Green
                "Ripe": '#FFCC80',          # Orange/yellow
                "Overripe": '#EF9A9A'       # Red
            }
        }
        
        # If we have an image, display it on the left
        if fruit_img:
            ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
            ax1.imshow(fruit_img)
            ax1.set_title("Fruit Image")
            ax1.axis('off')
            
            # Confidence chart on the right
            ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=3)
        else:
            # Just show the confidence chart
            ax2 = plt.subplot(1, 1, 1)
        
        # Clean the confidence distribution
        is_estimated = confidence_distribution.pop("estimated", False) if "estimated" in confidence_distribution else False
        
        # Get the categories and values
        categories = list(confidence_distribution.keys())
        values = list(confidence_distribution.values())
        
        # Define colors based on fruit type and categories
        bar_colors = []
        fruit_type_normalized = fruit_type.lower()
        
        # First try fruit-specific color mapping
        if fruit_type_normalized in fruit_specific_colors:
            specific_colors = fruit_specific_colors[fruit_type_normalized]
            for cat in categories:
                if cat in specific_colors:
                    bar_colors.append(specific_colors[cat])
                elif cat.replace(" ", "_") in specific_colors:  # Try with underscores
                    bar_colors.append(specific_colors[cat.replace(" ", "_")])
                elif cat.lower() in generic_colors:  # Fall back to generic colors
                    bar_colors.append(generic_colors[cat.lower()])
                else:
                    bar_colors.append('#BDBDBD')  # Default gray
        else:
            # Use generic color mapping
            for cat in categories:
                if cat in generic_colors:
                    bar_colors.append(generic_colors[cat])
                elif cat.lower() in generic_colors:
                    bar_colors.append(generic_colors[cat.lower()])
                else:
                    bar_colors.append('#BDBDBD')  # Default gray
        
        # Try to sort categories in a logical ripeness progression if possible
        try:
            # Define standard ripeness progression
            standard_progression = ["Unripe", "Green", "Breaker", "Turning", 
                                   "Partially Ripe", "Pink", "Light Red", "Ripe", 
                                   "Red", "Overripe", "Rotten"]
            
            # Create a mapping for sorting
            category_order = {}
            for i, cat in enumerate(standard_progression):
                category_order[cat] = i
                
            # Default order for unknown categories
            default_order = len(standard_progression)
            
            # Sort categories based on standard progression
            sorted_indices = sorted(range(len(categories)), 
                                   key=lambda i: category_order.get(categories[i], default_order))
            
            # Reorder data based on sorted indices
            categories = [categories[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            bar_colors = [bar_colors[i] for i in sorted_indices]
        except Exception as e:
            print(f"Error sorting categories: {e}")
            # Continue with unsorted data
        
        # Convert values to percentages
        values_percent = [v * 100 for v in values]
        
        # Create horizontal bar chart
        bars = ax2.barh(categories, values_percent, color=bar_colors, edgecolor='black', alpha=0.8)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 5:  # Only add text if bar is wide enough
                ax2.text(width - 5, bar.get_y() + bar.get_height()/2,
                       f'{width:.1f}%',
                       ha='right', va='center', fontsize=10, color='black', fontweight='bold')
            else:
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                       f'{width:.1f}%',
                       ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Set title and labels
        fruit_index = fruit_data.get("index", 0)
        bbox = fruit_data.get("bbox", {})
        confidence = bbox.get("confidence", 0)
        
        if is_estimated:
            title = f"{fruit_type.title()} #{fruit_index+1} Ripeness Distribution (Estimated)"
            title_color = 'darkred'
        else:
            title = f"{fruit_type.title()} #{fruit_index+1} Ripeness Distribution"
            title_color = 'black'
            
        ax2.set_title(title, fontsize=14, color=title_color)
        ax2.set_xlabel('Confidence (%)')
        ax2.set_xlim(0, 101)  # Set x-axis to go from 0-100%
        
        # Add a grid for readability
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add note about estimation if applicable
        if is_estimated:
            plt.figtext(0.5, 0.01, 
                      "Note: This distribution is estimated based on the top detection class confidence.",
                      ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        distribution_img = Image.open(buf)
        plt.close(fig)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            distribution_img.save(save_path)
            return save_path
        
        # Create default path
        timestamp = int(time.time())
        default_save_path = f"results/confidence_dist_{fruit_index}_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        distribution_img.save(default_save_path)
        return default_save_path
        
    except Exception as e:
        print(f"Error creating confidence distribution visualization: {e}")
        
        # Create error image
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Error creating visualization: {e}", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_facecolor('#fff0f0')  # Light red background to indicate error
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        error_img = Image.open(buf)
        plt.close(fig)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            error_img.save(save_path)
            return save_path
        else:
            timestamp = int(time.time())
            default_save_path = f"results/confidence_error_{timestamp}.png"
            os.makedirs("results", exist_ok=True)
            error_img.save(default_save_path)
            return default_save_path

def visualize_all_fruits_confidence(results, save_path=None):
    """
    Create visualization showing confidence distributions for all fruits in the image
    
    Args:
        results: Results dictionary from analyze_ripeness_enhanced
        save_path: Optional path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    try:
        fruit_type = results.get("fruit_type", "Unknown")
        fruits_data = results.get("fruits_data", [])
        confidence_distributions = results.get("confidence_distributions", [])
        
        # If no fruits, return error image
        if not fruits_data or not confidence_distributions:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No fruit data available", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            error_img = Image.open(buf)
            plt.close(fig)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                error_img.save(save_path)
                return save_path
            else:
                timestamp = int(time.time())
                default_save_path = f"results/all_fruits_error_{timestamp}.png"
                os.makedirs("results", exist_ok=True)
                error_img.save(default_save_path)
                return default_save_path
        
        # Calculate figure size based on number of fruits
        num_fruits = len(fruits_data)
        fig_height = 3 + (num_fruits * 2)  # 3 inch base + 2 inch per fruit
        
        # Create figure
        fig, axes = plt.subplots(num_fruits, 1, figsize=(10, fig_height))
        
        # If only one fruit, wrap axes in a list
        if num_fruits == 1:
            axes = [axes]
            
        # Define generic color maps for ripeness levels (same as in single fruit visualization)
        generic_colors = {
            "Unripe": '#A5D6A7',       # Green
            "Partially Ripe": '#FFF59D', # Yellow
            "Ripe": '#FFCC80',          # Orange
            "Overripe": '#EF9A9A',      # Red
            "Green": '#A5D6A7',         # Green (alias)
            "Yellow": '#FFF59D',        # Yellow (alias)
            "Red": '#FFCC80',           # Orange/Red (alias)
            "Rotten": '#EF9A9A'         # Red (alias)
        }
        
        # Create fruit-specific color mappings (same as in single fruit visualization)
        fruit_specific_colors = {
            "banana": {
                "FreshUnripe": '#A5D6A7',   # Green
                "FreshRipe": '#FFCC80',     # Orange
                "Overripe": '#EF9A9A'       # Red
            },
            "mango": {
                "Unripe-1-20-": '#A5D6A7',             # Green
                "Early_Ripe-21-40-": '#C5E1A5',       # Light green
                "Partially_Ripe-41-60-": '#FFF59D',   # Yellow
                "Ripe-61-80-": '#FFCC80',             # Orange
                "Over_Ripe-81-100-": '#EF9A9A'        # Red
            },
            "tomato": {
                "Green": '#A5D6A7',         # Green
                "Breaker": '#C5E1A5',       # Light green
                "Turning": '#FFF59D',       # Yellow
                "Pink": '#FFAB91',          # Light orange
                "Light_Red": '#FF8A65',     # Orange
                "Red": '#EF5350'            # Red
            },
            "strawberry": {
                "Unripe": '#A5D6A7',     # Green
                "Ripe": '#EF5350',      # Red
                "Overripe": '#C62828'   # Dark red/brown
            },
            "pineapple": {
                "Unripe": '#A5D6A7',        # Green
                "Ripe": '#FFCC80',          # Orange/yellow
                "Overripe": '#EF9A9A'       # Red
            }
        }
        
        # Define standard ripeness progression for sorting
        standard_progression = ["Unripe", "Green", "Breaker", "Turning", 
                               "Partially Ripe", "Pink", "Light Red", "Ripe", 
                               "Red", "Overripe", "Rotten"]
        
        # Create a mapping for sorting
        category_order = {cat: i for i, cat in enumerate(standard_progression)}
        
        # Process each fruit
        for i, (fruit_data, distribution) in enumerate(zip(fruits_data, confidence_distributions)):
            ax = axes[i]
            
            # Clean the distribution
            is_estimated = distribution.pop("estimated", False) if "estimated" in distribution else False
            
            if "error" in distribution:
                ax.text(0.5, 0.5, f"Error for fruit #{i+1}: {distribution['error']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Get the categories and values
            categories = list(distribution.keys())
            values = [distribution[cat] * 100 for cat in categories]  # Convert to percentages
            
            # Define colors based on fruit type and categories
            bar_colors = []
            fruit_type_normalized = fruit_type.lower()
            
            # First try fruit-specific color mapping
            if fruit_type_normalized in fruit_specific_colors:
                specific_colors = fruit_specific_colors[fruit_type_normalized]
                for cat in categories:
                    if cat in specific_colors:
                        bar_colors.append(specific_colors[cat])
                    elif cat.replace(" ", "_") in specific_colors:  # Try with underscores
                        bar_colors.append(specific_colors[cat.replace(" ", "_")])
                    elif cat.lower() in generic_colors:  # Fall back to generic colors
                        bar_colors.append(generic_colors[cat.lower()])
                    else:
                        bar_colors.append('#BDBDBD')  # Default gray
            else:
                # Use generic color mapping
                for cat in categories:
                    if cat in generic_colors:
                        bar_colors.append(generic_colors[cat])
                    elif cat.lower() in generic_colors:
                        bar_colors.append(generic_colors[cat.lower()])
                    else:
                        bar_colors.append('#BDBDBD')  # Default gray
            
            # Try to sort categories in a logical ripeness progression
            try:
                # Default order for unknown categories
                default_order = len(standard_progression)
                
                # Sort categories based on standard progression
                sorted_indices = sorted(range(len(categories)), 
                                      key=lambda j: category_order.get(categories[j], default_order))
                
                # Reorder data based on sorted indices
                categories = [categories[j] for j in sorted_indices]
                values = [values[j] for j in sorted_indices]
                bar_colors = [bar_colors[j] for j in sorted_indices]
            except Exception as e:
                print(f"Error sorting categories: {e}")
                # Continue with unsorted data
            
            # Create horizontal bar chart
            bars = ax.barh(categories, values, color=bar_colors, edgecolor='black', alpha=0.8)
            
            # Add percentage labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                if width > 5:  # Only add text if bar is wide enough
                    ax.text(width - 5, bar.get_y() + bar.get_height()/2,
                          f'{width:.1f}%',
                          ha='right', va='center', fontsize=10, color='black', fontweight='bold')
                else:
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                          f'{width:.1f}%',
                          ha='left', va='center', fontsize=10, fontweight='bold')
            
            # Set title and labels
            if is_estimated:
                title = f"Fruit #{i+1}: {fruit_type.title()} Ripeness Distribution (Estimated)"
                title_color = 'darkred'
            else:
                title = f"Fruit #{i+1}: {fruit_type.title()} Ripeness Distribution"
                title_color = 'black'
                
            ax.set_title(title, fontsize=12, color=title_color)
            ax.set_xlabel('Confidence (%)')
            ax.set_xlim(0, 101)  # Set x-axis to go from 0-100%
            
            # Add a grid for readability
            ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add overall title
        plt.suptitle(f"Ripeness Analysis: {num_fruits} {fruit_type.title()}(s) Detected", 
                    fontsize=16, y=0.98)
        
        # Add note about estimation if any distributions are estimated
        any_estimated = any(d.pop("estimated", False) if isinstance(d, dict) and "estimated" in d else False 
                           for d in confidence_distributions)
        if any_estimated:
            plt.figtext(0.5, 0.01, 
                      "Note: Distributions marked as 'Estimated' are approximated based on detection results.",
                      ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        all_fruits_img = Image.open(buf)
        plt.close(fig)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            all_fruits_img.save(save_path)
            return save_path
        
        # Create default path
        timestamp = int(time.time())
        default_save_path = f"results/all_fruits_distribution_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        all_fruits_img.save(default_save_path)
        return default_save_path
        
    except Exception as e:
        print(f"Error creating all fruits visualization: {e}")
        
        # Create error image
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Error creating visualization: {e}", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_facecolor('#fff0f0')  # Light red background to indicate error
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        error_img = Image.open(buf)
        plt.close(fig)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            error_img.save(save_path)
            return save_path
        else:
            timestamp = int(time.time())
            default_save_path = f"results/all_fruits_error_{timestamp}.png"
            os.makedirs("results", exist_ok=True)
            error_img.save(default_save_path)
            return default_save_path