import streamlit as st
import datetime
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from user_management import get_user_results, get_result_details, delete_user_result
import time

def show_history_page(username):
    """Display the user's saved analysis history with support for enhanced analysis"""
    st.title("📊 Analysis History")
    st.subheader(f"History for {username}")
    
    # Get all results for the user
    results = get_user_results(username)
    
    if not results:
        st.info("You don't have any saved analysis results yet.")
        st.write("Complete a fruit analysis to see it here!")
        return
    
    # Add session state variable for deletion confirmation
    if "delete_confirmation" not in st.session_state:
        st.session_state.delete_confirmation = None
    
    # Create tabs for different views
    list_tab, stats_tab = st.tabs(["Analysis List", "Statistics"])
    
    with list_tab:
        # Display results as a list
        for index, result in enumerate(results):
            # Handle missing or invalid timestamp
            try:
                timestamp = result.get('timestamp', '')
                formatted_date = datetime.datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp else 'Unknown Date'
            except ValueError:
                formatted_date = 'Unknown Date'
            
            # Get result type
            analysis_type = result.get('analysis_type', 'Single Image')
            if analysis_type == 'enhanced_two_stage':
                analysis_type = 'Enhanced Analysis'
            
            # Determine ripeness info based on result type
            ripeness_info = []
            if result.get('primary_ripeness') and result.get('primary_confidence'):
                # For enhanced results with primary_ripeness field
                ripeness_info.append({
                    'ripeness': result.get('primary_ripeness'),
                    'confidence': result.get('primary_confidence')
                })
            elif result.get('ripeness_predictions'):
                # For standard results
                ripeness_info = result.get('ripeness_predictions')
                
            with st.expander(
                f"#{index + 1}: {result.get('fruit_type', 'Unknown')} - {formatted_date}"
            ):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**Fruit Type:** {result.get('fruit_type', 'Unknown')}")
                    st.write(f"**Analysis Type:** {analysis_type}")
                    
                    try:
                        if timestamp:
                            formatted_full_date = datetime.datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            st.write(f"**Date:** {formatted_full_date}")
                        else:
                            st.write("**Date:** Unknown")
                    except ValueError:
                        st.write("**Date:** Invalid format")
                    
                    # Show ripeness information
                    st.write("**Ripeness Predictions:**")
                    if ripeness_info:
                        for pred in ripeness_info:
                            conf = pred.get('confidence', 0)
                            # Format confidence as percentage or decimal based on value
                            if conf > 0 and conf < 1:  # Decimal format
                                st.write(f"- {pred.get('ripeness')}: {conf:.2f}")
                            else:  # Percentage or other format
                                st.write(f"- {pred.get('ripeness')}: {conf:.2f}")
                    else:
                        st.write("- No ripeness predictions available")
                    
                    # Show user note if available
                    if result.get('user_note'):
                        st.write(f"**Note:** {result.get('user_note')}")
                
                with col2:
                    # Look for appropriate image to display
                    image_displayed = False
                    
                    # Try multiple possible image locations
                    if "image_paths" in result:
                        # Priority order for images to display
                        image_priority = [
                            "combined_visualization", 
                            "fruit_0_distribution",    # Enhanced distribution visualization
                            "all_fruits_distribution", # Multi-fruit visualization
                            "bounding_box_visualization",
                            "segmented",
                            "original"
                        ]
                        
                        for img_key in image_priority:
                            if img_key in result["image_paths"] and os.path.exists(result["image_paths"][img_key]):
                                try:
                                    img = Image.open(result["image_paths"][img_key])
                                    st.image(img, width=300, caption=img_key.replace("_", " ").title())
                                    image_displayed = True
                                    break
                                except Exception as e:
                                    continue
                    
                    if not image_displayed:
                        st.info("No image available")
                
                button_col1, button_col2 = st.columns(2)
                
                with button_col1:
                    if st.button(f"View Full Details", key=f"view_{result.get('id')}", use_container_width=True):
                        st.session_state.selected_result_id = result.get('id')
                        st.session_state.view_details = True
                        st.rerun()
                
                with button_col2:
                    if st.button(f"🗑️ Delete", key=f"delete_{result.get('id')}", 
                              type="secondary", use_container_width=True):
                        st.session_state.delete_confirmation = result.get('id')
                        st.rerun()
            
            # Handle delete confirmation
            if st.session_state.delete_confirmation == result.get('id'):
                with st.container():
                    st.warning(f"⚠️ Are you sure you want to delete this {result.get('fruit_type', 'Unknown')} analysis from {formatted_date}?")
                    
                    confirm_col1, confirm_col2 = st.columns(2)
                    
                    with confirm_col1:
                        if st.button("✅ Yes, Delete", key=f"confirm_{result.get('id')}", type="primary"):
                            # Call the delete function
                            if delete_user_result(username, result.get('id')):
                                st.success("Analysis result deleted successfully!")
                                # Reset confirmation state
                                st.session_state.delete_confirmation = None
                                # Refresh to update the list
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete the analysis result.")
                    
                    with confirm_col2:
                        if st.button("❌ Cancel", key=f"cancel_{result.get('id')}", type="secondary"):
                            # Reset confirmation state
                            st.session_state.delete_confirmation = None
                            st.rerun()
    
    with stats_tab:
        if len(results) > 0:
            # Prepare data for statistics
            fruit_types = {}
            ripeness_data = {}
            timestamps = []
            
            for result in results:
                # Count fruit types
                fruit_type = result.get('fruit_type', 'Unknown')
                fruit_types[fruit_type] = fruit_types.get(fruit_type, 0) + 1
                
                # Track top ripeness prediction for each result, handling both formats
                if result.get('ripeness_predictions'):
                    top_ripeness = result['ripeness_predictions'][0]['ripeness']
                    if fruit_type not in ripeness_data:
                        ripeness_data[fruit_type] = {}
                    ripeness_data[fruit_type][top_ripeness] = ripeness_data[fruit_type].get(top_ripeness, 0) + 1
                elif result.get('primary_ripeness'):
                    # Handle enhanced results format
                    top_ripeness = result.get('primary_ripeness')
                    if fruit_type not in ripeness_data:
                        ripeness_data[fruit_type] = {}
                    ripeness_data[fruit_type][top_ripeness] = ripeness_data[fruit_type].get(top_ripeness, 0) + 1
                
                # Track timestamps for activity chart - handle errors safely
                try:
                    timestamp = result.get('timestamp', '')
                    if timestamp:
                        timestamps.append(datetime.datetime.fromisoformat(timestamp))
                except ValueError:
                    pass
            
            # Create stats charts
            st.subheader("Analysis Statistics")
            
            # Fruit type distribution
            st.write("### Fruit Type Distribution")
            fig = px.pie(
                values=list(fruit_types.values()),
                names=list(fruit_types.keys()),
                title="Fruit Types Analyzed"
            )
            st.plotly_chart(fig)
            
            # Ripeness distribution by fruit type
            st.write("### Ripeness Analysis by Fruit Type")
            for fruit, ripeness_counts in ripeness_data.items():
                if ripeness_counts:
                    fig = px.bar(
                        x=list(ripeness_counts.keys()),
                        y=list(ripeness_counts.values()),
                        title=f"Ripeness Distribution for {fruit}"
                    )
                    st.plotly_chart(fig)
            
            # Activity over time
            if timestamps:
                st.write("### Analysis Activity")
                # Group by day
                timestamps.sort()
                dates = [ts.date() for ts in timestamps]
                date_counts = {}
                for date in dates:
                    date_counts[date] = date_counts.get(date, 0) + 1
                
                fig = px.line(
                    x=list(date_counts.keys()),
                    y=list(date_counts.values()),
                    title="Analyses Over Time",
                    labels={"x": "Date", "y": "Number of Analyses"}
                )
                st.plotly_chart(fig)
            else:
                st.info("No valid timestamps for activity chart.")
        else:
            st.info("Not enough data for statistics.")


def show_result_details(username, result_id):
    """Show detailed view of a single result with support for enhanced analysis"""
    result_data = get_result_details(username, result_id)
    
    if not result_data:
        st.error("Result not found")
        if st.button("Back to History"):
            if "selected_result_id" in st.session_state:
                del st.session_state.selected_result_id
            if "view_details" in st.session_state:
                del st.session_state.view_details
            st.rerun()
        return
    
    if st.button("← Back to History"):
        if "selected_result_id" in st.session_state:
            del st.session_state.selected_result_id
        if "view_details" in st.session_state:
            del st.session_state.view_details
        st.rerun()
    
    # Determine if this is an enhanced analysis
    is_enhanced = result_data.get('analysis_type') == 'enhanced_two_stage'
    
    # Get fruit type
    fruit_type = result_data.get('fruit_type', 'Unknown')
    
    st.subheader(f"{fruit_type.title()} {result_data.get('analysis_type', '').title()} Analysis")
    
    # Extract and display date information
    try:
        if "timestamp" in result_data and result_data["timestamp"]:
            timestamp = result_data["timestamp"]
            try:
                formatted_date = datetime.datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                st.write(f"**Date:** {formatted_date}")
            except ValueError:
                st.write(f"**Date:** {timestamp}")
        elif result_id and '_' in result_id:
            # Try to parse date from ID
            parts = result_id.split('_')
            if len(parts) >= 3:
                date_part = parts[-2]
                time_part = parts[-1]
                
                if len(date_part) == 8 and len(time_part) == 6:
                    year = date_part[0:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    
                    hour = time_part[0:2]
                    minute = time_part[2:4]
                    second = time_part[4:6]
                    
                    formatted_date = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                    st.write(f"**Date:** {formatted_date}")
                else:
                    st.write("**Date:** Unknown")
            else:
                st.write("**Date:** Unknown")
        else:
            st.write("**Date:** Unknown")
    except Exception as e:
        st.write(f"**Date:** Unable to format date")
    
    # Display user note if available
    if "user_note" in result_data and result_data["user_note"]:
        st.write(f"**Note:** {result_data['user_note']}")
    
    # Display images
    image_paths = result_data.get("image_paths", {})
    
    # For enhanced two-stage analysis, handle confidence distribution visualizations
    if is_enhanced:
        # First look for combined visualization
        if "combined_visualization" in image_paths:
            st.subheader("Analysis Visualization")
            img_path = image_paths["combined_visualization"]
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading visualization: {e}")
        
        # Look for fruit distribution visualizations - used in enhanced analysis
        dist_keys = [k for k in image_paths.keys() if k.endswith("_distribution")]
        if dist_keys:
            st.subheader("Ripeness Confidence Distributions")
            for dist_key in dist_keys:
                if os.path.exists(image_paths[dist_key]):
                    try:
                        img = Image.open(image_paths[dist_key])
                        st.image(img, use_container_width=True, 
                               caption=dist_key.replace("_", " ").title())
                    except Exception as e:
                        st.error(f"Error loading {dist_key}: {e}")
    else:
        # Standard visualization for regular analysis
        if "combined_visualization" in image_paths:
            st.subheader("Analysis Visualization")
            img_path = image_paths["combined_visualization"]
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading visualization: {e}")
    
    # Create tabs for different sections
    images_tab, ripeness_tab, technical_tab = st.tabs(["Images", "Ripeness Analysis", "Technical Details"])
    
    with images_tab:
        # Display original and segmented images
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Image**")
            original_found = False
            
            # Try different possible keys for original image
            original_keys = ["original", "original_image_path"]
            
            # Look in image_paths
            if "image_paths" in result_data:
                for key in original_keys:
                    if key in result_data["image_paths"]:
                        img_path = result_data["image_paths"][key]
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                                original_found = True
                                break
                            except Exception as e:
                                continue
            
            # If we still don't have an image, try looking directly in result_data
            if not original_found:
                if "original_image_path" in result_data:
                    img_path = result_data["original_image_path"]
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                            original_found = True
                        except Exception as e:
                            pass
            
            # If still no image found, display a message
            if not original_found:
                st.info("Original image not available")
        
        with col2:
            st.write("**Processed Image**")
            processed_found = False
            
            # Try different keys
            processed_keys = ["segmented", "bounding_box_visualization"]
            
            if "image_paths" in result_data:
                for key in processed_keys:
                    if key in result_data["image_paths"]:
                        img_path = result_data["image_paths"][key]
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                                processed_found = True
                                break
                            except Exception as e:
                                continue
            
            # Check segmentation_results if available
            if not processed_found and "segmentation_results" in result_data:
                if "segmented_image_path" in result_data["segmentation_results"]:
                    img_path = result_data["segmentation_results"]["segmented_image_path"]
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                            processed_found = True
                        except Exception as e:
                            pass
            
            # If still no image found, display a message
            if not processed_found:
                st.info("Processed image not available")
    
    with ripeness_tab:
        st.subheader("Ripeness Analysis")
        
        # Handle different types of ripeness data
        if is_enhanced:
            # For enhanced analysis, look for confidence_distributions
            if "confidence_distributions" in result_data and result_data["confidence_distributions"]:
                distributions = result_data["confidence_distributions"]
                
                # Display each distribution
                for i, distribution in enumerate(distributions):
                    if isinstance(distribution, dict):
                        # Filter out non-confidence keys
                        filtered_dist = {k: v for k, v in distribution.items() 
                                      if k not in ["error", "estimated"] 
                                      and isinstance(v, (int, float))}
                        
                        if filtered_dist:
                            st.write(f"**Fruit #{i+1} Ripeness Confidence:**")
                            
                            # Create table data
                            confidence_data = {
                                "Ripeness Level": list(filtered_dist.keys()),
                                "Confidence": [f"{v:.4f}" for v in filtered_dist.values()],
                                "Percentage": [f"{v*100:.1f}%" for v in filtered_dist.values()]
                            }
                            
                            st.table(confidence_data)
                            
                            # Create a chart
                            fig = px.bar(
                                x=list(filtered_dist.keys()),
                                y=[v*100 for v in filtered_dist.values()],
                                title=f"Ripeness Confidence for Fruit #{i+1}",
                                labels={"x": "Ripeness Level", "y": "Confidence (%)"}
                            )
                            st.plotly_chart(fig)
                        else:
                            st.info(f"No valid confidence distribution for Fruit #{i+1}")
            else:
                st.info("No ripeness confidence distributions available")
        else:
            # Standard ripeness predictions
            if "ripeness_predictions" in result_data and result_data["ripeness_predictions"]:
                # Create a table
                ripeness_data = {
                    "Ripeness Level": [],
                    "Confidence": []
                }
                
                for pred in result_data["ripeness_predictions"]:
                    ripeness_data["Ripeness Level"].append(pred.get("ripeness", "Unknown"))
                    ripeness_data["Confidence"].append(f"{pred.get('confidence', 0):.2f}")
                
                st.table(ripeness_data)
                
                # Create a visualization
                fig = px.bar(
                    x=[pred.get("ripeness", "Unknown") for pred in result_data["ripeness_predictions"]],
                    y=[pred.get("confidence", 0) for pred in result_data["ripeness_predictions"]],
                    title="Ripeness Confidence Scores",
                    labels={"x": "Ripeness Level", "y": "Confidence Score"}
                )
                st.plotly_chart(fig)
            else:
                st.info("No ripeness predictions available for this analysis.")
        
        # Check if this is a multi-angle analysis
        if "multi_angle" in result_data and result_data["multi_angle"] and "angle_results" in result_data:
            st.subheader("Multi-Angle Analysis")
            
            angle_results = result_data.get("angle_results", [])
            angle_names = result_data.get("angle_names", [])
            
            if angle_results and angle_names:
                # Create angle tabs
                angle_tabs = st.tabs(angle_names)
                
                for i, (tab, angle_result) in enumerate(zip(angle_tabs, angle_results)):
                    with tab:
                        st.write(f"**{angle_names[i]} View Ripeness Predictions:**")
                        
                        if "ripeness_predictions" in angle_result and angle_result["ripeness_predictions"]:
                            # Create a table
                            angle_ripeness_data = {
                                "Ripeness Level": [],
                                "Confidence": []
                            }
                            
                            for pred in angle_result["ripeness_predictions"]:
                                angle_ripeness_data["Ripeness Level"].append(pred.get("ripeness", "Unknown"))
                                angle_ripeness_data["Confidence"].append(f"{pred.get('confidence', 0):.2f}")
                            
                            st.table(angle_ripeness_data)
                            
                            # Add visualization if available in this angle
                            if "visualizations" in angle_result:
                                for viz_name, viz_path in angle_result["visualizations"].items():
                                    if os.path.exists(viz_path):
                                        try:
                                            img = Image.open(viz_path)
                                            st.image(img, use_container_width=True, 
                                                   caption=f"{angle_names[i]} {viz_name.replace('_', ' ').title()}")
                                        except Exception as e:
                                            continue
                        else:
                            st.info("No ripeness predictions for this angle")
    
    with technical_tab:
        # Display technical details about the analysis
        st.subheader("Technical Details")
        
        # Check if segmentation was used
        if "segmentation" in result_data:
            st.write(f"**Segmentation:** {'Enabled' if result_data.get('segmentation', False) else 'Disabled'}")
            
            if result_data.get("segmentation", False):
                st.write(f"**Segmentation Model:** SPEAR-UNet")
                
                if "refinement_method" in result_data:
                    st.write(f"**Refinement Method:** {result_data.get('refinement_method', 'all')}")
                
                # Look for mask metrics in different possible locations
                mask_metrics = None
                if "mask_metrics" in result_data:
                    mask_metrics = result_data["mask_metrics"]
                elif "segmentation_results" in result_data and "mask_metrics" in result_data["segmentation_results"]:
                    mask_metrics = result_data["segmentation_results"]["mask_metrics"]
                
                if mask_metrics:
                    st.write("**Mask Quality Metrics:**")
                    
                    # Handle both formats (direct value or numpy array)
                    coverage = mask_metrics.get('coverage_ratio', 0)
                    if hasattr(coverage, 'item'):  # Handle numpy scalars
                        coverage = coverage.item()
                        
                    complexity = mask_metrics.get('boundary_complexity', 0)
                    if hasattr(complexity, 'item'):
                        complexity = complexity.item()
                        
                    st.write(f"- Mask coverage: {coverage:.2%} of image")
                    st.write(f"- Boundary complexity: {complexity:.2f}")
        
        # Show processing info for enhanced analysis
        if is_enhanced:
            st.write("**Enhanced Two-Stage Processing:**")
            st.write("- Stage 1: Segmentation to isolate the fruit")
            st.write("- Stage 2: Multi-scale detection and classification")
            
            # Show neural network metrics if available
            if "segmentation_results" in result_data and "comparison_metrics" in result_data["segmentation_results"]:
                metrics = result_data["segmentation_results"]["comparison_metrics"]
                
                st.write("**Neural Network Performance:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    baseline_time = metrics.get("baseline_time", 0)
                    if hasattr(baseline_time, 'item'):
                        baseline_time = baseline_time.item()
                    st.metric("Base U-Net Time", f"{baseline_time*1000:.1f} ms")
                
                with col2:
                    enhanced_time = metrics.get("enhanced_time", 0)
                    if hasattr(enhanced_time, 'item'):
                        enhanced_time = enhanced_time.item()
                    st.metric("SPEAR-UNet Time", f"{enhanced_time*1000:.1f} ms")
                
                with col3:
                    if baseline_time > 0 and enhanced_time > 0:
                        speedup = baseline_time / enhanced_time
                        st.metric("Performance", f"{speedup:.2f}x", f"{(speedup-1)*100:.1f}%")
        
        # Display raw results
        with st.expander("Raw Results Data"):
            if "raw_results" in result_data:
                st.json(result_data["raw_results"])
            elif "classification_results" in result_data:
                st.json(result_data["classification_results"])
            else:
                st.write("No raw API results available")