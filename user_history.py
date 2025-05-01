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
    """Display the user's saved analysis history"""
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
                
            with st.expander(
                f"#{index + 1}: {result.get('fruit_type', 'Unknown')} - {formatted_date}"
            ):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Display basic information
                    st.write(f"**Fruit Type:** {result.get('fruit_type', 'Unknown')}")
                    st.write(f"**Analysis Type:** {result.get('analysis_type', 'Single Image')}")
                    
                    # Handle timestamp safely
                    try:
                        if timestamp:
                            formatted_full_date = datetime.datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            st.write(f"**Date:** {formatted_full_date}")
                        else:
                            st.write("**Date:** Unknown")
                    except ValueError:
                        st.write("**Date:** Invalid format")
                    
                    # Display ripeness predictions
                    st.write("**Ripeness Predictions:**")
                    for pred in result.get("ripeness_predictions", []):
                        st.write(f"- {pred.get('ripeness')}: {pred.get('confidence', 0):.2f}")
                
                with col2:
                    # Display the combined visualization if available
                    if "image_paths" in result and "combined_visualization" in result["image_paths"]:
                        img_path = result["image_paths"]["combined_visualization"]
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path)
                                st.image(img, width=400, caption="Analysis Visualization")
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                    elif "image_paths" in result and "original" in result["image_paths"]:
                        img_path = result["image_paths"]["original"]
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path)
                                st.image(img, width=300, caption="Original Image")
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                
                # Add buttons row at the bottom of the expander
                button_col1, button_col2 = st.columns(2)
                
                with button_col1:
                    # View details button
                    if st.button(f"View Full Details", key=f"view_{result.get('id')}", use_container_width=True):
                        st.session_state.selected_result_id = result.get('id')
                        st.session_state.view_details = True
                        st.rerun()
                
                with button_col2:
                    # Delete button
                    if st.button(f"🗑️ Delete", key=f"delete_{result.get('id')}", 
                              type="secondary", use_container_width=True):
                        # Set the current result for deletion confirmation
                        st.session_state.delete_confirmation = result.get('id')
                        st.rerun()
            
            # Check if this result is pending deletion confirmation
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
                
                # Track top ripeness prediction for each result
                if result.get('ripeness_predictions'):
                    top_ripeness = result['ripeness_predictions'][0]['ripeness']
                    if fruit_type not in ripeness_data:
                        ripeness_data[fruit_type] = {}
                    ripeness_data[fruit_type][top_ripeness] = ripeness_data[fruit_type].get(top_ripeness, 0) + 1
                
                # Track timestamps for activity chart - handle errors safely
                try:
                    timestamp = result.get('timestamp', '')
                    if timestamp:
                        timestamps.append(datetime.datetime.fromisoformat(timestamp))
                except ValueError:
                    # Skip invalid timestamps
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
    """Show detailed view of a single result"""
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
    
    # Debug expander remains the same
    with st.expander("Debug: Raw Result Data", expanded=False):
        st.write("Result ID:", result_id)
        st.write("Top-level keys:", list(result_data.keys()))
        
        if "image_paths" in result_data:
            st.write("Image path keys:", list(result_data["image_paths"].keys()))
            for key, path in result_data["image_paths"].items():
                st.write(f"- {key}: {path} (Exists: {os.path.exists(path) if path else 'No path'})")
        
        if "visualizations" in result_data:
            st.write("Visualization keys:", list(result_data["visualizations"].keys()))
    
    st.title("📋 Detailed Analysis Result")
    
    # Back button remains the same
    if st.button("← Back to History"):
        if "selected_result_id" in st.session_state:
            del st.session_state.selected_result_id
        if "view_details" in st.session_state:
            del st.session_state.view_details
        st.rerun()
    
    # Display metadata
    st.subheader(f"{result_data.get('fruit_type', 'Unknown')} Analysis")
    
    # EXTRACT TIMESTAMP FROM RESULT_ID
    # Format is typically: Username_YYYYMMDD_HHMMSS
    try:
        # Extract date and time from result_id
        if result_id and '_' in result_id:
            parts = result_id.split('_')
            if len(parts) >= 3:
                date_part = parts[-2]  # YYYYMMDD
                time_part = parts[-1]  # HHMMSS
                
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
                    # If format doesn't match expected pattern, fall back to existing timestamp
                    if "timestamp" in result_data and result_data["timestamp"]:
                        st.write(f"**Date:** {result_data['timestamp']}")
                    else:
                        st.write("**Date:** Unknown")
            else:
                st.write("**Date:** Cannot extract from result ID")
        else:
            # Fall back to timestamp in result_data if result_id extraction fails
            if "timestamp" in result_data and result_data["timestamp"]:
                st.write(f"**Date:** {result_data['timestamp']}")
            else:
                st.write("**Date:** Unknown")
    except Exception as e:
        # If any error occurs, try using timestamp from result_data
        if "timestamp" in result_data and result_data["timestamp"]:
            st.write(f"**Date:** {result_data['timestamp']}")
        else:
            st.write(f"**Date:** Unable to format date")
    
    st.write(f"**Analysis Type:** {'Multi-Angle' if result_data.get('multi_angle', False) else 'Single Image'}")
    
    # Display images
    image_paths = result_data.get("image_paths", {})
    
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
            # Try different possible keys for original image
            original_keys = ["original", "original_image_path"]
            original_found = False
            
            # Look first in image_paths
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
                                st.error(f"Error loading image ({key}): {e}")
            
            # Then look for keys with angle prefixes (for multi-angle)
            if not original_found and "image_paths" in result_data:
                angle_original_keys = [k for k in result_data["image_paths"].keys() 
                                    if k.startswith("original_")]
                if angle_original_keys:
                    img_path = result_data["image_paths"][angle_original_keys[0]]
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                            original_found = True
                        except Exception as e:
                            st.error(f"Error loading angle image: {e}")
            
            # If we still don't have an image, try looking in result_data directly
            if not original_found:
                if "original_image_path" in result_data:
                    img_path = result_data["original_image_path"]
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                            original_found = True
                        except Exception as e:
                            st.error(f"Error loading image from result_data: {e}")
                
                # Try angle results as last resort
                elif "angle_results" in result_data and result_data["angle_results"]:
                    for angle_result in result_data["angle_results"]:
                        if "original_image_path" in angle_result:
                            img_path = angle_result["original_image_path"]
                            if os.path.exists(img_path):
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, use_container_width=True)
                                    original_found = True
                                    break
                                except Exception as e:
                                    continue
            
            # If still no image found, display a message
            if not original_found:
                st.info("Original image not available")
        
        with col2:
            st.write("**Processed Image**")
            if "segmented" in image_paths:
                img_path = image_paths["segmented"]
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
            elif "bounding_box_visualization" in image_paths:
                img_path = image_paths["bounding_box_visualization"]
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
        
        # Show additional visualizations
        st.subheader("Additional Visualizations")
        
        # Create a list of all possible visualization types
        viz_types = ["comparison_visualization", "bounding_box_visualization", "combined_visualization"]
        viz_found = False
        
        for img_type in viz_types:
            if img_type in image_paths:
                img_path = image_paths[img_type]
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True, caption=img_type.replace("_", " ").title())
                        viz_found = True
                    except Exception as e:
                        st.error(f"Error loading {img_type}: {e}")
        
        # If no visualizations are found, check if they're in the result_data directly
        if not viz_found and "visualizations" in result_data:
            for viz_key, viz_path in result_data["visualizations"].items():
                if os.path.exists(viz_path):
                    try:
                        img = Image.open(viz_path)
                        st.image(img, use_container_width=True, caption=viz_key.replace("_", " ").title())
                        viz_found = True
                    except Exception as e:
                        st.error(f"Error loading visualization from result_data: {e}")
        
        # If still no visualizations found, display a message
        if not viz_found:
            st.info("No additional visualizations available")
    
    with ripeness_tab:
        # Display ripeness predictions
        st.subheader("Ripeness Predictions")
        
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
        if result_data.get("multi_angle", False) and "angle_results" in result_data:
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
                        else:
                            st.info("No ripeness predictions for this angle")
                
                # Create a comparison chart
                st.subheader("Angle Comparison")
                
                # Get all ripeness levels
                all_ripeness_levels = set()
                for result in angle_results:
                    if "ripeness_predictions" in result and result["ripeness_predictions"]:
                        for pred in result["ripeness_predictions"]:
                            all_ripeness_levels.add(pred.get("ripeness", "Unknown"))
                
                if all_ripeness_levels:
                    ripeness_levels = list(all_ripeness_levels)
                    ripeness_data = []
                    
                    for level in ripeness_levels:
                        level_data = {"ripeness": level}
                        for i, result in enumerate(angle_results):
                            confidence = 0
                            if "ripeness_predictions" in result and result["ripeness_predictions"]:
                                for pred in result["ripeness_predictions"]:
                                    if pred.get("ripeness", "") == level:
                                        confidence = pred.get("confidence", 0)
                                        break
                            level_data[angle_names[i]] = confidence
                        ripeness_data.append(level_data)
                    
                    # Create a dataframe
                    df = pd.DataFrame(ripeness_data)
                    
                    # Melt the dataframe for plotting
                    df_melted = pd.melt(
                        df, 
                        id_vars=["ripeness"], 
                        value_vars=angle_names,
                        var_name="Angle",
                        value_name="Confidence"
                    )
                    
                    # Create the plot
                    fig = px.bar(
                        df_melted,
                        x="ripeness",
                        y="Confidence",
                        color="Angle",
                        barmode="group",
                        title="Ripeness Predictions by Angle"
                    )
                    
                    st.plotly_chart(fig)
                else:
                    st.info("No ripeness predictions available for comparison")
    
    with technical_tab:
        # Display technical details about the analysis
        st.subheader("Technical Details")
        
        # Check if segmentation was used
        if "segmentation" in result_data:
            st.write(f"**Segmentation:** {'Enabled' if result_data.get('segmentation', False) else 'Disabled'}")
            
            if result_data.get("segmentation", False):
                st.write(f"**Segmentation Model:** SPEAR-net")
                st.write(f"**Refinement Method:** {result_data.get('refinement_method', 'all')}")
                
                if "mask_metrics" in result_data:
                    st.write("**Mask Quality Metrics:**")
                    metrics = result_data.get("mask_metrics", {})
                    st.write(f"- Mask coverage: {metrics.get('coverage_ratio', 0):.2%} of image")
                    st.write(f"- Boundary complexity: {metrics.get('boundary_complexity', 0):.2f}")
        
        # Display raw results
        if "raw_results" in result_data:
            with st.expander("Raw API Results"):
                st.json(result_data.get("raw_results", {}))