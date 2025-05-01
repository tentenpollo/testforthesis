import streamlit as st
import datetime
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from user_management import get_user_results, get_result_details

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
                
                # Add button to view full details
                if st.button(f"View Full Details", key=f"view_{result.get('id')}"):
                    st.session_state.selected_result_id = result.get('id')
                    st.session_state.view_details = True
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
    
    st.title("📋 Detailed Analysis Result")
    
    # Add a back button
    if st.button("← Back to History"):
        if "selected_result_id" in st.session_state:
            del st.session_state.selected_result_id
        if "view_details" in st.session_state:
            del st.session_state.view_details
        st.rerun()
    
    # Display metadata
    st.subheader(f"{result_data.get('fruit_type', 'Unknown')} Analysis")
    
    # Handle timestamp safely
    try:
        timestamp = result_data.get('timestamp', '')
        if timestamp:
            formatted_date = datetime.datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            st.write(f"**Date:** {formatted_date}")
        else:
            st.write("**Date:** Unknown")
    except ValueError:
        st.write("**Date:** Invalid format")
        
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
            if "original" in image_paths:
                img_path = image_paths["original"]
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
        
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
        for img_type in ["comparison_visualization", "bounding_box_visualization"]:
            if img_type in image_paths and img_type != "combined_visualization":
                img_path = image_paths[img_type]
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True, caption=img_type.replace("_", " ").title())
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
    
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
                st.write(f"**Segmentation Model:** UNet-ResNet50")
                st.write(f"**Refinement Method:** {result_data.get('refinement_method', 'all')}")
                
                if "mask_metrics" in result_data:
                    st.write("**Mask Quality Metrics:**")
                    metrics = result_data.get("mask_metrics", {})
                    st.write(f"- Number of objects: {metrics.get('num_objects', 0)}")
                    st.write(f"- Mask coverage: {metrics.get('coverage_ratio', 0):.2%} of image")
                    st.write(f"- Boundary complexity: {metrics.get('boundary_complexity', 0):.2f}")
        
        # Display raw results
        if "raw_results" in result_data:
            with st.expander("Raw API Results"):
                st.json(result_data.get("raw_results", {}))