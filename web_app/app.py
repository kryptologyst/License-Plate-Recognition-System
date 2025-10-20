"""
Streamlit web interface for License Plate Recognition System.

This module provides a modern web interface for the license plate
recognition system using Streamlit.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import json
from typing import Optional, List, Dict, Any
import logging

# Import our modules
import sys
sys.path.append('src')
from license_plate_detector import LicensePlateDetector, LicensePlateResult
from advanced_detector import YOLOLicensePlateDetector, AdvancedPlateResult
from data_generator import VehicleImageGenerator, MockDatabase
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="License Plate Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class LicensePlateWebApp:
    """Main web application class for license plate recognition."""
    
    def __init__(self):
        """Initialize the web application."""
        self.config = get_config()
        self.detector = LicensePlateDetector()
        self.advanced_detector = YOLOLicensePlateDetector()
        self.data_generator = VehicleImageGenerator()
        self.database = MockDatabase()
        
        # Initialize session state
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        if 'synthetic_data_generated' not in st.session_state:
            st.session_state.synthetic_data_generated = False
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🚗 License Plate Recognition System</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Advanced computer vision system for automatic license plate detection and recognition
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.title("🎛️ Controls")
        
        # Detection method selection
        detection_method = st.sidebar.selectbox(
            "Detection Method",
            ["Traditional CV", "Advanced (YOLO + OCR)", "Both"],
            help="Choose the detection method to use"
        )
        
        # Confidence threshold
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        # Show processing time
        show_processing_time = st.sidebar.checkbox(
            "Show Processing Time",
            value=True,
            help="Display processing time in results"
        )
        
        # Generate synthetic data
        if st.sidebar.button("🎲 Generate Synthetic Data"):
            self.generate_synthetic_data()
        
        # Clear history
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()
        
        return {
            'detection_method': detection_method,
            'confidence_threshold': confidence_threshold,
            'show_processing_time': show_processing_time
        }
    
    def generate_synthetic_data(self):
        """Generate synthetic vehicle images."""
        with st.spinner("Generating synthetic data..."):
            try:
                dataset = self.data_generator.generate_dataset(5, "data/synthetic")
                st.session_state.synthetic_data_generated = True
                st.success(f"Generated {len(dataset)} synthetic images!")
            except Exception as e:
                st.error(f"Failed to generate synthetic data: {e}")
    
    def process_uploaded_image(self, uploaded_file, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an uploaded image."""
        try:
            # Convert uploaded file to numpy array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Could not decode the uploaded image")
                return None
            
            results = {}
            
            # Process based on selected method
            if settings['detection_method'] in ["Traditional CV", "Both"]:
                start_time = time.time()
                result = self.detector.detect_license_plate_from_array(image)
                processing_time = time.time() - start_time
                
                if result:
                    results['traditional'] = {
                        'result': result,
                        'processing_time': processing_time,
                        'method': 'Traditional CV'
                    }
            
            if settings['detection_method'] in ["Advanced (YOLO + OCR)", "Both"]:
                start_time = time.time()
                result = self.advanced_detector.detect_license_plate_from_array(image)
                processing_time = time.time() - start_time
                
                if result:
                    results['advanced'] = {
                        'result': result,
                        'processing_time': processing_time,
                        'method': 'Advanced (YOLO + OCR)'
                    }
            
            return results
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    def display_results(self, results: Dict[str, Any], settings: Dict[str, Any]):
        """Display detection results."""
        if not results:
            st.warning("No license plates detected in the image")
            return
        
        # Create columns for results
        cols = st.columns(len(results))
        
        for i, (method_key, method_result) in enumerate(results.items()):
            with cols[i]:
                result = method_result['result']
                processing_time = method_result['processing_time']
                method_name = method_result['method']
                
                # Filter by confidence threshold
                if result.confidence < settings['confidence_threshold']:
                    st.warning(f"{method_name}: Confidence too low ({result.confidence:.2f})")
                    continue
                
                # Display result
                st.markdown(f"### {method_name}")
                
                # Success box
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ License Plate Detected!</h4>
                    <p><strong>Plate Text:</strong> {result.plate_text}</p>
                    <p><strong>Confidence:</strong> {result.confidence:.2f}</p>
                    {f'<p><strong>Processing Time:</strong> {processing_time:.2f}s</p>' if settings['show_processing_time'] else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Display images
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Image with Detection**")
                    if hasattr(result, 'detection_method'):
                        # Advanced result
                        vis_image = self.advanced_detector.visualize_result(result)
                    else:
                        # Traditional result
                        vis_image = self.detector.visualize_result(result)
                    
                    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    st.image(vis_image_rgb, use_column_width=True)
                
                with col2:
                    st.markdown("**Extracted License Plate**")
                    plate_rgb = cv2.cvtColor(result.plate_image, cv2.COLOR_BGR2RGB)
                    st.image(plate_rgb, use_column_width=True)
                
                # Store in history
                history_entry = {
                    'timestamp': time.time(),
                    'method': method_name,
                    'plate_text': result.plate_text,
                    'confidence': result.confidence,
                    'processing_time': processing_time,
                    'bounding_box': result.bounding_box
                }
                st.session_state.detection_history.append(history_entry)
                
                # Store in database
                self.database.add_detection(
                    image_path="uploaded_image",
                    plate_text=result.plate_text,
                    confidence=result.confidence,
                    bounding_box=result.bounding_box,
                    detection_method=method_name,
                    processing_time=processing_time
                )
    
    def render_dashboard(self):
        """Render the analytics dashboard."""
        st.markdown("## 📊 Analytics Dashboard")
        
        if not st.session_state.detection_history:
            st.info("No detection history available. Upload some images to see analytics!")
            return
        
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.detection_history)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", len(df))
        
        with col2:
            st.metric("Average Confidence", f"{df['confidence'].mean():.2f}")
        
        with col3:
            st.metric("Unique Plates", df['plate_text'].nunique())
        
        with col4:
            st.metric("Avg Processing Time", f"{df['processing_time'].mean():.2f}s")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            fig_conf = px.histogram(df, x='confidence', nbins=20, 
                                  title='Confidence Score Distribution')
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Processing time distribution
            fig_time = px.histogram(df, x='processing_time', nbins=20,
                                  title='Processing Time Distribution')
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Method comparison
        if 'method' in df.columns:
            method_stats = df.groupby('method').agg({
                'confidence': 'mean',
                'processing_time': 'mean',
                'plate_text': 'count'
            }).round(2)
            
            st.markdown("### Method Comparison")
            st.dataframe(method_stats)
        
        # Recent detections table
        st.markdown("### Recent Detections")
        recent_df = df.tail(10)[['method', 'plate_text', 'confidence', 'processing_time']]
        st.dataframe(recent_df, use_container_width=True)
    
    def render_synthetic_data_section(self):
        """Render the synthetic data section."""
        st.markdown("## 🎲 Synthetic Data Generator")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Generate synthetic vehicle images with license plates for testing and demonstration.
            This is useful when you don't have real vehicle images available.
            """)
        
        with col2:
            if st.button("Generate 5 Sample Images", type="primary"):
                self.generate_synthetic_data()
        
        # Display generated images if available
        synthetic_dir = Path("data/synthetic")
        if synthetic_dir.exists() and any(synthetic_dir.glob("*.jpg")):
            st.markdown("### Generated Images")
            
            image_files = list(synthetic_dir.glob("*.jpg"))[:6]  # Show max 6 images
            
            cols = st.columns(3)
            for i, img_path in enumerate(image_files):
                with cols[i % 3]:
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption=img_path.name, use_column_width=True)
    
    def run(self):
        """Run the main application."""
        # Render header
        self.render_header()
        
        # Render sidebar
        settings = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analytics", "🎲 Synthetic Data"])
        
        with tab1:
            st.markdown("## Upload and Process Images")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a vehicle image to detect license plates"
            )
            
            if uploaded_file is not None:
                # Process image
                with st.spinner("Processing image..."):
                    results = self.process_uploaded_image(uploaded_file, settings)
                
                if results:
                    self.display_results(results, settings)
        
        with tab2:
            self.render_dashboard()
        
        with tab3:
            self.render_synthetic_data_section()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>License Plate Recognition System v1.0.0 | Built with Streamlit</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app."""
    app = LicensePlateWebApp()
    app.run()


if __name__ == "__main__":
    main()
