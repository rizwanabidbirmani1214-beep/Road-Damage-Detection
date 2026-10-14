import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
from PIL import Image

# Page Configuration for the Streamlit web application
st.set_page_config(
    page_title="AI Road Damage & Pothole Inspector",
    layout="wide"
)

# Developer & Creator Credit Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Developer & AI Engineer")
st.sidebar.success("**Rizwan Abid**\n\n*Computer Vision & AI Researcher*\n*The University of Lahore*")
st.sidebar.markdown("---")

# Sidebar control for confidence threshold adjustment
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.01, 1.00, 0.25, 0.01)

st.title("AI Road Damage & Pothole Detection System")
st.markdown("Enterprise-grade YOLOv8-powered Road Inspection System for Images, Live Camera, and Videos.")

# Load the trained YOLOv8 model from GitHub repository weights
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}. Please ensure 'best.pt' is present in the GitHub repository.")

# Select input mode: Image upload, Live camera capture, or Dashcam video processing
option = st.radio("Select Input Mode:", ("Image Upload & Detection", "Live Camera Capture", "Dashcam Video Processing"))

if option == "Image Upload & Detection":
    uploaded_file = st.file_uploader("Upload a road surface image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured / Input Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("AI YOLOv8 Detection Result")
            with st.spinner("Analyzing road surface..."):
                results = model(image, conf=conf_threshold)
                res_plotted = results[0].plot()
                st.image(res_plotted, channels="BGR", use_container_width=True)

elif option == "Live Camera Capture":
    st.subheader("Live Camera Capture")
    st.markdown("Use your mobile or laptop camera to capture a road image for instant pothole detection.")
    
    # Built-in Streamlit camera input widget for mobile and desktop cameras
    camera_image = st.camera_input("Take a picture of the road")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Camera Frame")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("AI YOLOv8 Detection Result")
            with st.spinner("Analyzing live capture..."):
                results = model(image, conf=conf_threshold)
                res_plotted = results[0].plot()
                st.image(res_plotted, channels="BGR", use_container_width=True)

elif option == "Dashcam Video Processing":
    uploaded_video = st.file_uploader("Upload a dashcam video (.mp4 / .avi)...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save uploaded video to a temporary file for OpenCV processing
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        vidcap = cv2.VideoCapture(tfile.name)
        total_frames = int(vidcap.get(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) if hasattr(cv2, 'CAP_PROP_FRAME_COUNT') else 0))
        
        st.info("Video loaded successfully! Processing video stream...")
        
        st_frame = st.empty()
        pothole_counter = 0
        
        while vidcap.isOpened():
            success, frame = vidcap.read()
            if not success:
                break
                
            # Perform inference on each video frame
            results = model(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Accumulate total detections across all frames
            pothole_counter += len(results[0].boxes)
            
            # Render processed frame in real-time
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
        vidcap.release()
        st.success(f"Video processing complete! Total potholes detected across frames: {pothole_counter}")
