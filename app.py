import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Page Configuration (Corrected layout parameter)
st.set_page_config(
    page_title="AI Road Damage & Pothole Inspector",
    layout="wide"
)

# 1. Custom UI & Styling (Modern Enterprise CSS)
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stAlert {
        border-radius: 8px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Permanent Storage Setup (CSV Log Persistence)
CSV_LOG_FILE = "inspection_history.csv"

def load_logs():
    if os.path.exists(CSV_LOG_FILE):
        try:
            return pd.read_csv(CSV_LOG_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Timestamp", "Location Name", "Detections", "Severity"])

def save_log(new_row):
    df = load_logs()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_LOG_FILE, index=False)
    st.session_state.inspection_logs = df

if 'inspection_logs' not in st.session_state:
    st.session_state.inspection_logs = load_logs()

# Developer & Creator Credit Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Developer & AI Engineer")
st.sidebar.success("**Rizwan Abid**\n\n*Computer Vision & AI Researcher*\n*The University of Lahore*")
st.sidebar.markdown("---")

# Exact Location Input for Accurate Reporting
st.sidebar.header("📍 Location Settings")
exact_location_name = st.sidebar.text_input(
    "Inspection Location / Area Name:", 
    value="The University of Lahore, Defense Road"
)

st.sidebar.info(f"**Active Location:**\n\n{exact_location_name}")

# Model Confidence Threshold Slider
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.01, 1.00, 0.25, 0.01)

st.title("AI Road Damage & Pothole Detection System")
st.markdown("Enterprise-grade YOLOv8-powered Road Inspection System with Persistent Logs, Analytics & PDF Reporting.")

# Load YOLOv8 Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}. Please ensure 'best.pt' is in the GitHub repository.")

# Navigation Options
option = st.radio(
    "Select Input / Dashboard Mode:", 
    ("Image Upload & Detection", "Live Camera Capture", "Dashcam Video Processing", "GIS Map & Analytics Dashboard", "Automated PDF Audit Report")
)

current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                
                num_detections = len(results[0].boxes)
                severity = "Critical" if num_detections > 5 else ("Moderate" if num_detections > 2 else "Low")
                
                # Smart Alert System Banner
                if severity == "Critical":
                    st.error(f"🚨 ALERT: Critical road damage detected! ({num_detections} instances at {exact_location_name})")
                elif severity == "Moderate":
                    st.warning(f"⚠️ WARNING: Moderate road wear detected. ({num_detections} instances)")
                else:
                    st.success(f"✅ Good condition: Minor or no damage detected. ({num_detections} instances)")
                
                # Save to persistent storage
                new_log = {
                    "Timestamp": current_time_str,
                    "Location Name": exact_location_name,
                    "Detections": num_detections,
                    "Severity": severity
                }
                save_log(new_log)

elif option == "Live Camera Capture":
    st.subheader("Live Camera Capture")
    st.markdown("Use your mobile or laptop camera to capture a road image for instant pothole detection.")
    
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
                st.image(res
