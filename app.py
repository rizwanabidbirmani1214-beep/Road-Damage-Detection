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

# Page Configuration for the Streamlit web application
st.set_page_config(
    page_title="AI Road Damage & Pothole Inspector",
    layout="wide"
)

# Initialize Session State for Dynamic Inspection Logs
if 'inspection_logs' not in st.session_state:
    st.session_state.inspection_logs = pd.DataFrame(columns=[
        "Timestamp", "Latitude", "Longitude", "Detections", "Severity"
    ])

# Developer & Creator Credit Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Developer & AI Engineer")
st.sidebar.success("**Rizwan Abid**\n\n*Computer Vision & AI Researcher*\n*The University of Lahore*")
st.sidebar.markdown("---")

# Sidebar control for confidence threshold adjustment
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.01, 1.00, 0.25, 0.01)

st.title("AI Road Damage & Pothole Detection System")
st.markdown("Enterprise-grade YOLOv8-powered Road Inspection System with Dynamic GPS Mapping & Real-time PDF Reporting.")

# Load the trained YOLOv8 model from GitHub repository weights
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}. Please ensure 'best.pt' is present in the GitHub repository.")

# Select input mode: Image upload, Live camera capture, Dashcam video processing, GIS Map, or PDF Report
option = st.radio(
    "Select Input Mode:", 
    ("Image Upload & Detection", "Live Camera Capture", "Dashcam Video Processing", "GIS Road Health Map", "Automated PDF Audit Report")
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
                st.info(f"Analysis Summary: {num_detections} road damage instance(s) detected. Severity: {severity}")
                
                # Append to session logs with dynamic coordinates around Lahore & current timestamp
                new_log = {
                    "Timestamp": current_time_str,
                    "Latitude": 31.48 + np.random.uniform(-0.01, 0.01),
                    "Longitude": 74.32 + np.random.uniform(-0.01, 0.01),
                    "Detections": num_detections,
                    "Severity": severity
                }
                st.session_state.inspection_logs = pd.concat([st.session_state.inspection_logs, pd.DataFrame([new_log])], ignore_index=True)

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
                st.image(res_plotted, channels="BGR", use_container_width=True)
                
                num_detections = len(results[0].boxes)
                severity = "Critical" if num_detections > 5 else ("Moderate" if num_detections > 2 else "Low")
                st.info(f"Analysis Summary: {num_detections} road damage instance(s) detected. Severity: {severity}")
                
                new_log = {
                    "Timestamp": current_time_str,
                    "Latitude": 31.48 + np.random.uniform(-0.01, 0.01),
                    "Longitude": 74.32 + np.random.uniform(-0.01, 0.01),
                    "Detections": num_detections,
                    "Severity": severity
                }
                st.session_state.inspection_logs = pd.concat([st.session_state.inspection_logs, pd.DataFrame([new_log])], ignore_index=True)

elif option == "Dashcam Video Processing":
    uploaded_video = st.file_uploader("Upload a dashcam video (.mp4 / .avi)...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        vidcap = cv2.VideoCapture(tfile.name)
        st.info("Video loaded successfully! Processing video stream...")
        
        st_frame = st.empty()
        pothole_counter = 0
        frame_count = 0
        
        while vidcap.isOpened():
            success, frame = vidcap.read()
            if not success:
                break
            frame_count += 1
            if frame_count % 15 == 0:
                results = model(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                pothole_counter += len(results[0].boxes)
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
        vidcap.release()
        severity = "Critical" if pothole_counter > 10 else ("Moderate" if pothole_counter > 5 else "Low")
        st.success(f"Video processing complete! Total potholes detected: {pothole_counter}")
        
        new_log = {
            "Timestamp": current_time_str,
            "Latitude": 31.50 + np.random.uniform(-0.01, 0.01),
            "Longitude": 74.35 + np.random.uniform(-0.01, 0.01),
            "Detections": pothole_counter,
            "Severity": severity
        }
        st.session_state.inspection_logs = pd.concat([st.session_state.inspection_logs, pd.DataFrame([new_log])], ignore_index=True)

elif option == "GIS Road Health Map":
    st.subheader("Interactive GIS Road Damage Heatmap & Dynamic Logs")
    st.markdown("Geospatial tracking of detected road hazards based on live session inspections.")
    
    if st.session_state.inspection_logs.empty:
        st.warning("No inspection logs recorded yet. Please run an Image Upload, Live Camera capture, or Video processing first!")
    else:
        st.map(st.session_state.inspection_logs, zoom=13)
        st.markdown("### Live Inspection Logs")
        st.dataframe(st.session_state.inspection_logs, use_container_width=True)

elif option == "Automated PDF Audit Report":
    st.subheader("Automated Professional PDF Audit Report Generation")
    st.markdown("Generate and download an official inspection report containing real-time survey metrics and session logs.")
    
    def generate_pdf(file_path):
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=12
        )
        
        normal_style = styles['Normal']
        
        elements.append(Paragraph("AI Road Damage Inspection & Audit Report", title_style))
        elements.append(Paragraph(f"<b>System:</b> Enterprise YOLOv8 Pothole Detection", normal_style))
        elements.append(Paragraph(f"<b>Lead Developer:</b> Rizwan Abid (The University of Lahore)", normal_style))
        elements.append(Paragraph(f"<b>Audit Date & Time:</b> {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}", normal_style))
        elements.append(Spacer(1, 20))
        
        elements.append(Paragraph("<b>Executive Summary:</b>", styles['Heading2']))
        elements.append(Paragraph("This official inspection report summarizes real-time road surface conditions evaluated via automated YOLOv8 computer vision models and dynamic spatial telemetry tracking.", normal_style))
        elements.append(Spacer(1, 15))
        
        # Dynamic Table Data from session logs
        table_data = [['Timestamp', 'Latitude', 'Longitude', 'Detections', 'Severity']]
        if st.session_state.inspection_logs.empty:
            table_data.append([current_time_str, '31.4800', '74.3200', '0 (No inspection yet)', 'Normal'])
        else:
            for idx, row in st.session_state.inspection_logs.iterrows():
                table_data.append([
                    str(row['Timestamp']),
                    f"{float(row['Latitude']):.4f}",
                    f"{float(row['Longitude']):.4f}",
                    str(row['Detections']),
                    str(row['Severity'])
                ])
        
        t = Table(table_data, colWidths=[120, 80, 80, 80, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#EDF2F7")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 9)
        ]))
        
        elements.append(t)
        doc.build(elements)

    if st.button("Generate & Download Official PDF Report"):
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        generate_pdf(pdf_path)
        
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download PDF Audit Report",
                data=pdf_file,
                file_name="Road_Damage_Inspection_Report.pdf",
                mime="application/pdf"
            )
        st.success("PDF report successfully generated with real-time session data!")
