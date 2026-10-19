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

# Page Configuration
st.set_page_config(
    page_title="AI Road Damage & Pothole Inspector",
    page_layout="wide"
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
                st.image(res_plotted, channels="BGR", use_container_width=True)
                
                num_detections = len(results[0].boxes)
                severity = "Critical" if num_detections > 5 else ("Moderate" if num_detections > 2 else "Low")
                
                if severity == "Critical":
                    st.error(f"🚨 ALERT: Critical road damage detected! ({num_detections} instances)")
                elif severity == "Moderate":
                    st.warning(f"⚠️ WARNING: Moderate road wear detected. ({num_detections} instances)")
                else:
                    st.success(f"✅ Good condition. ({num_detections} instances)")
                
                new_log = {
                    "Timestamp": current_time_str,
                    "Location Name": exact_location_name,
                    "Detections": num_detections,
                    "Severity": severity
                }
                save_log(new_log)

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
        
        if severity == "Critical":
            st.error(f"🚨 Video Audit Complete: Critical Damage Level! Total Potholes: {pothole_counter}")
        else:
            st.success(f"Video processing complete! Total potholes detected: {pothole_counter}")
        
        new_log = {
            "Timestamp": current_time_str,
            "Location Name": exact_location_name,
            "Detections": pothole_counter,
            "Severity": severity
        }
        save_log(new_log)

elif option == "GIS Map & Analytics Dashboard":
    st.subheader("📊 Inspection Analytics & Persistent Logs Dashboard")
    st.markdown("Detailed breakdown of inspected zones, severity analytics, and permanent records.")
    
    logs_df = load_logs()
    
    if logs_df.empty:
        st.warning("No inspection logs recorded yet. Please run an Image Upload, Live Camera capture, or Video processing first!")
    else:
        # Analytics Section: Severity Distribution Chart
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Severity Breakdown")
            severity_counts = logs_df['Severity'].value_counts()
            st.bar_chart(severity_counts)
        with col_b:
            st.markdown("### Total Detections per Zone")
            if 'Location Name' in logs_df.columns and not logs_df.empty:
                loc_summary = logs_df.groupby('Location Name')['Detections'].sum()
                st.bar_chart(loc_summary)
        
        st.markdown("### 📋 Permanent Inspection Logs (Saved in Database)")
        st.dataframe(logs_df, use_container_width=True)
        
        if st.button("🗑️ Clear All Inspection History"):
            if os.path.exists(CSV_LOG_FILE):
                os.remove(CSV_LOG_FILE)
            st.session_state.inspection_logs = pd.DataFrame(columns=["Timestamp", "Location Name", "Detections", "Severity"])
            st.success("Inspection history cleared successfully!")
            st.rerun()

elif option == "Automated PDF Audit Report":
    st.subheader("Automated Professional PDF Audit Report Generation")
    st.markdown("Generate and download an official inspection report containing all persistent inspection logs and analytics.")
    
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
        elements.append(Paragraph("This official inspection report summarizes real-time road surface conditions evaluated via automated YOLOv8 computer vision models paired with permanent historical logging and analytics telemetry.", normal_style))
        elements.append(Spacer(1, 15))
        
        # Table Data from Persistent Logs
        logs_df = load_logs()
        table_data = [['Timestamp', 'Location / Area Name', 'Detections', 'Severity']]
        if logs_df.empty:
            table_data.append([current_time_str, exact_location_name, '0 (No inspection yet)', 'Normal'])
        else:
            for idx, row in logs_df.iterrows():
                table_data.append([
                    str(row['Timestamp']),
                    str(row['Location Name']),
                    str(row['Detections']),
                    str(row['Severity'])
                ])
        
        t = Table(table_data, colWidths=[120, 230, 70, 90])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#EDF2F7")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 8)
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
        st.success("PDF report successfully generated with analytics and historical data!")
