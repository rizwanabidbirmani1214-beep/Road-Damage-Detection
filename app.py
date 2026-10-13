import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Page Settings
st.set_page_config(
    page_title="AI Road Damage Detection & Severity System",
    page_icon="🛣️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 30px;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 15px;
        text-align: center;
        color: #4B5563;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛣️ AI Road Damage Detection & Severity Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Live Camera & Image AI Inspection System</div>', unsafe_allow_html=True)

# Load Custom Trained Model
MODEL_PATHS = [
    "best.pt",
    "runs/detect/road_damage_model-2/weights/best.pt",
    "runs/detect/road_damage_fast/weights/best.pt",
    "yolov8n.pt"
]

@st.cache_resource
def load_trained_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return YOLO(path), path
    return YOLO("yolov8n.pt"), "Base YOLOv8n"

model, loaded_path = load_trained_model()

# Severity Evaluation Logic
def calculate_severity(pothole_count, total_box_area_ratio):
    if pothole_count == 0:
        return "Normal / Safe Road", "Road condition is safe. No maintenance needed.", "green"
    elif pothole_count <= 2 and total_box_area_ratio < 5.0:
        return "Low Priority", "Minor damage detected. Add to scheduled maintenance.", "blue"
    elif pothole_count <= 4 and total_box_area_ratio < 12.0:
        return "Medium Priority", "Moderate damage. Repair recommended within 14 days.", "orange"
    else:
        return "High Priority (Critical)", "Severe road damage! Immediate repair required to prevent accidents.", "red"

# Sidebar Controls
st.sidebar.header("⚙️ System Control Panel")
st.sidebar.success(f"Model Active: {os.path.basename(loaded_path)}")

confidence = st.sidebar.slider("AI Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)

input_mode = st.sidebar.radio(
    "Choose Input Mode:",
    ("📸 Live Camera Snap", "📁 Upload Image File", "🗂️ Dataset Test Images")
)

st.sidebar.markdown("---")
st.sidebar.info("**Classes:** Pothole\n**Framework:** Ultralytics YOLOv8 + Streamlit")

selected_image = None

# Input Handling
if input_mode == "📸 Live Camera Snap":
    st.write("### 📸 Live Camera Capture")
    camera_photo = st.camera_input("Road ya pothole ki taraf camera karke 'Take Photo' dabayein:")
    if camera_photo is not None:
        selected_image = Image.open(camera_photo)

elif input_mode == "📁 Upload Image File":
    uploaded_file = st.file_uploader("Upload Road Picture (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)

else:
    dataset_folder = "pothole_dataset/images/valid"
    if os.path.exists(dataset_folder):
        available_images = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if available_images:
            chosen_file = st.selectbox("Choose Image from Validation Dataset:", available_images[:25])
            selected_image = Image.open(os.path.join(dataset_folder, chosen_file))
        else:
            st.warning("No images found in dataset valid folder.")
    else:
        st.info("Dataset folder not found. Please use Camera or Upload mode.")

# AI Prediction and Results
if selected_image is not None and model is not None:
    img_rgb = np.array(selected_image.convert("RGB"))
    img_h, img_w, _ = img_rgb.shape
    total_img_area = img_h * img_w
    
    with st.spinner("AI scanning image and calculating damage severity..."):
        results = model.predict(source=img_rgb, conf=confidence, imgsz=320)
        result = results[0]
        
        boxes = result.boxes
        potholes_count = len(boxes)
        total_pothole_pixels = 0
        
        if potholes_count > 0:
            for b in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b
                total_pothole_pixels += (x2 - x1) * (y2 - y1)
                
        area_percentage = round((total_pothole_pixels / total_img_area) * 100, 2)
        severity_label, action_text, color_code = calculate_severity(potholes_count, area_percentage)
        annotated_image = result.plot()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Captured / Input Image")
        st.image(selected_image, use_container_width=True)
    with col2:
        st.subheader("🎯 AI YOLOv8 Detection Result")
        st.image(annotated_image, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Damage & Severity Inspection Report")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Potholes Identified", f"{potholes_count}")
    m2.metric("Surface Damage Area", f"{area_percentage}%")
    m3.metric("Severity Level", severity_label)
    
    if "Critical" in severity_label:
        st.error(f"🚨 **Action Plan:** {action_text}")
    elif "Medium" in severity_label:
        st.warning(f"⚠️ **Action Plan:** {action_text}")
    elif "Low" in severity_label:
        st.info(f"ℹ️ **Action Plan:** {action_text}")
    else:
        st.success(f"✅ **Action Plan:** {action_text}")
