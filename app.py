import os
import sys

# Configure OpenCV for headless environments (required for Streamlit Cloud)
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'  # Force software rendering for OpenGL
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Disable GUI for Qt-based libraries

# Ensure headless mode for matplotlib and disable display
import matplotlib
matplotlib.use('Agg')

import io
import time
import json
import tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Import ultralytics with improved error handling for Streamlit Cloud
MODEL_LOAD_ERROR = None
try:
    from ultralytics import YOLO
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    YOLO = None

# Page Configuration
st.set_page_config(
    page_title="AI Vision Lab | YOLO Object Detector by Ekrash zahid",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Clean Light / White UI Theme)
st.markdown("""
<style>
    /* Clean Light / White Theme Overrides */
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Header Styling */
    .main-title {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .sub-title {
        color: #475569;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    
    /* Card Container */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
    }
    
    .metric-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-val {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }
    
    /* Class Badge Colors */
    .badge-cup { background-color: #E63946; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .badge-hand { background-color: #2A9D8F; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .badge-phone { background-color: #F4A261; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    
    /* Error Card */
    .error-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #F43F5E;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "best.pt")
UNSEEN_DIR = os.path.join(BASE_DIR, "unseen_test_images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Colors for bounding boxes (RGB & BGR)
CLASS_COLORS_RGB = {
    0: (230, 57, 70),   # Cup - Coral Red
    1: (42, 157, 143),  # Hand - Teal
    2: (244, 162, 97)   # Phone - Sand Gold
}

# Cache Model Loading
@st.cache_resource
def load_yolo_model(model_path):
    if YOLO is None:
        return None
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: `{model_path}`")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None

model = load_yolo_model(WEIGHTS_PATH)
CLASS_NAMES = model.names if model else {0: 'cup', 1: 'hand', 2: 'phone'}

# Check if model loaded successfully
if MODEL_LOAD_ERROR or model is None:
    st.warning(f"""
    ⚠️ **Model Loading Issue Detected**
    
    The app encountered an issue loading the YOLO model on this deployment.
    {f"Error: {MODEL_LOAD_ERROR}" if MODEL_LOAD_ERROR else "The model file may not be available."}
    
    This is typically a Streamlit Cloud compatibility issue with OpenCV/OpenGL graphics libraries.
    
    **Workaround**: Some features may be limited, but the app should display evaluation results and error analysis.
    """)


# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
st.sidebar.title("🎛️ Model Controls")
st.sidebar.markdown("---")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=1.00,
    value=0.25,
    step=0.01,
    help="Filter out detections with confidence below this value."
)

iou_threshold = st.sidebar.slider(
    "IoU (NMS) Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.45,
    step=0.05,
    help="Overlap threshold for Non-Maximum Suppression."
)

st.sidebar.markdown("### 🏷️ Filter Classes")
selected_classes = []
for cls_id, cls_name in CLASS_NAMES.items():
    if st.sidebar.checkbox(f"{cls_name.capitalize()}", value=True, key=f"cls_chk_{cls_id}"):
        selected_classes.append(cls_id)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Info")
st.sidebar.info(f"""
- **Model Target**: Custom YOLO
- **Task**: Object Detection
- **Classes**: {', '.join([c.capitalize() for c in CLASS_NAMES.values()])}
- **Weights Size**: ~6.2 MB
""")

# Main Header
st.markdown('<div class="main-title">🎯 AI Vision Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Object Detection, Video Processing, Performance Statistics & Error Analysis for Cup, Hand & Phone</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image Inference",
    "🎥 Video Inference & Live Camera",
    "📊 Model Evaluation",
    "🔍 Error Analysis (5 Cases)"
])

# Helper Function: Draw Bounding Boxes
def draw_predictions(image_pil, results, conf_thresh, iou_thresh, filter_classes):
    draw_img = image_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    
    detections = []
    boxes = results.boxes
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if conf < conf_thresh or cls_id not in filter_classes:
            continue
            
        xyxy = box.xyxy[0].tolist()
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color = CLASS_COLORS_RGB.get(cls_id, (0, 255, 0))
        
        # Bounding box
        draw.rectangle(xyxy, outline=color, width=3)
        
        # Label text
        label_text = f"{cls_name.upper()} {conf*100:.1f}%"
        text_bbox = draw.textbbox((xyxy[0], max(0, xyxy[1] - 22)), label_text)
        draw.rectangle([xyxy[0], max(0, xyxy[1] - 22), xyxy[0] + (text_bbox[2] - text_bbox[0]) + 10, max(0, xyxy[1])], fill=color)
        draw.text((xyxy[0] + 5, max(0, xyxy[1] - 18)), label_text, fill=(255, 255, 255))
        
        detections.append({
            "Class": cls_name.capitalize(),
            "Confidence": f"{conf*100:.2f}%",
            "Conf_Score": conf,
            "X1": round(xyxy[0], 1),
            "Y1": round(xyxy[1], 1),
            "X2": round(xyxy[2], 1),
            "Y2": round(xyxy[3], 1)
        })
        
    return draw_img, detections

# Note: Video processing with cv2 is not available in Streamlit Cloud
# This function is kept for reference but is not functional in cloud environments

# ---------------------------------------------------------
# TAB 1: IMAGE INFERENCE
# ---------------------------------------------------------
with tab1:
    st.markdown("### 📥 Select or Upload Image")
    
    input_source = st.radio(
        "Choose Input Source:",
        ["Choose from 20 Unseen Test Images", "Upload Custom Image"],
        horizontal=True
    )
    
    image_to_process = None
    image_name = "custom_image.jpg"
    
    if input_source == "Choose from 20 Unseen Test Images":
        if os.path.exists(UNSEEN_DIR):
            unseen_files = sorted([f for f in os.listdir(UNSEEN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            selected_file = st.selectbox("Select Unseen Test Sample:", unseen_files)
            if selected_file:
                image_path = os.path.join(UNSEEN_DIR, selected_file)
                image_to_process = Image.open(image_path).convert("RGB")
                image_name = selected_file
        else:
            st.warning(f"Unseen test image folder not found at: `{UNSEEN_DIR}`")
    else:
        uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name

    if image_to_process is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Input Image")
            st.image(image_to_process, use_container_width=True)
            
        with col2:
            st.markdown("#### 🎯 YOLO Detection Result")
            if model is None:
                st.error("❌ YOLO model is not available. Please check the deployment logs for details.")
                st.info("The model loading failed due to system library incompatibility on Streamlit Cloud.")
            else:
                start_t = time.time()
                
                # Run inference
                results = model(image_to_process, conf=0.01, iou=iou_threshold, verbose=False)[0]
                proc_time = (time.time() - start_t) * 1000
                
                annotated_img, detections = draw_predictions(
                    image_to_process, results, conf_threshold, iou_threshold, selected_classes
                )
                
                st.image(annotated_img, use_container_width=True)
                st.caption(f"⚡ Inference Time: `{proc_time:.1f} ms` | Active Confidence Cutoff: `{conf_threshold:.2f}`")

        st.markdown("---")
        st.markdown("### 📊 Detection Statistics Dashboard")
        
        if model is not None:
            # Count stats
            total_det = len(detections)
            cup_count = sum(1 for d in detections if d["Class"].lower() == "cup")
            hand_count = sum(1 for d in detections if d["Class"].lower() == "hand")
            phone_count = sum(1 for d in detections if d["Class"].lower() == "phone")
            avg_conf = np.mean([d["Conf_Score"] for d in detections]) * 100 if detections else 0.0
            
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Objects</div><div class="metric-val">{total_det}</div></div>', unsafe_allow_html=True)
            with m_col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Cups ☕</div><div class="metric-val" style="color:#E63946">{cup_count}</div></div>', unsafe_allow_html=True)
            with m_col3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Hands ✋</div><div class="metric-val" style="color:#2A9D8F">{hand_count}</div></div>', unsafe_allow_html=True)
            with m_col4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Phones 📱</div><div class="metric-val" style="color:#F4A261">{phone_count}</div></div>', unsafe_allow_html=True)
            with m_col5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Conf</div><div class="metric-val" style="color:#0284C7">{avg_conf:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            if detections:
                df_det = pd.DataFrame(detections).drop(columns=["Conf_Score"])
                st.markdown("#### 📋 Detailed Detections List")
                st.dataframe(df_det, use_container_width=True)
            else:
                st.info("No objects detected above the selected confidence threshold.")

            # Download Section
            st.markdown("---")
            st.markdown("### 💾 Save & Download Results")
            
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            # Prepare Image Download
            buf = io.BytesIO()
            annotated_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            with dl_col1:
                st.download_button(
                    label="📥 Download Result Image (PNG)",
                    data=byte_im,
                    file_name=f"yolo_result_{image_name}",
                    mime="image/png"
                )
                
            with dl_col2:
                csv_data = pd.DataFrame(detections).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Detection Report (CSV)",
                    data=csv_data,
                    file_name=f"detections_{image_name}.csv",
                    mime="text/csv"
                )
                
            with dl_col3:
                json_data = json.dumps(detections, indent=2)
                st.download_button(
                    label="📄 Download Detection JSON",
                    data=json_data,
                    file_name=f"detections_{image_name}.json",
                    mime="application/json"
                )

# ---------------------------------------------------------
# TAB 2: VIDEO INFERENCE & LIVE WEBCAM
# ---------------------------------------------------------
with tab2:
    st.markdown("### 🎥 Video Inference & Live Camera Detection")
    st.write("Run the custom YOLO model on **video files (MP4, AVI, MOV)**, **Sample Unseen Video Clip**, or **Live Camera Stream**.")
    
    video_source_mode = st.radio(
        "Choose Video Input Source:",
        ["📤 Upload Video File", "🎬 Demo Sample Video (Pre-loaded Unseen Clip)", "📸 Live Camera Snapshot / Stream"],
        horizontal=True
    )
    
    video_path_to_process = None
    
    if video_source_mode == "📤 Upload Video File":
        uploaded_video = st.file_uploader("Upload Video File (MP4, AVI, MOV):", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path_to_process = tfile.name
            st.video(video_path_to_process)
            
    elif video_source_mode == "🎬 Demo Sample Video (Pre-loaded Unseen Clip)":
        st.info("💡 **Video demo features are not available in Streamlit Cloud.** Please use the **Live Camera Snapshot** option or **Upload Video File** (if running locally).")

    elif video_source_mode == "📸 Live Camera Snapshot / Stream":
        st.markdown("#### 📷 Take a Camera Snapshot to Run Instant Detection")
        camera_photo = st.camera_input("Capture frame from your webcam:")
        if camera_photo is not None:
            if model is None:
                st.error("❌ YOLO model is not available. Camera detection requires the model to be loaded.")
            else:
                cam_pil = Image.open(camera_photo).convert("RGB")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### Captured Webcam Frame")
                    st.image(cam_pil, use_container_width=True)
                with c2:
                    st.markdown("##### YOLO Detection Result")
                    results = model(cam_pil, conf=0.01, iou=iou_threshold, verbose=False)[0]
                    ann_cam, cam_dets = draw_predictions(cam_pil, results, conf_threshold, iou_threshold, selected_classes)
                    st.image(ann_cam, use_container_width=True)
                    
                if cam_dets:
                    st.markdown("##### Detections List")
                    st.dataframe(pd.DataFrame(cam_dets).drop(columns=["Conf_Score"]), use_container_width=True)
                else:
                    st.info("No objects (cup, hand, phone) detected in captured frame.")

    # Process Video file if selected
    if video_path_to_process is not None and video_source_mode != "📸 Live Camera Snapshot / Stream":
        st.info("💡 **Video frame-by-frame processing is not available in Streamlit Cloud environments.** Video processing requires OpenCV (cv2) which has native dependencies not supported in the cloud. To use this feature, run the app locally.")
        st.button("🚀 Process & Annotate Entire Video", type="primary", disabled=True)

# ---------------------------------------------------------
# TAB 3: MODEL EVALUATION
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📊 Comprehensive Model Evaluation Dashboard")
    st.write("Quantitative benchmark results evaluated on **20 unseen test images** containing ground-truth cup, hand, and phone instances.")
    
    # Load Evaluation Summary JSON
    eval_json_path = os.path.join(RESULTS_DIR, "evaluation", "evaluation_summary.json")
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            eval_data = json.load(f)
            
        e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)
        with e_col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-val" style="color:#059669">{eval_data["precision"]*100:.1f}%</div></div>', unsafe_allow_html=True)
        with e_col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-val" style="color:#EA580C">{eval_data["recall"]*100:.1f}%</div></div>', unsafe_allow_html=True)
        with e_col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">mAP@50</div><div class="metric-val" style="color:#0284C7">{eval_data["mAP50"]*100:.1f}%</div></div>', unsafe_allow_html=True)
        with e_col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">mAP@50-95</div><div class="metric-val" style="color:#9333EA">{eval_data["mAP50_95"]*100:.1f}%</div></div>', unsafe_allow_html=True)
        with e_col5:
            st.markdown(f'<div class="metric-card"><div class="metric-label">F1-Score</div><div class="metric-val" style="color:#D97706">{eval_data["f1_score"]*100:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detection Counts Matrix
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Test Images</div><div class="metric-val">{eval_data["num_test_images"]}</div></div>', unsafe_allow_html=True)
        with c_col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Correct Detections (TP)</div><div class="metric-val" style="color:#059669">{eval_data["true_positives"]}</div></div>', unsafe_allow_html=True)
        with c_col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">False Detections (FP)</div><div class="metric-val" style="color:#DC2626">{eval_data["false_positives"]}</div></div>', unsafe_allow_html=True)
        with c_col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Missed Objects (FN)</div><div class="metric-val" style="color:#D97706">{eval_data["false_negatives"]}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Charts Display
        chart_col1, chart_col2 = st.columns(2)
        metrics_img_path = os.path.join(RESULTS_DIR, "evaluation", "metrics_summary.png")
        per_class_img_path = os.path.join(RESULTS_DIR, "evaluation", "per_class_metrics.png")
        
        with chart_col1:
            if os.path.exists(metrics_img_path):
                st.image(metrics_img_path, caption="Overall Model Evaluation Summary", use_container_width=True)
        with chart_col2:
            if os.path.exists(per_class_img_path):
                st.image(per_class_img_path, caption="Per-Class Performance Comparison", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Per-Class Metric Table")
        df_per_class = pd.DataFrame(eval_data["per_class_metrics"]).T
        df_per_class.index.name = "Class"
        df_per_class.reset_index(inplace=True)
        df_per_class["Class"] = df_per_class["Class"].str.capitalize()
        st.dataframe(df_per_class, use_container_width=True)
    else:
        st.info("Run `python evaluate_model.py` to generate the evaluation summary json file.")

# ---------------------------------------------------------
# TAB 4: ERROR ANALYSIS (5 DIFFICULT CASES)
# ---------------------------------------------------------
with tab4:
    st.markdown("### 🔍 Error Analysis: 5 Difficult Test Cases")
    st.write("Detailed deep-dive analysis into the 5 failure modes encountered by the model during unseen data testing.")
    
    diff_json_path = os.path.join(RESULTS_DIR, "evaluation", "difficult_cases.json")
    if os.path.exists(diff_json_path):
        with open(diff_json_path, "r") as f:
            difficult_cases = json.load(f)
            
        for case in difficult_cases:
            st.markdown(f"#### Case {case['id']}: {case['error_type']}")
            
            c1, c2 = st.columns([1, 1.2])
            with c1:
                img_path = os.path.join(RESULTS_DIR, "predictions", f"pred_{case['filename']}")
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Sample: {case['filename'][:25]}...", use_container_width=True)
                else:
                    orig_path = os.path.join(UNSEEN_DIR, case['filename'])
                    if os.path.exists(orig_path):
                        st.image(orig_path, use_container_width=True)
                        
            with c2:
                st.markdown(f"""
                <div class="error-card">
                    <p><b>🏷️ Class Affected:</b> <code>{case['class_affected'].upper()}</code></p>
                    <p><b>⚠️ Failure Description:</b> {case['description']}</p>
                    <p><b>🔬 Root Cause:</b> {case['root_cause']}</p>
                    <p><b>💡 Recommended Mitigation:</b> {case['mitigation']}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("Run `python evaluate_model.py` to populate difficult error cases.")
