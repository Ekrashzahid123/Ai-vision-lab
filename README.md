# 🎯 Day-39 AI Vision Lab: End-to-End Object Detection & Error Analysis

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](app.py)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](weights/best.pt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](requirements.txt)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](#deployment)

An end-to-end Computer Vision application and comprehensive model evaluation suite built for the **Day-39 CV Challenge**. Powered by a custom YOLO object detection model (`best.pt`), an interactive Streamlit web dashboard, and quantitative benchmark analysis.

---

## 📌 Project Overview & What Model Detects

The custom YOLO object detector (`weights/best.pt`) detects 3 primary object classes in real time:
1. ☕ **Cup** (`class_id: 0`) — Coffee mugs, tea cups, paper cups, glass cups.
2. ✋ **Hand** (`class_id: 1`) — Human hands, gestures, grips holding objects.
3. 📱 **Phone** (`class_id: 2`) — Smartphones, mobile devices, screens.

---

## 📊 Model Performance & Benchmarks

The model was tested on **20 completely unseen test images** (`unseen_test_images/`) under challenging indoor lighting, reflective desk surfaces, and overlapping object arrangements.

| Metric | Score / Value | Description |
| :--- | :---: | :--- |
| **Test Dataset Size** | **20 Images** | 46 total ground-truth objects |
| **Correct Detections (TP)** | **41** | True Positives ($\text{IoU} \ge 0.50$) |
| **False Detections (FP)** | **3** | False Positives |
| **Missed Objects (FN)** | **5** | False Negatives |
| **Precision** | **93.18%** | $\text{TP} / (\text{TP} + \text{FP})$ |
| **Recall** | **89.13%** | $\text{TP} / (\text{TP} + \text{FN})$ |
| **mAP@50** | **91.50%** | Mean Average Precision at IoU 0.50 |
| **mAP@50-95** | **74.20%** | Mean Average Precision across IoU 0.50:0.95 |

---

## 🔬 Main Errors Found (5 Difficult Cases)

1. **Low-Confidence Duplicate Box (`20220307_191030`)**: Overlapping candidate proposals along adjacent vertical cup and phone borders caused a secondary 30% confidence box.
2. **Boundary Truncation (`20220307_191240`)**: Partial object cut off (~15% visible) at extreme image boundary lowered confidence to 39%.
3. **Specular Glare Reflection (`unseen_02_20220307_190939`)**: Metallic reflections on desk surface mimicked rectangular screen geometry, causing a 45% confidence false positive.
4. **Scale Variation / Distance (`unseen_05_20220307_190942`)**: Background small cup (<32x32 px) received lower confidence due to spatial downsampling strides.
5. **Hand-Grip Occlusion (`20220307_191510`)**: Shadow cast over hand holding phone obscured skin tones, resulting in a missed hand detection.

---

## 🚀 How the Application Works (`app.py`)

The Streamlit web application provides a multi-tab interface:
- 🖼️ **Image Inference Tab**: Upload custom images or pick from 20 pre-loaded unseen test photos. Adjust **Confidence Threshold** (0.01 - 1.00) and **IoU NMS Threshold** (0.10 - 0.95) live with instant visual redraw.
- 🎬 **Video Inference Tab**: Process short video files (MP4, AVI, MOV) frame-by-frame with progress status, FPS meter, and annotated MP4 video download.
- 📊 **Model Evaluation Tab**: View high-level metrics, per-class charts, and full 20-image summary matrices.
- 🔍 **Error Analysis Tab**: Explore the 5 difficult cases with side-by-side bounding box visualizations and root cause descriptions.
- 💾 **Export Options**: Download annotated prediction images (PNG), detection tables (CSV), and detection metadata (JSON).

---

## 📂 Repository Structure

```
d:\custom dataset and model\
├── weights/
│   └── best.pt                     # Trained custom YOLO model weights (~6.2 MB)
├── unseen_test_images/             # 20 real unseen test images
├── results/
│   ├── predictions/                # Annotated prediction images
│   └── evaluation/                 # Metrics summary charts, CSV, JSON
├── app.py                          # Interactive Streamlit web application
├── evaluate_model.py               # Quantitative evaluation & benchmarking script
├── requirements.txt                # Dependencies for local & Hugging Face deployment
├── evaluation_report.md            # Comprehensive model evaluation report
├── README.md                       # Project documentation
└── demo_guide.md                   # Video recording & deployment guide
```

---

## 🛠️ Local Installation & Running the App

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/day39-cv-yolo-app.git
cd day39-cv-yolo-app

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Model Evaluation Script
```bash
python evaluate_model.py
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🤗 Deployment to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Select **Streamlit** as the Space SDK.
3. Commit `app.py`, `requirements.txt`, `weights/best.pt`, `unseen_test_images/`, and `results/` to your Hugging Face Space repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/DAY39-YOLO-VISION-APP
   git push hf main
   ```
4. Hugging Face Spaces will automatically build and launch the application.

---

## 💡 What We Would Improve With More Time

1. **Dataset Balancing**: Collect more annotated hand gesture and grip samples under varied lighting to equalize class distribution.
2. **Multi-Scale Training**: Train with input resolutions up to $1024 \times 1024$ and add a P2 high-resolution feature layer for small background objects.
3. **Advanced Data Augmentations**: Incorporate Mosaic, HSV jittering, and Random Erasing to improve resilience against specular reflections and image edge truncations.
4. **Model Quantization & Acceleration**: Export model to ONNX / TensorRT / CoreML for ultra-fast edge device inference ($>100\text{ FPS}$).
