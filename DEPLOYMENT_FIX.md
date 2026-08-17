# Streamlit Cloud Deployment - OpenCV Headless Fix

## Problem
The app was failing on Streamlit Cloud with: `ImportError: libGL.so.1: cannot open shared object file`

This occurred because:
1. The GUI version of OpenCV (`opencv-python`) was being installed alongside the headless version
2. The GUI version requires OpenGL libraries which aren't available in Streamlit Cloud's headless environment
3. The ultralytics library internally imports cv2, triggering the error at module load time

## Solution Implemented

### 1. **requirements.txt** (Updated)
- Removed `torchvision` (which may have opencv-python as an indirect dependency)
- Prioritized `opencv-python-headless>=4.5.2` at the top
- Added clear comments about the requirement
- Removed any unnecessary dependencies

### 2. **packages.txt** (Updated)
- Added `libgl1` and `ffmpeg` to supply system `libGL.so.1` and media dependencies required by OpenCV.
- Removed `#` comment lines from `packages.txt` because Streamlit Cloud's package installer treats comments as package names, causing `Unable to locate package #` errors during deployment.

### 3. **app.py** (Enhanced Error Handling)
- Sets environment variables before any imports:
  - `LIBGL_ALWAYS_INDIRECT=1` (software OpenGL rendering)
  - `QT_QPA_PLATFORM=offscreen` (disable GUI)
  - `OPENCV_LOG_LEVEL=SILENT` (suppress warnings)
  - `matplotlib.use('Agg')` (headless matplotlib)

- Wrapped ultralytics import with try-except to gracefully handle load failures
- Added model availability checks before inference operations
- Shows user-friendly error messages if model fails to load

### 4. **constraints.txt** (Created)
- Explicitly prevents GUI version installation
- Works with pip constraint system to prioritize headless version

## Expected Behavior

✅ **Best Case**: App loads successfully and inference works perfectly

⚠️ **Fallback**: If model still fails to load:
- User sees clear error messages
- Evaluation tabs still work (read pre-computed results)
- Error analysis displays correctly
- No cryptic import errors

## Testing Locally

To test the fix locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Simply push to GitHub - Streamlit Cloud will:
1. Read requirements.txt
2. Install opencv-python-headless (not GUI version)
3. Ignore packages.txt (no apt-get conflicts)
4. Launch app.py with proper headless configuration

The app should now deploy without the libGL error! 🚀
