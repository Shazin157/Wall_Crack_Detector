import os
import glob
import cv2
import streamlit as st
import numpy as np

from crack_inspector import run_inspection, CameraCalibration, ClassificationThresholds
from report_generator import build_report_text, render_annotated_figure

st.set_page_config(page_title="Wall Crack Inspector", layout="wide")
st.title("🧱 Automated Wall Crack Inspection")

# --- Sidebar: segmentation method ---------------------------------------
st.sidebar.header("Segmentation method")
method = st.sidebar.radio(
    "Which detector to use",
    ["Classical (rule-based, always available)", "Trained U-Net (needs a checkpoint file)"],
)

segmenter = None  # None = crack_inspector defaults to segment_crack_classical
if method.startswith("Trained"):
    checkpoint_path = st.sidebar.text_input("Checkpoint path", value="pretrained_unet.pt")
    if os.path.isfile(checkpoint_path):
        try:
            from unet_train_template import UNetSegmenter
            segmenter = UNetSegmenter(checkpoint_path=checkpoint_path)
            st.sidebar.success(f"Loaded {checkpoint_path}")
        except Exception as e:
            st.sidebar.error(f"Failed to load checkpoint: {e}")
            st.sidebar.warning("Falling back to classical method for this run.")
    else:
        st.sidebar.warning(f"'{checkpoint_path}' not found — falling back to classical method.")

# --- Sidebar: calibration & thresholds ---------------------------------
st.sidebar.header("Calibration")
mm_per_pixel = st.sidebar.number_input(
    "mm per pixel (calibrate once at your fixed camera distance)",
    min_value=0.001, max_value=5.0, value=0.15, step=0.01, format="%.3f",
)
st.sidebar.header("Classification thresholds")
hairline_max = st.sidebar.number_input("Hairline max (mm)", value=0.3, step=0.05)
medium_max = st.sidebar.number_input("Medium max (mm)", value=1.0, step=0.1)

calib = CameraCalibration(mm_per_pixel=mm_per_pixel)
thresholds = ClassificationThresholds(hairline_max_mm=hairline_max, medium_max_mm=medium_max)

# --- Ingestion mode -------------------------------------------------------
mode = st.radio("Image source", ["Upload manually", "Watch robot upload folder"], horizontal=True)

images_to_process = []

if mode == "Upload manually":
    uploaded = st.file_uploader(
        "Upload wall image(s) from the robot", type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            images_to_process.append((f.name, img))

else:
    folder = st.text_input("Folder path the robot uploads images into", value="./robot_uploads")
    if os.path.isdir(folder):
        paths = sorted(
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.png")) +
            glob.glob(os.path.join(folder, "*.tif")) +
            glob.glob(os.path.join(folder, "*.tiff"))
        )
        st.write(f"Found {len(paths)} image(s) in {folder}")
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                images_to_process.append((os.path.basename(p), img))
    else:
        st.info("Folder does not exist yet. Create it, or point the robot's upload script at it.")

# --- Run pipeline & display ------------------------------------------------
for name, img in images_to_process:
    st.divider()
    st.subheader(f"📷 {name}")

    report = run_inspection(img, calib=calib, thresholds=thresholds, segmenter=segmenter)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = render_annotated_figure(img, report)
        st.pyplot(fig)
    with col2:
        st.text(build_report_text(report, wall_id=name))
        if report["detected"]:
            severity_color = {"Hairline": "green", "Medium": "orange", "Large": "red"}.get(
                report["classification"], "gray"
            )
            st.markdown(f"**Severity:** :{severity_color}[{report['classification']}]")

if not images_to_process:
    st.info("Upload wall images, or point the app at your robot's upload folder, to generate reports.")
