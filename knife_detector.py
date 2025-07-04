import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2

# BYPASS PyTorch weight-only loading error (for older PyTorch versions on Streamlit Cloud)
torch._storage_dtypes = {}

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🔪 Knife Detector (Upload Image or Video)")

media_type = st.radio("Pilih jenis media:", ["Gambar", "Video"])

if media_type == "Gambar":
    uploaded_image = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Gambar Asal", use_column_width=True)

        results = model.predict(image_np, conf=0.25)
        annotated = results[0].plot()
        st.image(annotated, caption="Hasil Pengesanan", use_column_width=True)

elif media_type == "Video":
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=0.25)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame)

        cap.release()
