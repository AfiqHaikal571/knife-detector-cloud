import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import tempfile

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
st.title("🔪 Knife Detector (Upload)")

media_type = st.radio("Pilih jenis media:", ["Gambar", "Video"])

if media_type == "Gambar":
    uploaded_image = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Gambar Asal", use_column_width=True)

        results = model.predict(image_np, conf=0.25)
        annotated = results[0].plot()
        st.image(annotated, caption="Hasil Pengesanan", use_column_width=True)

else:
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
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
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated)
        cap.release()
