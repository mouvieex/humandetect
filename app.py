import streamlit as st
try:
    import cv2
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Detektor Lyudey", page_icon="👤")
st.title("👤 Детектор Людей")


@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')


model = load_model()


uploaded_file = st.file_uploader("Загрузите изображение", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

    img_array = np.array(Image.open(uploaded_file))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


    results = model(img_bgr)


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])


            if class_id == 0 and confidence > 0.5:
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)


    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, use_column_width=True)

    
    result_bytes = cv2.imencode('.jpg', img_bgr)[1].tobytes()
    st.download_button(
        label="📥 Скачать",
        data=result_bytes,
        file_name="result.jpg",
        mime="image/jpeg"

    )
