import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Detektor Lyudey", page_icon="👤")
st.title("👤 Детектор Людей")


@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()


uploaded_file = st.file_uploader("Загрузите изображение", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file)
    
   
    img_array = np.array(img)
    
    
    results = model(img_array)
    
   
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
           
            if class_id == 0 and confidence > 0.5:
                
                img_array[y1:y2, x1:x1+2] = [0, 255, 0]  
                img_array[y1:y2, x2-2:x2] = [0, 255, 0]  
                img_array[y1:y1+2, x1:x2] = [0, 255, 0]  
                img_array[y2-2:y2, x1:x2] = [0, 255, 0]  
    
    result_img = Image.fromarray(img_array)
    st.image(result_img, use_column_width=True)
    
   
    from io import BytesIO
    img_bytes = BytesIO()
    result_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    st.download_button(
        label="📥 Скачать",
        data=img_bytes.getvalue(),
        file_name="result.jpg",
        mime="image/jpeg"
    )
