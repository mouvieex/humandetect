import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import gc

st.set_page_config(page_title="Detektor", page_icon="👤")
st.title(" Детектор людей")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

uploaded_file = st.file_uploader("Загрузить фото", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((320, 320))
    st.image(img, caption="Foto", use_container_width=True)
    
    if st.button(" Найти людей", type="primary"):
        with st.spinner("Процесс..."):
            try:
                arr = np.array(img)
                results = model(arr, verbose=False, conf=0.5)
                
                cnt = 0
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 0:
                            cnt += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            arr[y1:y2, x1:x1+2] = [255, 0, 0]
                            arr[y1:y2, x2-2:x2] = [255, 0, 0]
                            arr[y1:y1+2, x1:x2] = [255, 0, 0]
                            arr[y2-2:y2, x1:x2] = [255, 0, 0]
                
                res_img = Image.fromarray(arr)
                st.image(res_img, caption=f"✅ Найдено: {cnt}", use_container_width=True)
                
                buf = BytesIO()
                res_img.save(buf, format='JPEG', quality=60)
                buf.seek(0)
                
                st.download_button("📥 Загрузить изображение", buf.getvalue(), "result.jpg", "image/jpeg")
                
            except Exception as e:
                st.error(f"❌ {e}")
            finally:
                gc.collect()



