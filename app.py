import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import torch

st.set_page_config(page_title="Детектор Людей", page_icon="👤", layout="wide")
st.title("👤 Детектор Людей")
st.markdown("### Быстрая детекция людей на фото")

@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')
    model.overrides['verbose'] = False
    return model

model = load_model()

st.sidebar.header("⚙️ Настройки")
confidence_threshold = st.sidebar.slider("Минимальная уверенность", 0.1, 1.0, 0.5, 0.1)
img_size = st.sidebar.selectbox("Размер изображения", [320, 480, 640], index=0)

uploaded_file = st.file_uploader("📤 Загрузите фото (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    original_size = img.size
    img_resized = img.resize((img_size, img_size))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="📷 Оригинал", use_container_width=True)
        st.caption(f"Размер: {original_size[0]}x{original_size[1]}")
    
    if st.button("🔍 Найти людей", type="primary", use_container_width=True):
        with st.spinner("⏳ Обрабатываю..."):
            try:
                img_array = np.array(img_resized)
                
                results = model(
                    img_array,
                    verbose=False,
                    conf=confidence_threshold,
                    iou=0.45,
                    max_det=100,
                    device='cpu',
                    half=False,
                    augment=False,
                    agnostic=False,
                )
                
                people_count = 0
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if cls_id == 0 and conf >= confidence_threshold:
                                people_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                scale_x = original_size[0] / img_size
                                scale_y = original_size[1] / img_size
                                x1 = int(x1 * scale_x)
                                y1 = int(y1 * scale_y)
                                x2 = int(x2 * scale_x)
                                y2 = int(y2 * scale_y)
                                
                                img_array_orig = np.array(img)
                                thickness = max(2, min(original_size) // 200)
                                
                                img_array_orig[y1:y1+thickness, x1:x2] = [0, 255, 0]
                                img_array_orig[y2-thickness:y2, x1:x2] = [0, 255, 0]
                                img_array_orig[y1:y2, x1:x1+thickness] = [0, 255, 0]
                                img_array_orig[y1:y2, x2-thickness:x2] = [0, 255, 0]
                                
                                img = Image.fromarray(img_array_orig)
                
                with col2:
                    st.image(img, caption=f"✅ Найдено: {people_count} чел.", use_container_width=True)
                
                st.success(f"🎯 Найдено людей: **{people_count}**")
                
                if people_count > 0:
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='JPEG', quality=95)
                    img_bytes.seek(0)
                    
                    st.download_button(
                        label=f"📥 Скачать результат ({people_count} чел.)",
                        data=img_bytes.getvalue(),
                        file_name="people_detected.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Ошибка обработки: {str(e)}")
                st.exception(e)

st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Советы:**
- Меньший размер = быстрее
- 320px: ~5-10 сек
- 640px: ~15-30 сек
""")


