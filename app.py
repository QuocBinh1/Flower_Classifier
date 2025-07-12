import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Thêm thư mục src vào path
sys.path.append('src')

from predict import FlowerPredictor

# Cấu hình trang
st.set_page_config(
    page_title="AI Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B9D;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B9D;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Tải model một lần và cache"""
    try:
        predictor = FlowerPredictor()
        return predictor
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        return None

def get_confidence_color(confidence):
    """Trả về màu sắc dựa trên độ tin cậy"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_prediction_chart(predictions):
    """Tạo biểu đồ dự đoán"""
    classes = [pred['class'] for pred in predictions]
    probabilities = [pred['probability'] for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities,
            y=classes,
            orientation='h',
            marker_color=['#FF6B9D' if i == 0 else '#4A90E2' for i in range(len(classes))],
            text=[f'{prob:.1%}' for prob in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Kết quả dự đoán",
        xaxis_title="Xác suất",
        yaxis_title="Loài hoa",
        xaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌸 AI Flower Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Phân loại 5 loài hoa: Daisy, Dandelion, Rose, Sunflower, Tulip</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🌺 Cài đặt")
    
    # Tải model
    with st.spinner("Đang tải model..."):
        predictor = load_model()
    
    if predictor is None:
        st.error("Không thể tải model. Vui lòng kiểm tra lại file model.")
        return
    
    # Thông tin model
    st.sidebar.markdown("### 📊 Thông tin Model")
    st.sidebar.info(f"**Loài hoa được hỗ trợ:** {', '.join(predictor.flower_classes)}")
    
    # Upload ảnh
    st.markdown('<h2 class="sub-header">📸 Upload ảnh hoa</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Chọn ảnh hoa để phân loại",
        type=['jpg', 'jpeg', 'png'],
        help="Hỗ trợ định dạng JPG, JPEG, PNG"
    )
    
    # Demo ảnh
    st.markdown("### 🎯 Hoặc chọn ảnh demo:")
    demo_cols = st.columns(5)
    
    demo_images = {}
    for i, flower_class in enumerate(predictor.flower_classes):
        with demo_cols[i]:
            # Tìm ảnh demo trong thư mục flowers
            demo_path = None
            flower_dir = os.path.join('flowers', flower_class)
            if os.path.exists(flower_dir):
                for img_name in os.listdir(flower_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        demo_path = os.path.join(flower_dir, img_name)
                        break
                    if demo_path:
                        break
            
            if demo_path and st.button(f"Demo {flower_class.title()}", key=f"demo_{flower_class}"):
                demo_images[flower_class] = demo_path
    
    # Xử lý ảnh
    image_to_process = None
    
    if uploaded_file is not None:
        # Xử lý ảnh upload
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã upload", use_column_width=True)
        
        # Lưu ảnh tạm thời
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        image_to_process = temp_path
        
    elif demo_images:
        # Xử lý ảnh demo
        flower_class, demo_path = list(demo_images.items())[0]
        image = Image.open(demo_path)
        st.image(image, caption=f"Demo ảnh {flower_class}", use_column_width=True)
        image_to_process = demo_path
    
    # Dự đoán
    if image_to_process:
        st.markdown('<h2 class="sub-header">🔮 Kết quả dự đoán</h2>', unsafe_allow_html=True)
        
        with st.spinner("Đang phân tích ảnh..."):
            try:
                # Dự đoán
                result = predictor.predict_image(image_to_process)
                
                # Hiển thị kết quả chính
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"**Loài hoa dự đoán:** {result['predicted_class'].title()}")
                    
                    confidence_class = get_confidence_color(result['confidence'])
                    st.markdown(f"**Độ tin cậy:** <span class='{confidence_class}'>{result['confidence']:.1%}</span>", 
                              unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Biểu đồ dự đoán
                    fig = create_prediction_chart(result['all_predictions'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Bảng chi tiết
                st.markdown("### 📋 Chi tiết dự đoán")
                prediction_df = {
                    'Loài hoa': [pred['class'].title() for pred in result['all_predictions']],
                    'Xác suất': [f"{pred['probability']:.1%}" for pred in result['all_predictions']],
                    'Thứ hạng': [pred['rank'] for pred in result['all_predictions']]
                }
                
                import pandas as pd
                df = pd.DataFrame(prediction_df)
                st.dataframe(df, use_container_width=True)
                
                # Xóa file tạm nếu có
                if image_to_process == "temp_upload.jpg" and os.path.exists("temp_upload.jpg"):
                    os.remove("temp_upload.jpg")
                    
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌸 AI Flower Classifier - Sử dụng Deep Learning để phân loại hoa 🌸</p>
        <p>Model: CNN với Transfer Learning | Framework: TensorFlow/Keras</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 