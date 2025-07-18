# Dự án Phân loại Hoa AI

## Mô tả
Dự án này sử dụng Deep Learning để phân loại 5 loài hoa khác nhau:
- Daisy (Hoa cúc)
- Dandelion (Bồ công anh)
- Rose (Hoa hồng)
- Sunflower (Hoa hướng dương)
- Tulip (Hoa tulip)

## Cấu trúc dự án
```
Project_AI_Flowers/
├── flowers/                 # Dữ liệu huấn luyện
│   ├── daisy/              # 506 ảnh hoa cúc
│   ├── dandelion/          # 1049 ảnh bồ công anh
│   ├── rose/               # 781 ảnh hoa hồng
│   ├── sunflower/          # 730 ảnh hoa hướng dương
│   └── tulip/              # 981 ảnh hoa tulip
├── src/                    # Mã nguồn
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── models/                 # Lưu trữ model đã train
├── app.py                  # Ứng dụng Streamlit
├── requirements.txt        # Thư viện cần thiết
└── README.md
```

## Cài đặt

1. Cài đặt Python 3.8+ và pip
2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Tiền xử lý dữ liệu
```bash
python src/data_preprocessing.py
```

### 2. Huấn luyện model
```bash
python src/train.py
```

### 3. Chạy ứng dụng web
```bash
streamlit run app.py
```

## Tính năng
- Phân loại 5 loài hoa với độ chính xác cao
- Giao diện web thân thiện với người dùng
- Upload ảnh và dự đoán real-time
- Hiển thị kết quả với độ tin cậy
- Hỗ trợ nhiều định dạng ảnh (JPG, PNG, JPEG)

## Model Architecture
- Sử dụng CNN (Convolutional Neural Network)
- Transfer Learning với pre-trained models
- Data augmentation để tăng độ đa dạng dữ liệu
- Early stopping và model checkpointing

## Kết quả mong đợi
- Độ chính xác validation: >90%
- Thời gian dự đoán: <1 giây
- Hỗ trợ ảnh với kích thước khác nhau #   F l o w e r _ C l a s s i f i e r  
 