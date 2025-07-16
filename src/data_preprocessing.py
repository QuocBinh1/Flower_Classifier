import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, data_dir='flowers', img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.flower_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        
    def load_and_preprocess_data(self):
        """Tải và tiền xử lý dữ liệu ảnh"""
        images = []
        labels = []
        
        print("Đang tải dữ liệu...")
        for class_name in self.flower_classes:
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_path):
                print(f"Đang xử lý {class_name}...")
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            # Đọc và resize ảnh
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.resize(img, self.img_size) #resize ảnh về 224x224
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = img.astype(np.float32) / 255.0  # Chuẩn hóa pixel values [0,1]
                                
                                images.append(img)
                                labels.append(class_name)
                        except Exception as e:
                            print(f"Lỗi khi xử lý {img_path}: {e}")
        
        print(f"Đã tải {len(images)} ảnh từ {len(set(labels))} loài hoa")
        return np.array(images), np.array(labels)
    
    #chuyển nhãn thành số , daisy -> 0 . dandelion -> 1
    def encode_labels(self, labels):
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        return encoded_labels, label_encoder
    
    #chia train 60 , test 20 , val 20
    def split_data(self, images, labels, test_size=0.2, val_size=0.2):
        # Chia train và test
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Chia train thành train và validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessed_data(self, data_dict, filename='preprocessed_data.pkl'):
        """Lưu dữ liệu đã tiền xử lý"""
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Đã lưu dữ liệu vào {filename}")
    
    def plot_data_distribution(self, labels, title="Phân bố dữ liệu"):
        """Vẽ biểu đồ phân bố dữ liệu"""
        unique, counts = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts)
        plt.title(title)
        plt.xlabel('Loài hoa')
        plt.ylabel('Số lượng ảnh')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data_distribution.png')
        plt.show()

def main():
    # Khởi tạo preprocessor
    preprocessor = DataPreprocessor()
    
    # Tải và tiền xử lý dữ liệu
    images, labels = preprocessor.load_and_preprocess_data()
    
    # Mã hóa nhãn
    encoded_labels, label_encoder = preprocessor.encode_labels(labels)
    
    # Chia dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        images, encoded_labels
    )
    X_train, X_val, X_test = map(np.array, [X_train, X_val, X_test])

    
    # Lưu label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Lưu dữ liệu đã tiền xử lý
    data_dict = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'flower_classes': preprocessor.flower_classes
    }
    
    preprocessor.save_preprocessed_data(data_dict)
    
    # Vẽ biểu đồ phân bố dữ liệu
    preprocessor.plot_data_distribution(labels)
    
    print(f"Kích thước dữ liệu:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

if __name__ == "__main__":
    main() 