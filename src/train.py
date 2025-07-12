import pickle
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from model import FlowerClassifier
import os

def plot_training_history(history, save_path='training_history.png'):
    """Vẽ biểu đồ lịch sử huấn luyện"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    print("=== BẮT ĐẦU HUẤN LUYỆN MODEL PHÂN LOẠI HOA ===\n")
    
    # Kiểm tra xem đã có dữ liệu tiền xử lý chưa
    if os.path.exists('preprocessed_data.pkl'):
        print("Tải dữ liệu đã tiền xử lý...")
        with open('preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        label_encoder = data['label_encoder']
        flower_classes = data['flower_classes']
        
        print(f"Đã tải dữ liệu:")
        print(f"Train: {X_train.shape}")
        print(f"Validation: {X_val.shape}")
        print(f"Test: {X_test.shape}")
        
    else:
        print("Tiền xử lý dữ liệu...")
        preprocessor = DataPreprocessor()
        images, labels = preprocessor.load_and_preprocess_data()
        encoded_labels, label_encoder = preprocessor.encode_labels(labels)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            images, encoded_labels
        )
        
        # Lưu dữ liệu đã tiền xử lý
        data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'flower_classes': preprocessor.flower_classes
        }
        
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        flower_classes = preprocessor.flower_classes
    
    # Khởi tạo classifier
    classifier = FlowerClassifier(num_classes=len(flower_classes))
    
    # Huấn luyện model
    print("\nBắt đầu huấn luyện model...")
    history = classifier.train_model(
        X_train, y_train, X_val, y_val,
        model_type='transfer',  # Sử dụng transfer learning
        epochs=30,
        batch_size=32
    )
    
    # Đánh giá model
    print("\nĐánh giá model trên test set...")
    test_loss, test_accuracy = classifier.evaluate_model(X_test, y_test)
    
    # Vẽ biểu đồ lịch sử huấn luyện
    print("\nVẽ biểu đồ lịch sử huấn luyện...")
    plot_training_history(history)
    
    # Lưu model
    print("\nLưu model...")
    classifier.save_model('models/flower_classifier_model.h5')
    
    # Lưu thông tin model
    model_info = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'flower_classes': flower_classes,
        'label_encoder': label_encoder
    }
    
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\n=== HOÀN THÀNH HUẤN LUYỆN ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Model đã được lưu tại: models/flower_classifier_model.h5")

if __name__ == "__main__":
    main() 