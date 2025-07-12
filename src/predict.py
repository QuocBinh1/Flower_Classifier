import cv2
import numpy as np
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt

class FlowerPredictor:
    def __init__(self, model_path='models/flower_classifier_model.h5', 
                 info_path='models/model_info.pkl'):
        self.model = None
        self.label_encoder = None
        self.flower_classes = None
        self.img_size = (224, 224)
        
        # Tải model và thông tin
        self.load_model_and_info(model_path, info_path)
    
    def load_model_and_info(self, model_path, info_path):
        """Tải model và thông tin liên quan"""
        try:
            # Tải model
            self.model = keras.models.load_model(model_path)
            print(f"Model đã được tải từ {model_path}")
            
            # Tải thông tin model
            with open(info_path, 'rb') as f:
                model_info = pickle.load(f)
            
            self.flower_classes = model_info['flower_classes']
            self.label_encoder = model_info['label_encoder']
            
            print(f"Thông tin model đã được tải:")
            print(f"Flower classes: {self.flower_classes}")
            print(f"Test accuracy: {model_info['test_accuracy']:.4f}")
            
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Tiền xử lý ảnh cho dự đoán"""
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        # Resize ảnh
        img = cv2.resize(img, self.img_size)
        
        # Chuyển từ BGR sang RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Chuẩn hóa pixel values
        img = img / 255.0
        
        return img
    
    def predict_image(self, image_path):
        """Dự đoán loài hoa từ ảnh"""
        # Tiền xử lý ảnh
        img = self.preprocess_image(image_path)
        
        # Thêm batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Dự đoán
        predictions = self.model.predict(img_batch)
        
        # Lấy kết quả dự đoán
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.flower_classes[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Tạo kết quả chi tiết
        results = []
        for i, (class_name, prob) in enumerate(zip(self.flower_classes, predictions[0])):
            results.append({
                'class': class_name,
                'probability': prob,
                'rank': i + 1
            })
        
        # Sắp xếp theo xác suất giảm dần
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': results,
            'image': img
        }
    
    def predict_from_array(self, image_array):
        """Dự đoán từ numpy array"""
        # Resize nếu cần
        if image_array.shape[:2] != self.img_size:
            image_array = cv2.resize(image_array, self.img_size)
        
        # Chuyển sang RGB nếu cần
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Chuẩn hóa
        image_array = image_array / 255.0
        
        # Thêm batch dimension
        img_batch = np.expand_dims(image_array, axis=0)
        
        # Dự đoán
        predictions = self.model.predict(img_batch)
        
        # Lấy kết quả
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.flower_classes[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def plot_prediction(self, image_path, save_path=None):
        """Vẽ ảnh và kết quả dự đoán"""
        # Dự đoán
        result = self.predict_image(image_path)
        
        # Tạo figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Hiển thị ảnh
        ax1.imshow(result['image'])
        ax1.set_title(f'Predicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.2%}')
        ax1.axis('off')
        
        # Hiển thị top 5 predictions
        top_5 = result['all_predictions'][:5]
        classes = [item['class'] for item in top_5]
        probabilities = [item['probability'] for item in top_5]
        
        bars = ax2.barh(classes, probabilities)
        ax2.set_xlabel('Probability')
        ax2.set_title('Top 5 Predictions')
        ax2.set_xlim(0, 1)
        
        # Thêm giá trị trên bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.2%}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return result

def main():
    # Test predictor
    predictor = FlowerPredictor()
    
    # Tìm một ảnh để test
    import os
    test_image = None
    
    for flower_class in ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']:
        flower_dir = os.path.join('flowers', flower_class)
        if os.path.exists(flower_dir):
            for img_name in os.listdir(flower_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(flower_dir, img_name)
                    break
            if test_image:
                break
    
    if test_image:
        print(f"Testing với ảnh: {test_image}")
        result = predictor.plot_prediction(test_image)
        print(f"Kết quả dự đoán: {result['predicted_class']} (Confidence: {result['confidence']:.2%})")
    else:
        print("Không tìm thấy ảnh để test")

if __name__ == "__main__":
    main() 