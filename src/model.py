import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class FlowerClassifier:
    def __init__(self, num_classes=5, img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
    #tạo model CNN
    def create_cnn_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_transfer_learning_model(self, base_model_name='mobilenet'):
        """Tạo model sử dụng transfer learning"""
        if base_model_name == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'resnet':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError("base_model_name phải là 'mobilenet' hoặc 'resnet'")
        
        # Đóng băng base model
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_data_augmentation(self):
        """Tạo data augmentation cho training"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile model với optimizer và loss function"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def create_callbacks(self, model_path='best_model.h5'):
        """Tạo callbacks cho training"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        return callbacks
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   model_type='transfer', epochs=50, batch_size=32):
        """Huấn luyện model"""
        print(f"Bắt đầu huấn luyện model {model_type}...")
        
        # Tạo model
        if model_type == 'cnn':
            self.model = self.create_cnn_model()
        elif model_type == 'transfer':
            self.model = self.create_transfer_learning_model('mobilenet')
        else:
            raise ValueError("model_type phải là 'cnn' hoặc 'transfer'")
        
        # Compile model
        self.model = self.compile_model(self.model)
        
        # Tạo callbacks
        callbacks = self.create_callbacks()
        
        # Tạo data augmentation
        datagen = self.create_data_augmentation()
        
        # Huấn luyện model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            steps_per_epoch=len(X_train) // batch_size,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Đánh giá model trên test set"""
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, image):
        """Dự đoán một ảnh"""
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện")
        
        # Reshape ảnh nếu cần
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Dự đoán
        predictions = self.model.predict(image)
        return predictions
    
    def save_model(self, filepath):
        """Lưu model"""
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện")
        
        self.model.save(filepath)
        print(f"Model đã được lưu tại {filepath}")
    
    def load_model(self, filepath):
        """Tải model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model đã được tải từ {filepath}")

def main():
    # Test tạo model
    classifier = FlowerClassifier()
    
    # Tạo model CNN
    cnn_model = classifier.create_cnn_model()
    print("CNN Model Summary:")
    cnn_model.summary()
    
    # Tạo transfer learning model
    transfer_model = classifier.create_transfer_learning_model()
    print("\nTransfer Learning Model Summary:")
    transfer_model.summary()

if __name__ == "__main__":
    main() 