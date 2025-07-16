from tensorflow.keras.models import load_model

# Load mô hình
model = load_model('C:/Users/binhp/AI/AI_Flower_Classifier/models/flower_classifier_model.h5')

# Xem kiến trúc mô hình
model.summary()