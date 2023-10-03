import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Загрузка данных
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

# Load the model's saved weights.
model.load_weights('model.h5')

# Преобразуйте изображения в одномерные массивы
test_images_flat = test_images.reshape((-1, 784))

# Предсказание на первых 5 тестовых изображениях
predictions = model.predict(test_images_flat[:6])

# Выводим предсказания и фактические метки
print(np.argmax(predictions, axis=1))  # Предсказанные метки
print(test_labels[:6])  # Фактические метки
