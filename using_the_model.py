import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

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

# Преобразование изображений в одномерный массив длиной 784
test_images = test_images.reshape((-1, 784))

# Предсказание на первых 5 изображениях
predictions = model.predict(test_images[:5])

# Визуализация предсказанных цифр и их истинных значений.
for i in range(5):
    # Отображение изображения
    plt.imshow(np.reshape(test_images[i], (28, 28)), cmap='gray')
    plt.show()
    # Печать предсказания модели и истинного значения
    print("Model's prediction:", np.argmax(predictions[i]))
    print("True label:", test_labels[i])
