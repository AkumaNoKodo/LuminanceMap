import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. Příprava dat
# Předpokládejme, že obrázky jsou ve složce 'train_images' a mají názvy 'img_1.jpg', 'img_2.jpg', ...
# a odpovídající štítky jsou v listu `labels` (0 nebo 1)

image_paths = ['train_images/img_1.jpg', 'train_images/img_2.jpg', ...]
images = []
for path in image_paths:
    img = load_img(path, target_size=(128, 128))  # změní velikost obrázku na 128x128
    images.append(img_to_array(img))

x_train = np.array(images)
y_train = np.array(labels)  # předpokládejme, že `labels` je list [0, 1, 0, 1, ...]

# 2. Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Trénování
model.fit(x_train, y_train, epochs=5, validation_split=0.2)  # 20% dat bude použito pro validaci
