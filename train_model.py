# train_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
import cv2
from tqdm import tqdm
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# ================== تنظیمات ==================
IMG_SIZE = 224
NUM_CLASSES = 7
EPOCHS = 50
BATCH_SIZE = 64
CHANNELS = 3
CSV_PATH = "fer2013.csv"
LIMIT = 7000              # تعداد نمونه‌ها برای کاهش مصرف RAM
USE_ALIGNMENT = True        # فعال یا غیرفعال کردن تراز چهره با MTCNN
# ============================================

# بررسی وجود فایل CSV
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(" The file fer2013.csv was not found!")

# بارگذاری داده‌ها و کاهش حجم
data = pd.read_csv(CSV_PATH)
data = data.sample(n=LIMIT, random_state=42).reset_index(drop=True)

pixels = data['pixels'].tolist()
emotions = data['emotion'].values

# تعریف MTCNN برای تراز چهره
detector = MTCNN()

def align_face(img_gray):
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    faces = detector.detect_faces(img_rgb)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        aligned = img_rgb[y:y+h, x:x+w]
    else:
        aligned = img_rgb
    aligned = cv2.resize(aligned, (IMG_SIZE, IMG_SIZE))
    return aligned

# پردازش و تراز چهره‌ها
images = np.array([np.array(pix.split(), dtype='float32') for pix in pixels])
images = images.reshape(-1, 48, 48).astype('uint8')

print(" Processing and preparing images ...")
if USE_ALIGNMENT:
    aligned_images = [align_face(img) for img in tqdm(images)]
else:
    aligned_images = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (IMG_SIZE, IMG_SIZE)) for img in tqdm(images)]

aligned_images = np.array(aligned_images).astype('float32') / 255.0
labels = to_categorical(emotions, NUM_CLASSES)

# تقسیم داده‌ها
X_train, X_val, y_train, y_val = train_test_split(aligned_images, labels, test_size=0.1, random_state=42)

print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of validation samples: {X_val.shape[0]}")

# افزایش داده
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=[0.8, 1.2],
    shear_range=20,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# ساخت مدل MobileNetV2
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تعریف callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint("best_emotion_model.keras", save_best_only=True, monitor='val_accuracy')
]

# آموزش مدل
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks
)

# ذخیره کلاس‌ها
class_indices = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

print(" Model saved: best_emotion_model.keras")
