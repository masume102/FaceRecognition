import os
import json
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# غیرفعال‌سازی هشدار oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# مسیر فایل‌های مدل و کلاس‌ها
MODEL_PATH = 'final_emotion_model.keras'
CLASS_INDICES_PATH = 'class_indices.json'
IMG_WIDTH, IMG_HEIGHT = 48, 48

# بارگذاری مدل
if not os.path.exists(MODEL_PATH):
    st.error(" Model 'final_emotion_model.keras' not found.")
    st.stop()

if not os.path.exists(CLASS_INDICES_PATH):
    st.error(" Model 'class_indices.json' not found.")
    st.stop()

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

class_names = list(class_indices.values())

# تابع پیش‌پردازش
def preprocess_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# تابع پیش‌بینی احساس
def predict_emotion(image_np):
    processed = preprocess_image(image_np)
    prediction = model.predict(processed, verbose=0)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    return class_names[class_id], confidence

# رابط کاربری Streamlit
st.set_page_config(page_title="Face Emotion Detection", layout="centered")
st.title("Emotion Recognition from Face")
st.write("Upload an image or activate the webcam to detect facial emotions.")

# بخش آپلود تصویر
uploaded_file = st.file_uploader(" Upload face image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    emotion, confidence = predict_emotion(image_np)
    st.success(f" Detected Emotion: **{emotion}** ({confidence:.2%} Confidence)")

# وب‌کم
st.subheader("Real-time Detection with Webcam")
run_webcam = st.checkbox("Activate Webcam")

# بارگذاری مدل تشخیص چهره
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if run_webcam:
    st.warning("To stop, uncheck the box above.")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while st.session_state.get("run_webcam", True):
        ret, frame = cap.read()
        if not ret:
            st.error("Error in receiving image from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion, confidence = predict_emotion(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.0%})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
#streamlit run streamlit_app.py