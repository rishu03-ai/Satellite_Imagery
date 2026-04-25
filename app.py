import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("models/mobilenet_best.keras")

# Class labels
class_names = ["Landslide", "Non-Landslide"]

# Normalization values
mean = [0.2978, 0.3699, 0.3155]
std  = [0.1520, 0.1374, 0.1314]

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image

# UI
st.title("🌍 Landslide Detection System")

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        result = "Landslide"
        confidence = prediction
    else:
        result = "Non-Landslide"
        confidence = 1 - prediction

    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}")