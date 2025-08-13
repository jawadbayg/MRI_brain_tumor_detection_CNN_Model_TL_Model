import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/brain_tumor_model.h5')

model = load_model()

# Define the class names
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Title
st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload an MRI image to detect the type of brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))  # ✅ MATCH YOUR MODEL'S INPUT SHAPE
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions)) * 100
    predicted_class = class_names[np.argmax(predictions)]

    # Output
    st.success(f"🎯 Predicted Tumor Type: **{predicted_class}**")
    st.info(f"📊 Model Confidence: **{confidence:.2f}%**")
