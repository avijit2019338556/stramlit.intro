import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Load Class Indices
working_dir = os.path.dirname(os.path.abspath(__file__))
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Ensure class indices file exists
if not os.path.exists(class_indices_path):
    st.error("Class indices file not found!")
    st.stop()

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)


# Function to Load and Preprocess the Image
def load_and_preprocess_image(image):
    # Ensure image is RGB format
    img = Image.open(image).convert("RGB")

    # Resize the image
    img = img.resize((224, 224))

    # Convert image to numpy array
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize

    # Ensure shape is (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Function to Load Model (Inside Streamlit Block)
@st.cache_resource()  # Caches the model for performance
def load_model():
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")

    # Ensure model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}!")
        st.stop()

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # Optional
    return model


# Function to Predict the Class of an Image
def predict_image_class(model, image):
    preprocessed_img = load_and_preprocess_image(image)

    # Debugging: Print shape and raw predictions
    st.write(f"🔹 Image shape: {preprocessed_img.shape}")
    predictions = model.predict(preprocessed_img)
    st.write(f"🔹 Raw Predictions: {predictions}")

    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Handle potential key errors in class mapping
    if str(predicted_class_index) not in class_indices:
        return "Unknown Class"

    return class_indices[str(predicted_class_index)]


# Streamlit App
st.title("🌱 Plant Disease Classifier")

# Load Model Here (Avoids Missing ScriptRunContext Issue)
model = load_model()

uploaded_image = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        if st.button("🔍 Classify"):
            prediction = predict_image_class(model, uploaded_image)
            st.success(f"Prediction: {prediction}")


import os
port = int(os.environ.get("PORT", 8501))