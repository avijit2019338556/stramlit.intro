import os
import json
import gdown  # Add this import
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

# Function to Load Model
@st.cache_resource()
def load_model():
    # Google Drive direct download link
    MODEL_URL = "https://drive.google.com/uc?id=1E4-LXchopuKMMRFMgcDLvrqHIPywZTfu"
    MODEL_DIR = os.path.join(working_dir, "trained_model")
    MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_prediction_model.h5")
    
    # Create directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Download model if not found
    if not os.path.exists(MODEL_PATH):
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()
    
    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Rest of your code remains the same...
def load_and_preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image_class(model, image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices.get(str(predicted_class_index), "Unknown Class")

# Streamlit UI
st.title("🌱 Plant Disease Classifier")
model = load_model()
uploaded_image = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(uploaded_image).resize((150, 150)), caption="Uploaded Image")
    with col2:
        if st.button("🔍 Classify"):
            prediction = predict_image_class(model, uploaded_image)
            st.success(f"Prediction: {prediction}")

# Render port configuration
import os
port = int(os.environ.get("PORT", 8501))
