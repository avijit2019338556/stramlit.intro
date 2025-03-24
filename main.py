import os
import json
import gdown
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Configure for Render deployment
if 'RENDER' in os.environ:
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Load Class Indices
working_dir = os.path.dirname(os.path.abspath(__file__))
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Ensure class indices file exists
if not os.path.exists(class_indices_path):
    st.error("Class indices file not found!")
    st.stop()

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Function to Load Model with robust error handling
@st.cache_resource()
def load_model():
    MODEL_URL = "https://drive.google.com/uc?id=1E4-LXchopuKMMRFMgcDLvrqHIPywZTfu"
    MODEL_DIR = os.path.join(working_dir, "trained_model")
    MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_prediction_model.h5")
    
    # Create directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Download model if not found
    if not os.path.exists(MODEL_PATH):
        try:
            temp_path = f"{MODEL_PATH}.temp"
            gdown.download(MODEL_URL, temp_path, quiet=False)
            
            # Verify download completed
            if os.path.exists(temp_path):
                # Wait for filesystem sync
                time.sleep(2)
                os.rename(temp_path, MODEL_PATH)
                os.chmod(MODEL_PATH, 0o644)  # Set proper permissions
                st.success("✅ Model downloaded successfully!")
            else:
                raise Exception("Download failed - file not created")
                
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()
    
    # Load the model with verification
    try:
        # Additional verification
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        file_size = os.path.getsize(MODEL_PATH)
        if file_size < 1000000:  # 1MB minimum expected size
            raise ValueError(f"Model file too small ({file_size} bytes), likely corrupted")
            
        # Add retry mechanism for loading
        max_retries = 3
        for attempt in range(max_retries):
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                model.compile(optimizer="adam", 
                            loss="categorical_crossentropy", 
                            metrics=["accuracy"])
                return model
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
                
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# Image processing function
def load_and_preprocess_image(image):
    try:
        img = Image.open(image).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"❌ Image processing failed: {str(e)}")
        st.stop()

# Prediction function
def predict_image_class(model, image):
    try:
        preprocessed_img = load_and_preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        return class_indices.get(str(predicted_class_index), "Unknown Class")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return "Error"

# Main app function
def main():
    st.title("🌱 Plant Disease Classifier")
    model = load_model()
    
    uploaded_image = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(uploaded_image).resize((150, 150)), 
                    caption="Uploaded Image")
        with col2:
            if st.button("🔍 Classify"):
                with st.spinner("Classifying..."):
                    prediction = predict_image_class(model, uploaded_image)
                    st.success(f"Prediction: {prediction}")

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    if 'RENDER' in os.environ:
        os.system(f"streamlit run {__file__} --server.port={port} --server.address=0.0.0.0")
    else:
        main()
