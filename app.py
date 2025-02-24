
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from PIL import Image

# Model path
model_path = "trained_plant_disease_model.keras"
gdrive_url = "https://drive.google.com/file/d/1iPVEtnbga94anzsukYggwZnWpP1q2xec/view?usp=sharing"  # Modified Google Drive link

# Function to download the model if not found
def download_model():
    if not os.path.exists(model_path):
        st.warning("📥 Downloading model from Google Drive... ⏳")
        try:
            gdown.download(gdrive_url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"⚠ Model download failed: {e}")

# **Call download_model() before loading**
download_model()

# Load the trained model once and cache it
@st.cache_resource()
def load_model():
    try:
        model = tf.keras.models.load_model(model_path)  # Ensure correct path
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"⚠ Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Class labels (Modify based on dataset)
CLASS_NAMES = ['Potato_Early_blight', 'PotatoLate_blight', 'Potato_Healthy']

# Sidebar Navigation
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 HOME", "🔬 DISEASE RECOGNITION"])

# Image Preprocessing and Prediction
def model_prediction(image, model):
    """Process image and predict disease using the loaded model."""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Resize image to match model input size (128x128)
        img_resized = cv2.resize(img_array, (128, 128))

        # Expand dimensions to create a batch of size 1
        img_expanded = np.expand_dims(img_resized, axis=0)

        # Make a prediction
        predictions = model.predict(img_expanded)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        return predicted_index, confidence
    except Exception as e:
        st.error(f"⚠ Error during prediction: {e}")
        return None, None

# Main Page
if app_mode == "🏠 HOME":
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)
    st.write("This system helps farmers and agricultural researchers detect plant diseases efficiently.")

# Disease Recognition Page
elif app_mode == "🔬 DISEASE RECOGNITION":
    st.header("🔍 Plant Disease Detection System")
    
    test_image = st.file_uploader("📤 Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        image = Image.open(test_image).convert("RGB")  # Convert to RGB format
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)


        if st.button("🔍 Predict"):
            if model is None:
                st.error("⚠ Model could not be loaded. Please check the file path.")
            else:
                st.snow()  # Show animation effect
                st.write("⏳ Analyzing the image...")

                # Prediction
                result_index, confidence = model_prediction(image, model)

                if result_index is not None:
                    # Display Result
                    st.success(f"🩺 *Prediction:* {CLASS_NAMES[result_index]} ({confidence:.2f}% Confidence)")

                    # Show confidence as progress bar
                    st.progress(int(confidence))
                else:
                    st.error("❌ Prediction failed.")

if __name__ == "__main__":
    st.write("✅ Ready for Predictions")
