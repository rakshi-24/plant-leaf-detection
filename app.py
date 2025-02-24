import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define class labels (Update this list according to your dataset)
CLASS_NAMES = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']

# Load the model once at startup to improve performance
@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please check the path.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the model (cached for performance)
model = load_model()

# Function to preprocess image
def preprocess_image(image):
    try:
        img = Image.open(image)
        img = img.resize((128, 128))  # Resize to match model input size
        img_array = np.array(img) / 255.0  # Normalize image (scale to [0,1])
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Function to make predictions
def predict_disease(image):
    if model is None:
        st.error("Model not loaded. Please check the model file.")
        return None

    img_array = preprocess_image(image)
    if img_array is None:
        return None

    try:
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100  # Get confidence score
        return CLASS_NAMES[predicted_class_idx], confidence, predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Streamlit UI Layout
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.radio("Navigation", ["🏠 Home", "🔍 Disease Recognition"])

# Home Page
if app_mode == "🏠 Home":
    st.title("Plant Disease Detection System for Sustainable Agriculture")
    st.write("""
    🌿 This application helps in identifying plant diseases using **Deep Learning**.  
    📸 Upload an image of a plant leaf, and our model will predict whether it is healthy or has a disease.  
    🔬 Built using **TensorFlow** and **Streamlit**.
    """)
    st.image("https://www.syngenta-us.com/uploads/Images/_ArticleImage2020/sick-leaves.jpg", use_column_width=True)

# Disease Recognition Page
elif app_mode == "🔍 Disease Recognition":
    st.title("🔍 Plant Disease Detection")
    st.write("📌 Upload an image of a plant leaf, and the model will predict the disease.")

    uploaded_file = st.file_uploader("📁 Choose an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("🚀 Predict"):
            with st.spinner("Processing... 🔄"):
                result = predict_disease(uploaded_file)
                if result:
                    predicted_class, confidence, probabilities = result

                    st.success(f"✅ Model Prediction: **{predicted_class}**")
                    st.write(f"📊 Confidence: **{confidence:.2f}%**")

                    # Show probabilities of all classes
                    st.write("### 🔢 Prediction Probabilities:")
                    for class_name, prob in zip(CLASS_NAMES, probabilities):
                        st.write(f"**{class_name}:** {prob * 100:.2f}%")

                    st.snow()  # 🎈 Celebrate successful prediction!

import gdown
import os

file_id = "1iPVEtnbga94anzsukYggwZnWpP1q2xec"
url = 'https://drive.google.com/file/d/1iPVEtnbga94anzsukYggwZnWpP1q2xec/view?usp=drive_link'
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
