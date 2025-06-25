import streamlit as st
import numpy as np
import pickle

# Import safe feature extraction from bytes
from features_safe import extract_features_from_bytes

# Load the model
@st.cache_resource
def load_model():
    try:
        with open("voice_emotion_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run train_model.py first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Extract features from uploaded file
def extract_features(file):
    try:
        audio_bytes = file.read()
        features = extract_features_from_bytes(audio_bytes)
        return features
    except Exception as e:
        st.error(f"❌ Failed to process audio: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="🎙️ AI Voice Mood Detector", page_icon="🎧")
st.title("🎙️ AI Voice Mood Detector")
st.write("Upload a `.wav` voice clip and I'll detect the emotion!")

# Load trained model
model_data = load_model()
if model_data is None:
    st.stop()

model = model_data['model']
scaler = model_data['scaler']
emotions = model_data['emotions']
feature_type = model_data.get('feature_type', 'basic')
n_features = model_data.get('n_features', 'unknown')

st.info(f"Model Info: SVM with {n_features} {feature_type} features")
st.info(f"Trained emotions: {', '.join(emotions)}")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("🔍 Analyzing audio..."):
        features = extract_features(uploaded_file)

        if features is not None:
            st.success(f"✅ Extracted {len(features)} features")

            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            confidence = model.predict_proba(features_scaled).max()

            st.success(f"🎯 Predicted Emotion: **{prediction.upper()}** ({confidence:.1%} confidence)")

# Help section
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. Record or upload a `.wav` file.
    2. The app will analyze your voice to predict your emotion.
    3. You'll get a top emotion label and confidence score.

    **Tips for best results:**
    - Use clear, expressive speech
    - Minimize background noise
    """)

# About section
with st.expander("🔬 About this model"):
    st.markdown(f"""
    This app uses machine learning to detect emotions in voice recordings.

    - **Algorithm**: SVM (Support Vector Machine)
    - **Features**: {n_features} `{feature_type}` features
    - **Emotions Trained**: {', '.join(emotions)}
    - Includes: MFCCs, spectral features, zero-crossing, tempo, etc.
    """)

