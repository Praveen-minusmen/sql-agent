import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from pathlib import Path
from warnings import filterwarnings
filterwarnings('ignore')


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Potato Disease Classification', layout='centered')

    # Modern light UI inspired by dashboard style
    page_background_color = """
    <style>
    :root {
        --bg: #f3f6fb;
        --card-bg: #ffffff;
        --text-main: #1a2f4d;
        --text-muted: #5f6f86;
        --primary: #1f7aec;
        --border: #dde7f5;
        --green-1: #1d6f28;
        --green-2: #67bf6b;
        --soft-green: #e8f5e9;
        --soft-warning: #fff4dd;
    }

    .stApp {
        background: var(--bg);
        color: var(--text-main);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: var(--card-bg);
        border-right: 1px solid var(--border);
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
    }

    .hero-wrap {
        background: linear-gradient(105deg, var(--green-1), var(--green-2));
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 10px 24px rgba(35, 90, 40, 0.25);
        margin-bottom: 1.1rem;
    }

    .hero-wrap h1 {
        margin: 0;
        font-size: 2.05rem;
        color: #ffffff;
        font-weight: 700;
        text-align: center;
    }

    .subtitle {
        margin-top: 0.55rem;
        color: #ecffef;
        font-size: 0.99rem;
        text-align: center;
    }

    .section-chip {
        background: #ece4d2;
        color: #5e4d2c;
        border-radius: 10px;
        padding: 0.55rem 0.85rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .result-card, .info-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 8px 20px rgba(26, 45, 75, 0.06);
    }

    .result-title {
        color: var(--text-main);
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
    }

    .result-value, .project-list {
        color: var(--text-main);
        font-size: 1rem;
        line-height: 1.65;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #f8fbff;
        border: 1.5px dashed #b9c8db;
        border-radius: 12px;
    }

    .prediction-pill {
        margin-top: 0.8rem;
        background: #1f7a35;
        color: #ffffff;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        font-weight: 700;
        display: inline-block;
    }

    .confidence-badge {
        margin-top: 0.8rem;
        background: var(--soft-green);
        border: 1px solid #cce6cf;
        color: #24663a;
        border-radius: 9px;
        padding: 0.6rem 0.75rem;
        font-weight: 600;
    }

    .warn-note {
        margin-top: 0.55rem;
        background: var(--soft-warning);
        border: 1px solid #f1dfb0;
        color: #7b5d17;
        border-radius: 9px;
        padding: 0.55rem 0.75rem;
        font-size: 0.92rem;
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # Hero section
    st.markdown(
        """
        <div class="hero-wrap">
            <h1>🥔 🌿 Potato Leaf Disease Detection</h1>
            <div class="subtitle">
                Upload a potato leaf image to detect plant health
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Streamlit Configuration Setup
streamlit_config()

MODEL_PATH = Path(__file__).resolve().parent / "model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def run_prediction(image_file):
    class_map = {
        'Potato___Early_blight': 'Early Blight',
        'Potato___Late_blight': 'Late Blight',
        'Potato___healthy': 'Healthy'
    }
    class_names = list(class_map.keys())
    model = load_model()
    img, img_array = preprocess_image(image_file)
    pred = model.predict(img_array, verbose=0)
    predicted_raw = class_names[int(np.argmax(pred))]
    predicted_class = class_map[predicted_raw]
    confidence = float(np.max(pred)) * 100
    return img, predicted_class, round(confidence, 2)


st.markdown("<div class='section-chip'>📤 Upload Image</div>", unsafe_allow_html=True)
input_image = st.file_uploader(label='Upload Image', type=['jpg', 'jpeg', 'png'])

if input_image is not None:
    image, predicted_class, confidence = run_prediction(input_image)
    confidence_ratio = min(max(confidence / 100, 0.0), 1.0)

    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown(
            "<div class='result-title'>📊 Confidence Level</div>",
            unsafe_allow_html=True
        )
        st.progress(confidence_ratio, text=f"{confidence:.2f}%")
        st.markdown(
            f"<div class='confidence-badge'>🔥 Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )
        if predicted_class == "Healthy":
            st.markdown(
                "<div class='warn-note'>✅ Leaf appears healthy. Keep monitoring crop hygiene.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='warn-note'>⚠️ Disease detected. Consider expert guidance for treatment.</div>",
                unsafe_allow_html=True
            )

    st.markdown(
        f"<div class='prediction-pill'>⚠️ Prediction: {predicted_class}</div>",
        unsafe_allow_html=True
    )

st.markdown("### ℹ️ Project Information")
st.markdown(
    """
    <div class='info-card'>
        <div class='project-list'>
            <b>Project:</b> Potato Leaf Disease Detection using Deep Learning<br>
            <b>Model:</b> CNN model trained to classify Early Blight, Late Blight, and Healthy leaves<br>
            <b>Tech Stack:</b> Python, TensorFlow/Keras, Streamlit<br>
            <b>Created by:</b> Praveen Ketannavar<br>
            <b>USN:</b> 2KE22CS098
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

