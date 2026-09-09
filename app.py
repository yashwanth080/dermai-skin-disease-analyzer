import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="DermAI | Skin Health Analyzer",
    page_icon="🔬",
    layout="wide"
)

# -------------------------------------------------
# LOAD HUGGINGFACE MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(
        "Ateeqq/skin-disease-prediction-exp-v1"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "Ateeqq/skin-disease-prediction-exp-v1"
    )
    return processor, model

processor, model = load_model()

# -------------------------------------------------
# THEME TOGGLE
# -------------------------------------------------
st.sidebar.markdown("## 🎨 Appearance")

mode = st.sidebar.radio(
    "Select Theme",
    ["Dark 🌙", "Light ☀️"],
    label_visibility="collapsed"
)

# -------------------------------------------------
# THEME COLORS
# -------------------------------------------------
if mode == "Dark 🌙":
    bg = "#0f172a"
    text = "#f8fafc"
    secondary_text = "#cbd5e1"
    card_bg = "#1e293b"
    sidebar_bg = "#111827"
    border = "#334155"
    input_bg = "#1e293b"
else:
    bg = "#f8fafc"
    text = "#0f172a"
    secondary_text = "#475569"
    card_bg = "#ffffff"
    sidebar_bg = "#e5e7eb"
    border = "#cbd5e1"
    input_bg = "#ffffff"

# -------------------------------------------------
# APPLY THEME USING CSS
# -------------------------------------------------
st.markdown(
    f"""
    <style>

    /* Main application */
    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    /* Main content */
    .main {{
        background-color: {bg};
        color: {text};
    }}

    /* All text */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp div {{
        color: {text};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{
        color: {text} !important;
    }}

    /* Hero */
    .hero {{
        background-color: {card_bg};
        border: 1px solid {border};
        border-radius: 25px;
        padding: 40px 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }}

    .hero-icon {{
        font-size: 60px;
    }}

    .hero-title {{
        color: {text} !important;
        font-size: 48px;
        font-weight: 800;
        margin: 5px 0;
    }}

    .hero-subtitle {{
        color: {text} !important;
        font-size: 22px;
        margin: 5px 0;
    }}

    .hero-description {{
        color: {secondary_text} !important;
        font-size: 14px;
        margin-top: 10px;
    }}

    /* Cards */
    .card {{
        background-color: {card_bg};
        border: 1px solid {border};
        border-radius: 20px;
        padding: 25px;
        margin-top: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }}

    .card h2 {{
        color: #22c55e !important;
    }}

    .card h4 {{
        color: {text} !important;
    }}

    .card p {{
        color: {secondary_text} !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: {card_bg};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 10px;
    }}

    /* Text input */
    input {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}

    /* Select box */
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}

    /* Number input */
    div[data-testid="stNumberInput"] input {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}

    /* Buttons */
    .stButton > button {{
        width: 100%;
        background-color: #22c55e;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        font-weight: 600;
    }}

    .stButton > button:hover {{
        background-color: #16a34a;
        color: white !important;
    }}

    /* Divider */
    hr {{
        border-color: {border};
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">🔬</div>
        <div class="hero-title">DermAI</div>
        <div class="hero-subtitle">Intelligent Skin Health Analyzer</div>
        <div class="hero-description">
            AI-assisted skin image classification powered by computer vision
        </div>
        <div class="badge">
            ✦ AI • COMPUTER VISION • SKIN HEALTH
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    """
    <style>
    .developer-footer {
        position: fixed;
        bottom: 15px;
        right: 22px;
        font-size: 13px;
        font-weight: 600;
        opacity: 0.7;
        z-index: 999999;
        padding: 7px 14px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,0.25);
        backdrop-filter: blur(10px);
    }
    </style>

    <div class="developer-footer">
        🔬 DermAI &nbsp;•&nbsp; Developed by <b>Yashwanth Nimmanagoti   wqd</b>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🩺 Skin Disease Analysis")

# -------------------------------------------------
# SIDEBAR PATIENT INFO
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("👤 Patient Profile")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120)
gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
st.markdown("### 📷 Upload Skin Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "png", "jpeg"]
)

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def predict(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    label = model.config.id2label[predicted_class.item()]
    conf = round(confidence.item() * 100, 2)

    return label, conf, probs[0].numpy()

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
if st.button("🔍 Analyze Skin Image"):

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Skin Image",
            width=350
        )

        # Get prediction
        result, confidence, probabilities = predict(image)

        # -------------------------------------------------
        # AI RESULT CARD
        # -------------------------------------------------
        st.markdown("## 🩺 AI Analysis Result")

        st.markdown(
            f"""<div class="card">
<div style="text-align:center;">
<div style="font-size:45px;">🩺</div>
<h2 style="color:#22c55e; margin:5px 0;">{result.upper()}</h2>
<p style="font-size:18px; margin:5px 0;">AI Confidence</p>
<h1 style="margin:5px 0;">{confidence}%</h1>
</div>
<hr>
<p style="text-align:center;">
This prediction is generated by an AI image-classification model.
</p>
<p style="text-align:center; font-size:13px; opacity:0.7;">
⚠️ This application is for educational/research purposes and should
not replace professional medical diagnosis.
</p>
</div>""",
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # PROBABILITY CHART
        # -------------------------------------------------
        labels = list(model.config.id2label.values())

        prob_data = {
            labels[i]: float(probabilities[i]) * 100
            for i in range(len(labels))
        }

        st.markdown("## 📊 Prediction Confidence")

        st.caption(
            "Confidence scores generated by the AI model for all supported classes."
        )

        st.bar_chart(prob_data)

    else:
        st.warning("⚠️ Please upload an image first.")