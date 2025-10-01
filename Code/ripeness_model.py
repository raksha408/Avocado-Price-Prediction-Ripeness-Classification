import streamlit as st
import numpy as np
from PIL import Image, ImageFile
from io import BytesIO
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Dropout
import base64

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Model paths ───────────────────────────────────────────────────────
MODEL_PATHS = {
    "EfficientNetB4": "avocado_ripeness/models/EffnetB4/finetuned_best.h5",
    "MobileNetV2":    "avocado_ripeness/models/MobilenetV2/final_mobv2_model_80_finished.h5",
    "ResNet50":       "avocado_ripeness/models/Resnet50/resnet50_finetuned_best.h5",
    "DenseNet201":    "avocado_ripeness/models/Densenet201/finetuned_dense201_best.h5"
}

# ── Confidence threshold adjustment (strict) ─────────────────────────
CONFIDENCE_THRESHOLD = 0.9  # require ≥90% to consider high confidence

# ── Paths for confusion matrices and graphs ─────────────────────────────
CONFUSION_MATRIX_PATHS = {
    "EfficientNetB4": "assets/matrix/EfficientNetB4_confusion.png",
    "MobileNetV2":    "assets/matrix/MobileNetV2_confusion.png",
    "ResNet50":       "assets/matrix/ResNet50_confusion.png",
    "DenseNet201":    "assets/matrix/confusion_dense80.png"
}
GRAPH_PATHS = {
    "EfficientNetB4": {"Accuracy": "assets/matrix/EfficientNetB4_graph1.png", "Loss": "assets/matrix/EfficientNetB4_graph2.png"},
    "MobileNetV2":    {"Accuracy": "assets/matrix/MobileNetV2_graph1.png",  "Loss": "assets/matrix/MobileNetV2_graph2.png"},
    "ResNet50":       {"Accuracy": "assets/matrix/ResNet50_graph1.png",     "Loss": "assets/matrix/ResNet50_graph2.png"},
    "DenseNet201":    {"Accuracy": "assets/matrix/DenseNet201_graph1.png",  "Loss": "assets/matrix/DenseNet201_graph2.png"}
}

# ── Custom objects for loading models ───────────────────────────────────
CUSTOM_OBJECTS = {"swish": swish, "FixedDropout": Dropout}

# ── Load all models ────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            st.error(f"❌ Model file not found: {abs_path}")
            continue
        try:
            models[name] = load_model(abs_path, custom_objects=CUSTOM_OBJECTS)
        except Exception as e:
            st.error(f"❌ Failed to load model {name}: {e}")
    return models

models = load_all_models()

# ── Streamlit UI Configuration ─────────────────────────────────────────
st.set_page_config(page_title="Avocado Ripeness Classification", page_icon="🥑", layout="centered")

# ── Custom CSS Styling ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #e6f4ea; }
    .cartoon-title { font-family: 'Comic Sans MS'; font-size:50px; color:#5aa469; text-align:center; font-weight:bold; text-shadow:2px 2px 8px rgba(90,164,105,0.5); margin-bottom:20px; }
    .stButton>button { width:100%; font-size:16px; padding:12px; border-radius:10px; border:2px solid #4caf50; background-color:#fff; color:#4caf50; font-weight:bold; transition:0.3s; }
    .stButton>button:hover { background-color:#4caf50; color:#fff; }
    .section-title { font-family:'Baloo Bhai 2'; font-size:28px; color:#4caf50; font-weight:700; margin-top:30px; margin-bottom:10px; }
    .section-subtitle { font-family:'Baloo Bhai 2'; font-size:20px; color:#388e3c; text-align:center; margin:10px 0 20px; }

    /* Hide Streamlit top bar including Deploy button */
    header {visibility: hidden;}
    /* Hide top-right hamburger menu */
    #MainMenu {visibility: hidden;}
    /* Hide "Made with Streamlit" footer */
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ── App Title ──────────────────────────────────────────────────────────
st.markdown("<div class='cartoon-title'>Avocado Ripeness Classification</div>", unsafe_allow_html=True)

# ── Model Selection ─────────────────────────────────────────────────────
st.markdown("<div class='section-title'> Choose Model</div>", unsafe_allow_html=True)
model_names = list(models.keys())
if not model_names:
    st.stop()

default_idx = model_names.index("EfficientNetB4") if "EfficientNetB4" in model_names else 0
if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_names[default_idx]

cols = st.columns(len(model_names))
for i, name in enumerate(model_names):
    if cols[i].button(name, key=name):
        st.session_state.selected_model = name

st.markdown(f"<div class='section-subtitle'> Selected Model: <span style='color:#388e3c'>{st.session_state.selected_model}</span></div>", unsafe_allow_html=True)

# ── Image Upload ───────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an Avocado Image", type=["png","jpg","jpeg"])
if not uploaded:
    st.stop()

uploaded.seek(0)
img_bytes = uploaded.read()
try:
    pil_img = Image.open(BytesIO(img_bytes)); pil_img.load(); img = pil_img.convert("RGB")
except:
    try:
        tf_img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = Image.fromarray(tf_img.numpy().astype("uint8"))
    except:
        st.error("Failed to open image. Please upload a valid JPG/PNG.")
        st.stop()

# ── Preprocess & Display ───────────────────────────────────────────────
H, W = models[st.session_state.selected_model].input_shape[1:3]
img_resized = img.resize((W, H))
img_array = np.asarray(img_resized)

buf = BytesIO()
img_resized.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

st.markdown(f"<div style='text-align:center;'><h4>Input Image</h4><img src='data:image/png;base64,{b64}' width='250'/></div>", unsafe_allow_html=True)
st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────
do_classify = st.button("🥑 Classify Ripeness")
if do_classify:
    x = np.expand_dims(img_array.astype("float32")/255.0, axis=0)
    preds = models[st.session_state.selected_model].predict(x, verbose=0)[0]

    labels = ["Breaking","Overripe","Ripe_First_Stage","Ripe_Second_Stage","Underripe"]

    if preds.shape[-1] != len(labels):
        st.error(f"⚠ Classes mismatch: {preds.shape[-1]} vs {len(labels)}")
        st.stop()

    preds = preds / preds.sum() if preds.sum() > 0 else preds
    scores_pct = np.minimum(preds * 100, 99.99)

    max_conf = np.max(scores_pct)
    idx = int(np.argmax(scores_pct))

    # Always display classification
    st.markdown(f"<h3 style='text-align:center;'>🥑 Classified as: <strong>{labels[idx]}</strong></h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>Confidence: <strong>{max_conf:.2f}%</strong></p>", unsafe_allow_html=True)

    # Confidence Scores Layout
    st.markdown("#### Confidence Scores")
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Breaking:** {scores_pct[0]:.2f}%")
    c2.write(f"**Ripe_First_Stage:** {scores_pct[2]:.2f}%")
    c3.write(f"**Underripe:** {scores_pct[4]:.2f}%")

    c4, c5, _ = st.columns(3)
    c4.write(f"**Overripe:** {scores_pct[1]:.2f}%")
    c5.write(f"**Ripe_Second_Stage:** {scores_pct[3]:.2f}%")

    # Confusion Matrix & Graphs
    cm = CONFUSION_MATRIX_PATHS.get(st.session_state.selected_model)
    if cm and os.path.exists(cm):
        st.markdown("#### Confusion Matrix")
        st.image(cm, use_container_width=True)

    for title, path in GRAPH_PATHS.get(st.session_state.selected_model, {}).items():
        if os.path.exists(path):
            st.markdown(f"#### {title}")
            st.image(path, use_container_width=True)
