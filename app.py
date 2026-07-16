"""
Chest X-Ray Pneumonia Detection — Streamlit App
Deep learning (DenseNet121 transfer learning) with Grad-CAM explainability.

Educational project — NOT a diagnostic tool. See disclaimer in app.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chest X-Ray Disease Detection",
    page_icon="🫁",
    layout="wide"
)

MODEL_PATH = "final_model.keras"       # phase2 model, fine-tuned DenseNet121
DECISION_THRESHOLD = 0.4               # tuned for recall, see README
IMG_SIZE = (224, 224)

# ----------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def get_base_model(_model):
    """Find the nested DenseNet121 submodel inside the full model."""
    for layer in _model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested base model found.")

# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, full_model, base_model, last_conv_layer_name="relu"):
    x = tf.keras.applications.densenet.preprocess_input(img_array)

    conv_layer_model = tf.keras.Model(base_model.input, base_model.get_layer(last_conv_layer_name).output)

    base_model_index = next(i for i, layer in enumerate(full_model.layers) if layer is base_model)
    head_layers = full_model.layers[base_model_index + 1:]

    with tf.GradientTape() as tape:
        conv_output = conv_layer_model(x)
        tape.watch(conv_output)
        y = conv_output
        for layer in head_layers:
            y = layer(y)
        loss = y[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(pil_img, heatmap, alpha=0.4):
    img = pil_img.resize(IMG_SIZE)
    img_arr = np.array(img).astype("uint8")

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = heatmap_colored * alpha + img_arr
    superimposed = np.uint8(np.clip(superimposed, 0, 255))
    return superimposed


# ----------------------------------------------------------------------------
# Session state for prediction history
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------
# Sidebar — model info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Model Info")
    st.markdown("""
    **Architecture:** DenseNet121 (ImageNet pretrained, fine-tuned last 27 layers)

    **Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle, ~5,800 images

    **Decision threshold:** 0.4 (tuned to prioritize recall — minimizing missed pneumonia cases)

    **Test set performance:**
    | Metric | Score |
    |---|---|
    | Recall (Pneumonia) | 96.4% |
    | Precision | 83.4% |
    | Accuracy | 85.7% |

    **⚠️ Known limitation:** Grad-CAM analysis showed the model sometimes
    attends to non-anatomical regions (image markers, borders) rather than
    exclusively lung fields — a sign of possible shortcut learning from
    dataset artifacts. See project README for full analysis.
    """)

    st.divider()
    st.header("📊 Session History")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-10:]):
            st.write(f"`{h['time']}` — **{h['label']}** ({h['confidence']:.1%})")
    else:
        st.caption("No predictions yet this session.")

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("🫁 Chest X-Ray Disease Detection")
st.caption("CNN transfer learning (DenseNet121) with Grad-CAM explainability")

st.warning(
    "⚠️ **Educational project only.** This tool is NOT a certified diagnostic "
    "device and must not be used for real medical decisions. Always consult "
    "a qualified healthcare professional.",
    icon="⚠️"
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (JPEG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-Ray")
        st.image(pil_img, use_container_width=True)

    with st.spinner("Running model..."):
        model = load_model()
        base_model = get_base_model(model)

        img_resized = pil_img.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        prob_pneumonia = float(model.predict(img_array, verbose=0)[0][0])
        prob_normal = 1 - prob_pneumonia

        label = "PNEUMONIA" if prob_pneumonia >= DECISION_THRESHOLD else "NORMAL"
        confidence = prob_pneumonia if label == "PNEUMONIA" else prob_normal

        heatmap = make_gradcam_heatmap(img_array, model, base_model)
        overlay = overlay_gradcam(pil_img, heatmap)

    with col2:
        st.subheader("Grad-CAM Overlay")
        st.image(overlay, use_container_width=True)
        st.caption("Red/yellow regions show where the model focused most for this prediction.")

    st.divider()

    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        if label == "PNEUMONIA":
            st.error(f"### Prediction: {label}")
        else:
            st.success(f"### Prediction: {label}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.caption(f"Decision threshold: {DECISION_THRESHOLD} (tuned for recall)")

    with result_col2:
        st.subheader("Confidence Breakdown")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        classes = ["NORMAL", "PNEUMONIA"]
        probs = [prob_normal, prob_pneumonia]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.barh(classes, probs, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for bar, p in zip(bars, probs):
            ax.text(p + 0.02, bar.get_y() + bar.get_height()/2, f"{p:.1%}", va="center")
        ax.axvline(DECISION_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        st.pyplot(fig)

    # Log to session history
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "label": label,
        "confidence": confidence
    })

else:
    st.info("Upload a chest X-ray image above to get a prediction.")

st.divider()
st.caption(
    "Built with TensorFlow/Keras · DenseNet121 transfer learning · Grad-CAM · Streamlit  \n"
    "[GitHub Repo](#) — replace with your repo link"
)
