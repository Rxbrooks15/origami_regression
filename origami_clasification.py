import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from datetime import datetime

# --- Load trained model ---
model = load_model("origami_image_classification.keras")

# --- Difficulty Map ---
difficulty_map = {0: "Easy", 1: "Intermediate", 2: "Complex"}

# --- Reference images ---
reference_images = {
    "Easy": "https://raw.githubusercontent.com/Rxbrooks15/origami_regression/main/origami_images/DSC00617-export-3000x3000.jpg",
    "Intermediate": "https://raw.githubusercontent.com/Rxbrooks15/origami_regression/main/origami_images/DSC02215-export-scaled.jpg",
    "Complex": "https://raw.githubusercontent.com/Rxbrooks15/origami_regression/main/origami_images/DSC03255-export-900x900.jpg"
}

# Initialize feedback dataset in session state
if "feedback_df" not in st.session_state:
    st.session_state.feedback_df = pd.DataFrame(columns=[
        "timestamp", "image_name", "edge_count", "confidence",
        "predicted_difficulty", "user_class", "rating_5scale", "feedback"
    ])

# --- Functions ---
def preprocess_image(image, IMG_SIZE=(128,128)):
    img = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img_rgb, img_bgr

def get_gradcam(model, img_batch, pred_class):
    last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        predictions = tf.reshape(predictions, (1, -1))
        loss = predictions[:, pred_class]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    heatmap = np.sum(conv_outputs * pooled_grads.numpy(), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    return cv2.resize(heatmap, (128,128))

# --- Streamlit App ---
st.title("📸 Origami Difficulty Classification: User Feedback Integration")
uploaded_file = st.file_uploader("Upload an Origami Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_rgb, img_bgr = preprocess_image(image)
    edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 75, 150)
    edge_count = np.sum(edges > 0)
    edges_norm = edges.astype(np.float32) / 255.0
    edges_norm = np.expand_dims(edges_norm, axis=-1)
    combined_input = np.concatenate([img_rgb, edges_norm], axis=-1)
    img_batch = np.expand_dims(combined_input, axis=0)

    preds = model.predict(img_batch)
    pred_class = np.argmax(preds[0])
    confidence = np.max(preds[0])
    predicted_difficulty = difficulty_map[pred_class]

    # Show results
    st.image(image, caption="Uploaded Origami Image", use_container_width=True)

    st.subheader("📌 Reference Difficulty Examples")
    cols = st.columns(3)
    for idx, (level, img_path) in enumerate(reference_images.items()):
        with cols[idx]:
            st.image(img_path, caption=f"{level} Example", use_container_width=True)

    # --- User Feedback Form ---
    with st.form(key="feedback_form"):
        user_class = st.radio("What do you think the difficulty should be?", 
                              ["Easy", "Intermediate", "Complex"])
        rating = st.radio("Rate Difficulty on 5-point Scale", 
                          ["Easy", "Moderate", "Intermediate", "Hard", "Complex"])
        feedback_text = st.text_area("Leave your feedback here")
        submit_button = st.form_submit_button("Submit Feedback")

        if submit_button:
            new_feedback = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": uploaded_file.name,
                "edge_count": int(edge_count),
                "confidence": round(float(confidence), 2),
                "predicted_difficulty": predicted_difficulty,
                "user_class": user_class,
                "rating_5scale": rating,
                "feedback": feedback_text
            }])

            st.session_state.feedback_df = pd.concat(
                [st.session_state.feedback_df, new_feedback], ignore_index=True
            )

            st.success("✅ Thank you! Your feedback has been saved.")

    # Display live dataset
    st.subheader("📊 Current Feedback Dataset")
    st.dataframe(st.session_state.feedback_df)

    # Download button for dataset
    csv = st.session_state.feedback_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Feedback CSV", csv, "user_feedback.csv", "text/csv")
