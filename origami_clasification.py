import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# --- Load trained model ---
model = load_model("origami_image_classification.keras")

# --- Difficulty Map ---
difficulty_map = {0: "Easy", 1: "Intermediate", 2: "Complex"}

# --- Reference images for each difficulty ---
reference_images = {
    "Easy": "DSC00617-export-3000x3000.jpg",        # Rat
    "Intermediate": "DSC02215-export-scaled.jpg",   # Unicorn
    "Complex": "DSC03255-export-900x900.jpg"        # Dragon
}

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
st.title("📸 Origami Difficulty Classification: Application of Convolutional Neural Networks in the Realm of Origami")
uploaded_file = st.file_uploader("Upload an Origami Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read uploaded image
    image = Image.open(uploaded_file)

    # Preprocess
    img_rgb, img_bgr = preprocess_image(image)
    edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 75, 150)
    edge_count = np.sum(edges > 0)
    edges_norm = edges.astype(np.float32) / 255.0
    edges_norm = np.expand_dims(edges_norm, axis=-1)
    combined_input = np.concatenate([img_rgb, edges_norm], axis=-1)
    img_batch = np.expand_dims(combined_input, axis=0)

    # Predict
    preds = model.predict(img_batch)
    pred_class = np.argmax(preds[0])
    confidence = np.max(preds[0])

    # Grad-CAM
    heatmap = get_gradcam(model, img_batch, pred_class)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted((img_rgb*255).astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)

    # --- Show Results in 4 Panels ---
    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    axes[0,0].imshow(img_rgb)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis("off")

    axes[0,1].imshow(edges, cmap="gray")
    axes[0,1].set_title(f"Edge Map (Count: {edge_count})")
    axes[0,1].axis("off")

    axes[1,0].imshow(overlay)
    axes[1,0].set_title(f"Grad-CAM Heatmap\nPred: {difficulty_map[pred_class]} ({confidence:.2f})")
    axes[1,0].axis("off")

    axes[1,1].text(0.5, 0.5,
                  f"Predicted: {difficulty_map[pred_class]}\nConfidence: {confidence:.2f}\nEdges: {edge_count}",
                  fontsize=14, ha="center", va="center")
    axes[1,1].set_title("Prediction Metrics")
    axes[1,1].axis("off")

    st.pyplot(fig)

    # Show uploaded image at the end
    st.image(image, use_container_width=True, caption="Uploaded Origami Image")

    # --- Show Reference Images Before Rating ---
    st.subheader("📌 Reference Difficulty Examples")
    cols = st.columns(3)
    for idx, (level, img_path) in enumerate(reference_images.items()):
        with cols[idx]:
            st.image(img_path, caption=f"{level} Example", use_container_width=True)

    # --- Rating Section ---
    st.subheader("⭐ Rate the Prediction")
    rating = st.slider("How close was the prediction to what you expected?", 
                       min_value=1, max_value=5, value=3)
    
    # --- Feedback Section ---
    st.subheader("💬 Feedback")
    feedback_text = st.text_area("Leave your feedback here")
    
    # --- User's own classification ---
    st.subheader("🔎 Your Classification")
    user_class = st.radio("What do you think the difficulty should be?", 
                          ["Easy", "Intermediate", "Complex"])

    # --- Submit button ---
    if st.button("Submit Feedback"):
        st.success(f"✅ Thank you! You rated this {rating}/5, chose '{user_class}', and left feedback: {feedback_text}")

# --- Citations ---
st.markdown("""
---
**Acknowledgments**

This project leverages Convolutional Neural Networks to predict origami difficulty (easy, intermediate, and complex).  
It would not have been possible without:

- *OrigamiSet1.0: Two New Datasets for Origami Classification and Difficulty Estimation*  
  D. Ma, G. Friedland, M.M. Krell (2018). In *Proceedings of Origami Science, Mathematics, and Education (7OSME)*, Oxford, UK.  
- [Origami Database](https://origamidatabase.com) — includes high-resolution images, metadata, and external resources such as diagrams, video tutorials, and crease patterns.

These resources help bring origami into the digital world, supporting both scientific research and broader community access.
""")
