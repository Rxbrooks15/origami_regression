import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from datetime import datetime
import requests
import base64
import json
import io


# --- Load trained model ---
model = load_model("origami_image_classification.keras")

# --- Difficulty Map ---
difficulty_map = {0: "Easy", 1: "Intermediate", 2: "Complex"}

# --- Reference images for each difficulty (raw GitHub URLs) ---
reference_images = {
    "Easy": "https://raw.githubusercontent.com/Rxbrooks15/origami_regression/main/origami_images/DSC00617-export-3000x3000.jpg",
    "Intermediate": "https://raw.githubusercontent.com/Rxbrooks15/origami_regression/main/origami_images/DSC02215-export-scaled.jpg",
    "Complex": "https://raw.githubusercontent.com/Rxbrooks15/origami_regression/main/origami_images/DSC03255-export-900x900.jpg"
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

def save_image_to_github(image, uploaded_file, token, owner, repo):
    """Save user-uploaded image to GitHub repo under new_origami/ folder"""
    headers = {"Authorization": f"token {token}"}

    # Convert the uploaded image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode()

    # File path in your repo
    img_filename = f"new_origami/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
    img_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{img_filename}"

    # Create payload for GitHub
    img_payload = {
        "message": "Add new origami upload",
        "content": img_base64
    }

    # Upload to GitHub
    response = requests.put(img_url, headers=headers, data=json.dumps(img_payload))

    if response.status_code in [200, 201]:
        st.info("📤 Image successfully uploaded to GitHub (new_origami).")
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{img_filename}"
        return raw_url  # return direct link to raw image
    else:
        st.error(f"⚠️ Error uploading image: {response.json()}")
        return None

# --- Streamlit App ---
st.title("📸 Origami Difficulty Classifier: CNN + Edge Detection + Grad-CAM")

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
    predicted_label = difficulty_map[pred_class]

    # Grad-CAM
    heatmap = get_gradcam(model, img_batch, pred_class)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted((img_rgb*255).astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)

    # --- Show Results in 4 Panels ---
    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    axes[0,0].imshow(img_rgb)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis("off")

    axes[1,1].imshow(edges, cmap="gray")
    axes[1,1].set_title(f"Edge Map (Count: {edge_count})")
    axes[1,1].axis("off")

    axes[1,0].imshow(overlay)
    axes[1,0].set_title(f"Grad-CAM Heatmap\nPred: {predicted_label} ({confidence:.2f})")
    axes[1,0].axis("off")

    axes[0,1].text(0.5, 0.5,
                  f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}\nEdges: {edge_count}",
                  fontsize=14, ha="center", va="center")
    axes[0,1].set_title("Prediction Metrics")
    axes[0,1].axis("off")

    st.pyplot(fig)

    # Show uploaded image
    st.image(image, use_container_width=True, caption="Uploaded Origami Image")

    # Reference examples
    st.subheader("📌 Reference Difficulty Examples")
    cols = st.columns(3)
    for idx, (level, img_path) in enumerate(reference_images.items()):
        with cols[idx]:
            st.image(img_path, caption=f"{level} Example", use_container_width=True)

    # --- Feedback Form ---
    with st.form(key="feedback_form"):
        origami_is = st.radio("Is the model uploaded origami?",
                              ["Yes", "No"])
        rating = st.radio("What do you think the difficulty should be on a 5-point scale?",
                          ["Easy", "Moderate", "Intermediate", "Hard", "Complex"])
        user_class = st.radio("What do you think the difficulty should be on a 3-point scale?",
                              ["Easy", "Intermediate", "Complex"])
        
        feedback_text = st.text_area("Leave your feedback here or type N/a")
        submit_button = st.form_submit_button("Submit Feedback")

        if submit_button:
            owner = "Rxbrooks15"
            repo = "origami_regression"
            token = st.secrets["GITHUB_ORIGO_TOKEN"]
            feedback_file = "user_feedback.csv"

            # 1. Save image to GitHub
            image_url = save_image_to_github(image, uploaded_file, token, owner, repo)

            if image_url:
                # 2. Get existing CSV
                csv_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{feedback_file}"
                headers = {"Authorization": f"token {token}"}
                r = requests.get(csv_url, headers=headers)
                if r.status_code == 200:
                    file_data = r.json()
                    sha = file_data["sha"]
                    content = base64.b64decode(file_data["content"]).decode("utf-8")
                    df = pd.read_csv(io.StringIO(content))
                else:
                    sha = None
                    df = pd.DataFrame(columns=[
                        "timestamp", "image_name", "image_url", "edge_count", 
                        "confidence", "predicted_difficulty", 
                        "user_class", "rating_5scale", "feedback"
                    ])

                # 3. Add new feedback row
                new_feedback = pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_name": uploaded_file.name,
                    "image_url": image_url,
                    "edge_count": int(edge_count),
                    "confidence": round(float(confidence), 2),
                    "predicted_difficulty": predicted_label,
                    "user_class": user_class,
                    "rating_5scale": rating,
                    "feedback": feedback_text
                }])
                df = pd.concat([df, new_feedback], ignore_index=True)

                # 4. Upload updated CSV
                csv_data = df.to_csv(index=False)
                payload = {
                    "message": "Update feedback with new origami upload",
                    "content": base64.b64encode(csv_data.encode()).decode()
                }
                if sha:
                    payload["sha"] = sha

                put_r = requests.put(csv_url, headers=headers, data=json.dumps(payload))
                if put_r.status_code in [200, 201]:
                    st.success("✅ Feedback and uploaded image saved to GitHub!")
                else:
                    st.error(f"⚠️ Error saving feedback CSV: {put_r.json()}")
