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
# --- Load Models ---
# Recreate architecture
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

binary_model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Load weights
binary_model.load_weights("origami_yesno_final.keras")
binary_model.build(input_shape=(None, 224, 224, 3))

difficulty_model = load_model("origami_image_classification.keras")  # Difficulty classifier

# --- Difficulty Map ---
difficulty_map = {0: "Easy", 1: "Intermediate", 2: "Complex"}

st.title("📸 Origami Classifier: Binary + Difficulty (CNN + Grad-CAM)")

uploaded_file = st.file_uploader("Upload an Origami Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read uploaded image
    image = Image.open(uploaded_file)

    # --- Step 1: Run Origami/Not Origami ---
    img_binary = image.resize((224,224)).convert("RGB")
    img_bin_array = np.expand_dims(np.array(img_binary), axis=0) / 255.0

    bin_pred = binary_model.predict(img_bin_array)
    is_origami = bin_pred[0][0] > 0.5
    bin_confidence = float(bin_pred[0][0])
    origami_label = "Origami" if is_origami else "Not Origami"

    # --- Step 2: If Origami, run Difficulty Model ---
    if is_origami:
        img_rgb, img_bgr = preprocess_image(image)
        edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 75, 150)
        edge_count = np.sum(edges > 0)
        edges_norm = edges.astype(np.float32) / 255.0
        edges_norm = np.expand_dims(edges_norm, axis=-1)
        combined_input = np.concatenate([img_rgb, edges_norm], axis=-1)
        img_batch = np.expand_dims(combined_input, axis=0)

        preds = difficulty_model.predict(img_batch)
        pred_class = np.argmax(preds[0])
        diff_confidence = np.max(preds[0])
        predicted_label = difficulty_map[pred_class]

        # Grad-CAM
        heatmap = get_gradcam(difficulty_model, img_batch, pred_class)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted((img_rgb*255).astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)
    else:
        predicted_label, diff_confidence, edge_count, overlay, edges = "N/A", 0, 0, None, None

    # --- Show Results in 4 Panels ---
    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    axes[0,0].imshow(np.array(image))
    axes[0,0].set_title("Original Image")
    axes[0,0].axis("off")

    # Metrics panel
    metrics_text = (
        f"Origami Prediction: {origami_label}\n"
        f"Origami Confidence: {bin_confidence:.2f}\n"
        f"Difficulty: {predicted_label}\n"
        f"Difficulty Confidence: {diff_confidence:.2f}\n"
        f"Edge Count: {edge_count}"
    )
    axes[0,1].text(0.5, 0.5, metrics_text,
                  fontsize=14, ha="center", va="center")
    axes[0,1].set_title("Prediction Metrics")
    axes[0,1].axis("off")

    if is_origami:
        axes[1,0].imshow(overlay)
        axes[1,0].set_title(f"Grad-CAM Heatmap ({predicted_label})")
        axes[1,0].axis("off")

        axes[1,1].imshow(edges, cmap="gray")
        axes[1,1].set_title(f"Edge Map (Count: {edge_count})")
        axes[1,1].axis("off")
    else:
        axes[1,0].axis("off")
        axes[1,1].axis("off")

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
