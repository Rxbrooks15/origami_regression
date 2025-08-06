import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- Page Setup ---
st.title("📸 Origami Binary Classifier")
st.write("""
Upload an image, and this app will predict whether it is **Origami** or **Not Origami**  
using a Convolutional Neural Network (CNN) trained on origami datasets.
""")

# --- Load Binary Model ---
@st.cache_resource
def load_binary_model():
    return load_model("origami_yesno_final.h5")  # or origami_yesno_final.keras

model = load_binary_model()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB").resize((224,224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    label = "Origami" if confidence > 0.5 else "Not Origami"

    # Show results
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.4f}")

    # Explain Binary Classification vs. SVC
    st.markdown("""
    ---
    ### 🔍 About the Model

    - **Binary Classification**:  
      The model decides between two classes: **Origami** or **Not Origami**.  
      A threshold of 0.5 determines the class.

    - **CNN vs. SVC**:  
      - This app uses a **Convolutional Neural Network (CNN)**, which automatically learns visual features such as folds, edges, and textures.  
      - A **Support Vector Classifier (SVC)**, often used in earlier origami classification research, works well with numerical features but typically requires handcrafted features from images (like edges or shape descriptors).  
      - CNNs generally outperform SVCs for complex image recognition tasks because they learn features directly from raw pixels.

    - **Confidence Score**:  
      Indicates how certain the model is about its prediction.  
    """)
