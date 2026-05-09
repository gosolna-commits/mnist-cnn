#app.py

# streamlit run Göran_ML_CNN_sparad.py

#mnist-cnn-app-bappbydwvbjjfcrt63rted4.streamlit.app

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# -----------------------------
# Läs accuracy
# -----------------------------
with open("accuracy.txt", "r") as f:
    accuracy = f.read()
    

# -----------------------------
# Ladda modell
# -----------------------------
model = load_model("mnist_cnn.keras")

st.title("MNIST CNN Digit Recognizer")

st.write(f"Model Accuracy: {float(accuracy)*100:.2f}%")

st.write("Rita en siffra mellan 0 och 9")

# -----------------------------
# Canvas
# -----------------------------
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# Om något ritats
# -----------------------------
if canvas_result.image_data is not None:

    img = canvas_result.image_data

    # -----------------------------
    # Ta grayscale-kanalen
    # -----------------------------
    img_gray = img[:, :, 0]

    # -----------------------------
    # Kontrollera om något ritats
    # -----------------------------
    pixels_drawn = np.count_nonzero(img_gray > 50)

    # Kräver minst 50 ljusa pixlar
    if pixels_drawn > 50:

        # -----------------------------
        # PIL image
        # -----------------------------
        img_pil = Image.fromarray(
            img_gray.astype(np.uint8)
        )

        # -----------------------------
        # Resize
        # -----------------------------
        img_pil = img_pil.resize((28, 28))

        # -----------------------------
        # Convert to numpy
        # -----------------------------
        img_array = np.array(img_pil)

        # -----------------------------
        # Normalize
        # -----------------------------
        img_array = img_array.astype("float32") / 255.0

        # -----------------------------
        # CNN input shape
        # -----------------------------
        img_input = img_array.reshape(1, 28, 28, 1)

        # -----------------------------
        # Predict
        # -----------------------------
        prediction = model.predict(img_input)

        predicted_digit = np.argmax(prediction)

        probabilities = prediction[0]

        # -----------------------------
        # Show image
        # -----------------------------
        st.image(img_array, width=150)

        # -----------------------------
        # Show prediction
        # -----------------------------
        st.header(f"Gissning: {predicted_digit}")

        # -----------------------------
        # Show probabilities
        # -----------------------------
        st.subheader("Sannolikheter")

        for i, prob in enumerate(probabilities):

            st.write(f"{i}: {prob:.2%}")

        st.bar_chart(probabilities)