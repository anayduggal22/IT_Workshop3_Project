import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.title("CNN Digit Recognizer")

st.write("Name: Anay Duggal \nRoll No.: 2025BCS0093")

st.write("Draw a digit from 0 to 9 below.")

# Loading model
model = keras.models.load_model("model.keras")


# Preprocessing image
def prep(i):
    
    # Converting to grayscale
    i = ImageOps.grayscale(i)

    # Resizing image to 28x28 and converting pixels to 0-1
    i = i.resize((28, 28))
    i = np.array(i).astype("float32")/255.0

    # Adding batch and channel
    return i.reshape(1, 28, 28, 1)


# Predicting digit
def pred(i):
    
    # Proba for digits 0-9
    proba = model.predict(i,verbose=0)[0]

    # d is digit with max proba
    d = np.argmax(proba)
    c = proba[d]*100

    st.success(f"Predicted Digit: {d}")
    st.write(f"Confidence: {c:.2f}%")

    # Proba bar chart of all digits
    st.bar_chart(proba)


# Drawing canvas
canvas = st_canvas(
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas"
)

# Predicting button
if st.button("Predict"):

    # If we drawed something
    if canvas.image_data is not None:

        # Converting output to image
        i = Image.fromarray(
            canvas.image_data.astype("uint8")
        )

        # Preprocessing and predicting
        pred(prep(i))

    # If we did not draw something
    else:
        st.warning("Please draw a Digit.")
