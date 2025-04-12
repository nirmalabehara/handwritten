import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Streamlit Title
st.title("📝 Handwritten Character Recognition - EMNIST Letters")
st.write("Upload an image of a handwritten letter (A-Z)")

# Load trained EMNIST model
@st.cache_resource
def load_emnist_model():
    return load_model("emnist_model.h5")

model = load_emnist_model()

# Mapping from class index to uppercase letter (EMNIST starts at 'A')
def get_letter(index):
    return chr(index + ord('A'))

# Preprocessing Function
def preprocess_image(image):
    image = np.array(image.convert('L'))  # Convert to grayscale
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = image[y:y+h, x:x+w]
    else:
        cropped = image

    resized = cv2.resize(cropped, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]
    predicted_label = np.argmax(predictions)
    confidence = predictions[predicted_label] * 100
    predicted_letter = get_letter(predicted_label)

    # Display result
    st.subheader(f"✅ Predicted Letter: **{predicted_letter}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

