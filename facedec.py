import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

def detect_emotions(image):
    """
    Detect emotions from the input image using DeepFace.
    """
    # Save the image temporarily
    temp_image_path = "sun.jpg"
    image.save(temp_image_path)

    # Analyze emotions
    analysis = DeepFace.analyze(temp_image_path, actions=['emotion'], enforce_detection=False)
    return analysis

def main():
    st.title("Emotion Detection App")
    st.text("Upload an image, and we'll analyze the emotions in the faces!")

    # File uploader
    uploaded_file = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect emotions
        st.text("Analyzing emotions...")
        analysis = detect_emotions(image)

        # Display results
        for face in analysis:
            st.write(f"Emotion: {face['dominant_emotion']}")
            st.write(f"Details: {face['emotion']}")

if __name__ == "__main__":
    main()
