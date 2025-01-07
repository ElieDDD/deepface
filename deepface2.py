import os
from PIL import Image
import streamlit as st
from deepface import DeepFace
import numpy as np

# Function to detect emotions from an image
def detect_emotions(image):
    """
    Detect emotions from the input image using DeepFace.
    """
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)  # Save the image temporarily
    analysis = DeepFace.analyze(temp_image_path, actions=["emotion"], enforce_detection=False)
    return analysis

# Display images in a grid with labels
def display_images_with_labels(image_paths, emotions):
    cols = st.columns(3)  # Set up 3 columns per row
    for idx, image_path in enumerate(image_paths):
        col = cols[idx % 3]  # Place each image in one of the three columns
        with col:
            # Display image
            image = Image.open(image_path)
            st.image(image, use_column_width=True)

            # Display emotion label
            emotion = emotions[idx]["dominant_emotion"]
            st.caption(f"Emotion: {emotion}")

# Main Streamlit app
def main():
    st.title("Emotion Detection for Folder of Images")
    st.text("Upload a folder of images, and we'll detect emotions for each image!")

    # Upload folder
    uploaded_files = st.file_uploader("Upload Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.text("Analyzing emotions for uploaded images...")
        
        image_paths = []  # Store uploaded image paths
        emotions = []  # Store emotion results
        
        for uploaded_file in uploaded_files:
            # Convert the uploaded file into a PIL Image
            image = Image.open(uploaded_file)

            # Check and convert RGBA to RGB
            if image.mode == "RGBA":
                image = image.convert("RGB")

            # Save image temporarily for DeepFace processing
            temp_image_path = f"temp_{uploaded_file.name}"
            image.save(temp_image_path)
            image_paths.append(temp_image_path)

            # Detect emotions
            emotion_result = detect_emotions(image)
            emotions.append(emotion_result)

        # Display results in a grid
        st.text("Here are the analyzed images with detected emotions:")
        display_images_with_labels(image_paths, emotions)

if __name__ == "__main__":
    main()
