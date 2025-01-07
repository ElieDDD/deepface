
#use_container_width=True
import os
from PIL import Image, ImageFilter
import streamlit as st
from deepface import DeepFace
import numpy as np
import pandas as pd  # Import pandas for creating the DataFrame

# Function to detect emotions from an image
def detect_emotions(image):
    """
    Detect emotions from the input image using DeepFace.
    Handles cases where DeepFace returns a list or no face is detected.
    """
    temp_image_path = "sun.jpg"
    image.save(temp_image_path)  # Save the image temporarily
    
    try:
        # Analyze emotions using DeepFace
        analysis = DeepFace.analyze(temp_image_path, actions=["emotion"], enforce_detection=False)
        
        # Check if output is a list or dictionary
        if isinstance(analysis, list):  # DeepFace may return a list
            return analysis[0]  # Access the first result in the list
        return analysis  # Return the dictionary directly if it's not a list
    except Exception as e:
        # Handle cases where no face is detected or another error occurs
        return {"dominant_emotion": "No face detected", "emotion": {}}

# Function to blur an image
def blur_image(image, radius=35):
    """
    Apply a blur effect to the given image using PIL.
    :param image: PIL Image object
    :param radius: Intensity of the blur (default is 5)
    :return: Blurred PIL Image object
    """
    return image.filter(ImageFilter.GaussianBlur(radius))

# Function to tally emotions
def tally_emotions(emotions):
    """
    Tally the occurrences of each detected emotion.
    :param emotions: List of emotion dictionaries
    :return: Dictionary of emotion tallies
    """
    emotion_counts = {}
    for emotion in emotions:
        dominant_emotion = emotion.get("dominant_emotion", "No face detected")
        emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
    return emotion_counts
numer = 10
# Display images in a grid with labels
def display_images_with_labels(image_paths, emotions):
    cols = st.columns(numer)  # Set up 3 columns per row
    for idx, image_path in enumerate(image_paths):
        col = cols[idx % numer]  # Place each image in one of the three columns
        with col:
            # Open and blur image
            image = Image.open(image_path)
            blurred_image = blur_image(image)  # Apply blur effect

            # Display blurred image
            st.image(blurred_image, use_container_width=True)

            # Display emotion label
            emotion = emotions[idx].get("dominant_emotion", "Error")
            st.caption(f"Emotion: {emotion}")

# Main Streamlit app
def main():
    st.title("Construct of Emotion")
    st.text("Decoding through computer vision is a set of decisions about how to interpret visual messages that is shaped by cultural and social values, in addition to producing them")
    st.text( "Arnold & Tilton, Distant Viewing 2023")
    st.text("Upload a folder of images, and detect emotional constructs while applying a blur effect to each image")

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

        # Display results in a grid with blur effect
        st.text("Images blurred for privacy:")
        display_images_with_labels(image_paths, emotions)

        # Tally and display emotion statistics as a bar chart
        emotion_tallies = tally_emotions(emotions)
        st.text("Statistics:")

        # Convert the emotion tally dictionary to a pandas DataFrame
        emotion_df = pd.DataFrame(list(emotion_tallies.items()), columns=["Emotion", "Count"])

        # Display the bar chart
        st.bar_chart(emotion_df.set_index("Emotion"))

if __name__ == "__main__":
    main()
