
import os
import tempfile
from PIL import Image, ImageFilter
import streamlit as st
from deepface import DeepFace
import pandas as pd
import altair as alt
from concurrent.futures import ThreadPoolExecutor

# Remove micropip import since it's unnecessary in standard environments

def detect_emotions(image_path):
    """
    Detect emotions using DeepFace.
    """
    try:
        analysis = DeepFace.analyze(image_path, actions=["emotion"], enforce_detection=False)
        return analysis[0] if isinstance(analysis, list) else analysis
    except Exception:
        return {"dominant_emotion": "Error", "emotion": {}}

def preprocess_image(image):
    """
    Resize the image to a smaller resolution for faster processing.
    """
    return image.resize((224, 224))

def save_temp_image(image):
    """
    Save the image temporarily and return its path.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        return temp_file.name

def blur_image(image, radius=3):
    return image.filter(ImageFilter.GaussianBlur(radius))

def tally_emotions(emotions):
    """
    Count the occurrences of each detected emotion.
    """
    emotion_counts = {}
    for emotion in emotions:
        dominant_emotion = emotion.get("dominant_emotion", "Error")
        emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
    return emotion_counts

def display_images_with_labels(image_paths, emotions):
    """
    Display images in a grid with labels.
    """
    num_cols = min(3, len(image_paths))
    cols = st.columns(num_cols)
    
    for idx, image_path in enumerate(image_paths):
        col = cols[idx % num_cols]
        with col:
            image = Image.open(image_path)
            st.image(blur_image(image), use_container_width=True)
            st.caption(f": {emotions[idx].get('dominant_emotion', 'Error')}")

def main():
    st.title("Construct of Emotion")
    st.text("Upload images to detect emotional constructs.")
    uploaded_files = st.file_uploader("Upload Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.text("Analyzing uploaded images...")
        image_paths = []
        emotions = []
        
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            image = preprocess_image(image)
            temp_path = save_temp_image(image)
            image_paths.append(temp_path)
        
        with ThreadPoolExecutor() as executor:
            emotions = list(executor.map(detect_emotions, image_paths))
        
        st.text("Here are the images with detected emotions:")
        display_images_with_labels(image_paths, emotions)
        
        emotion_tallies = tally_emotions(emotions)
        emotion_df = pd.DataFrame(list(emotion_tallies.items()), columns=["Emotion", "Count"])
        
        chart = alt.Chart(emotion_df).mark_bar().encode(
            x=alt.X('Emotion:N', sort=None),
            y='Count:Q',
            color='Emotion:N'
        ).properties(title="Statistics")
        
        st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
