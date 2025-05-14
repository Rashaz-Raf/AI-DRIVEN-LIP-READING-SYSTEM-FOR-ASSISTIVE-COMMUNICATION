import streamlit as st
import os
import imageio
import tempfile
import numpy as np
import tensorflow as tf
import subprocess  # For running ffmpeg safely
from gtts import gTTS
from utils import load_data, num_to_char
from modelutil import load_model

# Set the absolute path for the dataset
DATA_DIR = r"C:\Users\jeeva\OneDrive\Desktop\LR\data\s1"

# Check if the directory exists
if not os.path.exists(DATA_DIR):
    st.error(f"Directory not found: {DATA_DIR}. Please check the path and try again.")
    st.stop()

# Fetch list of available video files
options = os.listdir(DATA_DIR)

# Set the layout to the Streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('AI Driven Lip Reading System For Assistive Communication')
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if selected_video:
    file_path = os.path.join(DATA_DIR, selected_video)

    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')

        # Convert the video to mp4 format using ffmpeg
        output_video = "test_video.mp4"
        ffmpeg_command = f'ffmpeg -i "{file_path}" -vcodec libx264 {output_video} -y'
        process = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)

        # Check if ffmpeg ran successfully
        if process.returncode != 0:
            st.error("Error converting video. Check ffmpeg installation.")
            st.text(process.stderr)
            st.stop()

        # Ensure file exists before opening
        if os.path.exists(output_video):
            with open(output_video, 'rb') as video:
                video_bytes = video.read()
            st.video(video_bytes)
        else:
            st.error("Converted video file not found. Please check ffmpeg.")

    # Machine learning prediction
    with col2:
        st.info('The model extracts patterns from these frames to understand speech.')

        # Load and preprocess data
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        video_frames = video.numpy()  # Convert to NumPy array

        # Debug: Check shape and dtype
        st.text(f"Shape of video: {video_frames.shape}, dtype: {video_frames.dtype}")

        # Convert data type to uint8
        video_frames = (video_frames * 255).astype('uint8')

        # Fix grayscale format
        if video_frames.shape[-1] == 1:
            video_frames = video_frames.squeeze(-1)

        # Convert to list and save GIF
        frame_list = [frame for frame in video_frames]
        imageio.mimsave('animation.gif', frame_list, fps=10)
        st.image('animation.gif', width=400)

        # Load model and predict
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

        # Display raw token output
        st.text("Predicted Tokens:")
        st.text(list(decoder.flatten()))  # Convert decoder to list before displaying

        # Convert prediction to text
        st.subheader("Predicted Text:")
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

    # Speech Generation Function
    def text_to_speech(text):
        try:
            if not text.strip():
                st.error("Text for speech generation is empty.")
                return None

            # Create a temporary file for speech output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                tts = gTTS(text=text, lang="en")
                temp_audio_path = temp_audio.name
                tts.save(temp_audio_path)

            return temp_audio_path  # Return path of saved audio file

        except Exception as e:
            st.error(f"❌ Error generating speech: {e}")
            return None

    # Button to Generate and Play Speech
    if st.button("Speak Prediction"):
        audio_path = text_to_speech(converted_prediction)  # Generate speech

        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/mp3")
            st.success("Playing the predicted speech!")
        else:
            st.error("Speech generation failed.")
