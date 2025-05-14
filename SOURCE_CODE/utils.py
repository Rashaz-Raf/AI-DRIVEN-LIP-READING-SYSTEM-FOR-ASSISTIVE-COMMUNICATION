import tensorflow as tf
from typing import List
import cv2
import os
from gtts import gTTS
import pygame
import time

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str) -> List[float]: 
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path: str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = os.path.splitext(os.path.basename(path))[0]  # Extract filename without extension

    video_path = os.path.join("C:\\Users\\jeeva\\OneDrive\\Desktop\\LR\\data\\s1", f"{file_name}.mpg")
    alignment_path = os.path.join("C:\\Users\\jeeva\\OneDrive\\Desktop\\LR\\data\\alignments\\s1", f"{file_name}.align")

    print(f"Loading video: {video_path}")  # Debugging print statement
    print(f"Loading alignment: {alignment_path}")  # Debugging print statement

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(alignment_path):
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments

def text_to_speech(text):
    try:
        if not text:
            raise ValueError("Text for speech generation is empty.")

        # Create a temporary file for speech output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts = gTTS(text=text, lang="en")
            temp_audio_path = temp_audio.name
            tts.save(temp_audio_path)
        
        return temp_audio_path  # Return path of saved audio file

    except Exception as e:
        print(f"❌ Error generating speech: {e}")
        return None