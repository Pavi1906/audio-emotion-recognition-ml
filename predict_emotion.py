import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
model_path = r'C:\Users\aswin\Documents\VS code codes\ai\audio_emotion_recognition_model.h5'
model = load_model(model_path)

# Labels corresponding to emotion classes
labels = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised']

# Function to preprocess and predict emotion for a single .wav file
def predict_emotion(file_path):
    sample_rate = 22050
    max_pad_len = 500
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sample_rate)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Pad or truncate to the required length
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        # Add channel and batch dimensions
        features = mfcc[np.newaxis, ..., np.newaxis]
        # Predict emotion
        prediction = model.predict(features)
        # Get the label with the highest probability
        predicted_label = labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_label, confidence
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# Path to the new .wav file
file_path = r'C:\Users\aswin\Documents\VS code codes\ai\angry.wav'  # Replace with your .wav file path

# Predict emotion
predicted_emotion, confidence = predict_emotion(file_path)

if predicted_emotion:
    print(f"Predicted Emotion: {predicted_emotion} with Confidence: {confidence:.2f}%")
else:
    print("Error in processing the file.")
