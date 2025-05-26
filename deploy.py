from flask import Flask, render_template, Response, url_for
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import time
from collections import deque
import threading
import json
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# CONFIGURATION
MODEL_PATH = "emotion_model.pth"  # File model PyTorch (.pth)
SCALER_PATH = "cache/scaler.joblib"     # File scaler
LABEL_ENCODER_PATH = "cache/label_encoder.joblib"  # File label encoder
DEVICE_ID = 10  # Ganti dengan ID microphone Anda
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5  # Durasi chunk dalam detik
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.01
HISTORY_SIZE = 3

# Muat scaler dan label encoder
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

emotion_images = {
    "angry": "/static/angry.jpg",
    "disgust": "/static/disgust.jpg",
    "fear": "/static/fear.jpg",
    "happy": "/static/happy.jpg",
    "neutral": "/static/normal.jpg",
    "surprised": "/static/ps.jpg",
    "sad": "/static/sad.jpg"
}

# Definisikan model CNN yang sama seperti saat training
class EmotionCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding="same"),
            nn.GELU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate the size after convolutions
        test_input = torch.randn(1, 1, input_size)
        conv_out = self.conv3(self.conv2(self.conv1(test_input)))
        flattened_size = conv_out.view(-1).shape[0]
        
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Muat model
# Ganti input_size sesuai dengan model asli
input_size = 202  # Gunakan nilai yang sama saat training
model = EmotionCNN(input_size=input_size, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')

# Global state
current_emotion = "neutral"
last_update_time = time.time()
emotion_history = deque(maxlen=HISTORY_SIZE)
current_volume = 0.0

def extract_features(data, sample_rate):
    """Ekstrak fitur audio seperti pada training"""
    result = np.array([])
    
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_delta = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
    result = np.hstack((result, mfcc_mean, mfcc_delta, mfcc_delta2))

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

def audio_callback(indata, frames, time_info, status):
    global current_emotion, last_update_time, current_volume

    rms = np.sqrt(np.mean(indata**2))
    current_volume = float(rms)

    if rms > SILENCE_THRESHOLD:
        try:
            # Ekstrak fitur
            features = extract_features(indata.squeeze(), SAMPLE_RATE)
            
            # Scale fitur
            features_scaled = scaler.transform([features])
            
            # Konversi ke tensor
            inputs = torch.tensor(features_scaled, dtype=torch.float32)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            # Prediksi
            with torch.no_grad():
                outputs = model(inputs)
                pred = torch.argmax(outputs).item()
            
            # Decode label
            emotion = label_encoder.inverse_transform([pred])[0]
            emotion_history.append(emotion)
            
            # Update emotion berdasarkan voting
            new_emotion = max(set(emotion_history), key=emotion_history.count)
            
            if new_emotion != current_emotion:
                current_emotion = new_emotion
                last_update_time = time.time()
                
        except Exception as e:
            print(f"Error in processing: {e}")
    
    elif time.time() - last_update_time > 5.0 and current_emotion != "neutral":
        current_emotion = "neutral"

def start_audio_stream():
    with sd.InputStream(
        device=DEVICE_ID,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        while True:
            time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            data = {
                "emotion": current_emotion,
                "image_url": emotion_images.get(current_emotion, "/static/normal.jpg"),
                "volume": current_volume
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.05)
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    audio_thread = threading.Thread(target=start_audio_stream)
    audio_thread.daemon = True
    audio_thread.start()
    app.run(debug=True)
