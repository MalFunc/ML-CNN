import os
import numpy as np
import re
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# Constants
CACHE_DIR = "cache1"
FEATURES_CACHE = os.path.join(CACHE_DIR, "features_cache.joblib")
SCALER_CACHE = os.path.join(CACHE_DIR, "scaler.joblib")
LABEL_ENCODER_CACHE = os.path.join(CACHE_DIR, "label_encoder.joblib")

# Create cache directory if not exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Dataset Loading =====
def load_ravdess(path="C:/Users/malik/Downloads/GPNcTF/archive12"):
    emotion_map = {'01': 'neutral', '02': 'neutral', '03': 'happy', 
                   '04': 'sad', '05': 'angry', '06': 'fear',
                   '07': 'disgust', '08': 'surprised'}
    data = []
    for dirname, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".wav"):
                emotion = emotion_map[file.split("-")[2]]
                data.append((os.path.join(dirname, file), emotion))
    return data

def load_crema(path="C:/Users/malik/Downloads/GPNcTF/AudioWAV"):
    emotion_map = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
                   'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}
    data = []
    for dirname, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".wav"):
                emotion = emotion_map[file.split("_")[2]]
                data.append((os.path.join(dirname, file), emotion))
    return data

def valname(name):
    after_underscore = name.split('_')[-1]
    letters_only = re.sub(r'[^a-zA-Z]', '', after_underscore)
    return letters_only

def load_savee(path="C:/Users/malik/Downloads/GPNcTF/ALL"):
    emotion_map = {'a': 'angry', 'd': 'disgust', 'f': 'fear',
                   'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprised'}
    data = []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            prefix = valname(file)[:-3] if valname(file)[:-3] in emotion_map else file[0]
            emotion = emotion_map.get(prefix, 'unknown')
            data.append((os.path.join(path, file), emotion))
    return data

def load_tess(path="C:/Users/malik/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1/TESS Toronto emotional speech set data/"):
    data = []
    for dirname, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".wav"):
                emotion = file.split("_")[2].replace(".wav", "").lower()
                if emotion == 'ps': emotion = 'surprised'
                data.append((os.path.join(dirname, file), emotion))
    return data
def visualize_dataset(df):
    """Enhanced visualization of dataset distribution and characteristics"""
    plt.figure(figsize=(15, 10))
    
    # Emotion distribution
    plt.subplot(2, 2, 1)
    emotion_counts = df['emotion'].value_counts()
    palette = sns.color_palette("husl", len(emotion_counts))
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=palette)
    plt.title("Emotion Distribution", fontsize=12)
    plt.xlabel("Emotion", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.xticks(rotation=45)
    
    # Dataset source distribution
    plt.subplot(2, 2, 2)
    df['dataset'] = df['path'].apply(lambda x: 'RAVDESS' if 'archive12' in x 
                                    else 'TESS' if 'TESS' in x 
                                    else 'CREMA' if 'AudioWAV' in x 
                                    else 'SAVEE')
    dataset_counts = df['dataset'].value_counts()
    sns.barplot(x=dataset_counts.index, y=dataset_counts.values, palette="Blues_d")
    plt.title("Dataset Source Distribution", fontsize=12)
    plt.xlabel("Dataset", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    
    # Emotion distribution per dataset
    plt.subplot(2, 2, 3)
    cross_tab = pd.crosstab(df['emotion'], df['dataset'])
    cross_tab.plot(kind='bar', stacked=True, figsize=(12, 6), 
                   colormap='viridis', ax=plt.gca())
    plt.title("Emotion Distribution per Dataset", fontsize=12)
    plt.xlabel("Emotion", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Duration distribution
    plt.subplot(2, 2, 4)
    df['duration'] = df['path'].apply(lambda x: librosa.get_duration(filename=x))
    sns.boxplot(x='emotion', y='duration', data=df, palette="Set3")
    plt.title("Duration Distribution by Emotion", fontsize=12)
    plt.xlabel("Emotion", fontsize=10)
    plt.ylabel("Duration (seconds)", fontsize=10)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
def visualize_audio_features(file_path, title=""):
    """Enhanced visualization of audio features with more details"""
    data, sr = librosa.load(file_path, duration=3, offset=0.5)
    
    plt.figure(figsize=(18, 12))
    
    # Waveform
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(data, sr=sr, color='b')
    plt.title(f"Waveform - {title}", fontsize=12)
    plt.xlabel("Time (s)", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    
    # Spectrogram
    plt.subplot(3, 2, 2)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram - {title}", fontsize=12)
    
    # MFCCs
    plt.subplot(3, 2, 3)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.title(f"MFCC - {title}", fontsize=12)
    plt.ylabel("MFCC Coefficients", fontsize=10)
    
    # Chromagram
    plt.subplot(3, 2, 4)
    chroma = librosa.feature.chroma_stft(y=data, sr=sr)
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar()
    plt.title(f"Chromagram - {title}", fontsize=12)
    
    # Spectral Contrast
    plt.subplot(3, 2, 5)
    contrast = librosa.feature.spectral_contrast(y=data, sr=sr)
    librosa.display.specshow(contrast, x_axis='time', cmap='magma')
    plt.colorbar()
    plt.title(f"Spectral Contrast - {title}", fontsize=12)
    plt.ylabel("Frequency Bands", fontsize=10)
    
    # Tonnetz
    plt.subplot(3, 2, 6)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr)
    librosa.display.specshow(tonnetz, x_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.title(f"Tonnetz - {title}", fontsize=12)
    plt.ylabel("Tonal Features", fontsize=10)
    
    plt.tight_layout()
    plt.show()
def plot_training_history(history):
    """Enhanced training history visualization with more metrics"""
    plt.figure(figsize=(18, 6))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='coral', linewidth=2)
    plt.title("Training and Validation Loss", fontsize=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='royalblue', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='coral', linewidth=2)
    plt.title("Training and Validation Accuracy", fontsize=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Learning rate plot (if available)
    if 'lr' in history:
        plt.subplot(1, 3, 3)
        plt.plot(history['lr'], label='Learning Rate', color='green', linewidth=2)
        plt.title("Learning Rate Schedule", fontsize=12)
        plt.xlabel("Epochs", fontsize=10)
        plt.ylabel("Learning Rate", fontsize=10)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    """
    Enhanced confusion matrix visualization with normalization option
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = "Normalized Confusion Matrix"
    else:
        fmt = 'd'
        title = "Confusion Matrix"
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='lightgray')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def visualize_feature_distribution(X, y, le, n_features=5):
    """
    Visualize distribution of top features across different emotions
    """
    # Convert back to numpy for easier handling
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    
    # Get feature names (simplified for this example)
    feature_names = [
        'ZCR', 'Chroma1', 'Chroma2', 'Chroma3', 'Chroma4', 'Chroma5', 'Chroma6', 
        'Chroma7', 'Chroma8', 'Chroma9', 'Chroma10', 'Chroma11', 'Chroma12',
        *[f'MFCC_{i}' for i in range(1, 14)],
        *[f'MFCC_Delta_{i}' for i in range(1, 14)],
        *[f'MFCC_Delta2_{i}' for i in range(1, 14)],
        'RMS', *[f'Mel_{i}' for i in range(1, 129)]
    ]
    
    # Select top n features with highest variance
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-n_features:][::-1]
    
    # Create subplots
    plt.figure(figsize=(15, 3*n_features))
    
    for i, idx in enumerate(top_indices, 1):
        plt.subplot(n_features, 1, i)
        feature_name = feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}'
        
        # Create a DataFrame for easier plotting
        df_plot = pd.DataFrame({
            'feature': X[:, idx],
            'emotion': le.inverse_transform(y)
        })
        
        sns.boxplot(x='emotion', y='feature', data=df_plot, palette='Set3')
        plt.title(f'Distribution of {feature_name} by Emotion', fontsize=12)
        plt.xlabel('Emotion', fontsize=10)
        plt.ylabel(feature_name, fontsize=10)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()




# === Feature Extraction ===
def extract_features(data, sample_rate):
    """Extract audio features from raw data"""
    result = np.array([])
    
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC features
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

def augment_and_extract(path):
    """Apply data augmentation and extract features"""
    try:
        data, sample_rate = librosa.load(path, duration=3, offset=0.6)
        features = []

        # Original
        features.append(extract_features(data, sample_rate))

        # Noise added
        noise_data = data + 0.005 * np.random.normal(0, 1, len(data))
        features.append(extract_features(noise_data, sample_rate))

        # Pitch shifted
        pitch_data = librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=0.7)
        features.append(extract_features(pitch_data, sample_rate))

        # Time stretch
        stretch_data = librosa.effects.time_stretch(data, rate=1.1)
        features.append(extract_features(stretch_data, sample_rate))

        return features
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return []

# === Data Processing with Caching ===
def process_data(df, use_cache=True):
    """Process raw data into features with caching support"""
    if use_cache and os.path.exists(FEATURES_CACHE):
        print("Loading features from cache...")
        cache_data = joblib.load(FEATURES_CACHE)
        X = cache_data['X']
        y = cache_data['y']
        scaler = joblib.load(SCALER_CACHE)
        le = joblib.load(LABEL_ENCODER_CACHE)
    else:
        print("Processing data from scratch...")
        X, y = [], []
        for path, emotion in df.itertuples(index=False):
            feats = augment_and_extract(path)
            for f in feats:
                X.append(f)
                y.append(emotion)

        if not X:
            raise ValueError("Feature extraction failed for all files. Check dataset integrity.")

        X = np.array(X)
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Save to cache
        joblib.dump({'X': X, 'y': y}, FEATURES_CACHE)
        joblib.dump(scaler, SCALER_CACHE)
        joblib.dump(le, LABEL_ENCODER_CACHE)

    return X, y, scaler, le

# === PyTorch Dataset ===
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Model Definition ===
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

# === Training ===
def train_model(model, train_loader, val_loader, num_classes, epochs=200):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    scaler = GradScaler()  # For mixed precision training
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))
    return model, history

def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = running_loss / len(data_loader.dataset) if criterion else None
    accuracy = correct / total
    
    return loss, accuracy

# === Evaluation ===
def evaluate_model_performance(model, test_loader, le):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, xticklabels=le.classes_, yticklabels=le.classes_, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.show()

# === Prediction ===
def predict_emotion(model, file_path, scaler, le):
    """Predict emotion from audio file"""
    # Load and preprocess audio
    data, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    
    # Extract features
    feature = extract_features(data, sample_rate)
    # Scale and reshape
    feature = scaler.transform([feature])
    feature_tensor = torch.tensor(feature, dtype=torch.float32).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(feature_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_emotion = le.inverse_transform([predicted.item()])[0]
    
    return predicted_emotion

# === Main Execution ===
if __name__ == "__main__":
    # Load and combine datasets
    print("Loading datasets...")
    all_data = load_tess()
    df = pd.DataFrame(all_data, columns=["path", "emotion"])
    
    # Visualize dataset
    visualize_dataset(df)
    
    # Visualize sample audio features for each emotion
    sample_files = df.groupby('emotion').first()['path'].tolist()
    for file in sample_files:
        emotion = df[df['path'] == file]['emotion'].iloc[0]
        visualize_audio_features(file, emotion)
    
    # Process data (with caching)
    X, y, scaler, le = process_data(df, use_cache=True)
    
    # Visualize feature distribution
    visualize_feature_distribution(X, y, le, n_features=5)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    num_classes = len(le.classes_)
    print(X_train.shape[1])
    model = EmotionCNN(input_size=X_train.shape[1], num_classes=num_classes)
    print(X_train.shape[1])
    # Train model
    print("Training model...")
    model, history = train_model(model, train_loader, test_loader, num_classes)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    torch.save(model.state_dict(), "emotion_modelv2.pth")
    print("Model saved as emotion_model.pth")
    
    # Evaluate
    evaluate_model_performance(model, test_loader, le)
    
    # Example prediction
    test_file = "C:/Users/malik/Downloads/GPNcTF/archive12/Actor_01/03-01-01-01-01-01-01.wav"
    emotion = predict_emotion(model, test_file, scaler, le)
    print(f"Predicted Emotion: {emotion}")
