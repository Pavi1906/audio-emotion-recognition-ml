import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Dataset Path
data_dir = 'F:\audio_emotion_recognition_model\data'#REPLACE WITH WAV FILE DIRECTORY

# Labels and Parameters
labels = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted']
num_classes = len(labels)
sample_rate = 22050
max_pad_len = 500

# Feature Extraction
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load Dataset
X, y = [], []

for label_idx, label in enumerate(labels):
    folder_path = os.path.join(data_dir, label)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.wav'):
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label_idx)

X = np.array(X)
y = np.array(y)

# One-Hot Encoding
y = to_categorical(y, num_classes=num_classes)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, max_pad_len, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Save the Model
model.save('audio_emotion_recognition_model.h5')

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
