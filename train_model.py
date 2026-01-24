# train_asl_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# === STEP 1: Load Data ===
DATA_DIR = 'asl_data'
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
label_map = {label: idx for idx, label in enumerate(labels)}

X, y = [], []

for label in labels:
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            data = np.load(os.path.join(folder_path, file))
            if data.shape[0] == 63:
                X.append(data)
                y.append(label_map[label])

X = np.array(X)
y = to_categorical(y, num_classes=26)

print(f"✅ Loaded {len(X)} samples from {len(labels)} labels.")

# === STEP 2: Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 3: Build Model ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === STEP 4: Train Model ===
checkpoint = ModelCheckpoint("cnn8grps_rad1_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=30,
                    batch_size=32,
                    callbacks=[checkpoint])

# === STEP 5: Evaluate ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.2%}")
