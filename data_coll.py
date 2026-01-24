import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Directory to save data
DATA_DIR = 'asl_data'
os.makedirs(DATA_DIR, exist_ok=True)

# Full A-Z labels
labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]

# Number of samples per label
samples_per_label = 200

for label in labels:
    print(f"\nCollecting data for '{label}' sign. Press 'q' to quit early.")
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0

    while count < samples_per_label:
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            # Save data
            file_path = os.path.join(label_dir, f'{label}_{count}.npy')
            np.save(file_path, np.array(landmarks))
            count += 1

            # Draw hand
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(image, f'Label: {label} | Sample: {count}/{samples_per_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("ASL Data Collection", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

hands.close()
print("\n✅ Data collection completed.")
