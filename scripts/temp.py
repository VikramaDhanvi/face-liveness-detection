# Import libraries
import os
import cv2
import time
import math
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import xgboost as xgb

# ------------------------------------------------
# Step 1: Load Models
# ------------------------------------------------
xgb_model = joblib.load('../models/xgboost_liveness_model.pkl')
scaler_xgb = joblib.load('../models/scaler.pkl')

landmark_model = joblib.load('../models/landmark_model.pkl')
scaler_landmark = joblib.load('../models/scaler.pkl')
labels_landmark = ['mask', 'monitor', 'outline', 'print', 'print_cut', 'real', 'silicone']

# ------------------------------------------------
# Step 2: Initialize Mediapipe and DNN Face Detector
# ------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# ------------------------------------------------
# Step 3: Helper Functions
# ------------------------------------------------
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_deltas(prev, curr):
    return [c - p for p, c in zip(prev, curr)]

def extract_distances(landmarks):
    eye_left = (landmarks[33*2], landmarks[33*2+1])
    eye_right = (landmarks[263*2], landmarks[263*2+1])
    nose = (landmarks[1*2], landmarks[1*2+1])
    mouth = (landmarks[13*2], landmarks[13*2+1])
    return [
        dist(eye_left, eye_right),
        dist(nose, mouth),
        dist(eye_left, nose),
        dist(eye_right, mouth)
    ]

def calculate_head_tilt(landmarks):
    left_eye = np.array([landmarks[33*2], landmarks[33*2+1]])
    right_eye = np.array([landmarks[263*2], landmarks[263*2+1]])
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return np.degrees(math.atan2(dy, dx))

def eye_aspect_ratio(landmarks):
    p1 = np.array([landmarks[159*2], landmarks[159*2+1]])
    p2 = np.array([landmarks[145*2], landmarks[145*2+1]])
    p3 = np.array([landmarks[33*2], landmarks[33*2+1]])
    p4 = np.array([landmarks[133*2], landmarks[133*2+1]])
    return dist(p1, p2) / dist(p3, p4)

def mouth_aspect_ratio(landmarks):
    top = np.array([landmarks[13*2], landmarks[13*2+1]])
    bottom = np.array([landmarks[14*2], landmarks[14*2+1]])
    left = np.array([landmarks[78*2], landmarks[78*2+1]])
    right = np.array([landmarks[308*2], landmarks[308*2+1]])
    return dist(top, bottom) / dist(left, right)

def detect_face_dnn(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()
    h, w = frame.shape[:2]
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            faces.append(box)
    return len(faces) > 0

def detect_sudden_tilt(tilts, threshold=15.0):
    for i in range(1, len(tilts)):
        if abs(tilts[i] - tilts[i-1]) > threshold:
            return True
    return False

# ------------------------------------------------
# Step 4: Capture from Webcam
# ------------------------------------------------
print("[INFO] Opening webcam...")
cap = cv2.VideoCapture(0)
time.sleep(1)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

max_frames = 300
frame_count = 0

tilts, ears, mars = [], [], []
landmark_history = []
delta_list = []

print("🟡 Capturing 300 frames...")

prev_landmarks = None

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        continue

    if detect_face_dnn(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                coords = []
                for lm in face_landmarks.landmark:
                    coords.extend([lm.x, lm.y])

                if len(coords) == 936:
                    tilt = calculate_head_tilt(coords)
                    ear = eye_aspect_ratio(coords)
                    mar = mouth_aspect_ratio(coords)

                    tilts.append(tilt)
                    ears.append(ear)
                    mars.append(mar)
                    landmark_history.append(coords)

                    if prev_landmarks:
                        delta = compute_deltas(prev_landmarks, coords)
                        delta_list.append(delta)

                    prev_landmarks = coords
                    frame_count += 1

                    cv2.putText(frame, f"Frame {frame_count}/300", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("🧠 Capturing Frames", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
time.sleep(1)

print("✅ 300 frames captured!")

# ------------------------------------------------
# Step 5: Feature Calculation
# ------------------------------------------------
tilt_change = max(tilts) - min(tilts) if tilts else 0
ear_change = max(ears) - min(ears) if ears else 0
mar_change = max(mars) - min(mars) if mars else 0
avg_movement = np.mean(np.abs(delta_list)) if delta_list else 0
sudden_tilt = detect_sudden_tilt(tilts)

# ------------------------------------------------
# Step 6: Model Predictions
# ------------------------------------------------
xgb_features = pd.DataFrame([{
    'tilt_change': tilt_change,
    'ear_change': ear_change,
    'mar_change': mar_change,
    'avg_movement': avg_movement
}])

print("\n🔍 XGBoost Features:")
for col, val in xgb_features.iloc[0].items():
    print(f"  - {col}: {val:.5f}")

xgb_prediction = xgb_model.predict(xgb_features)[0]
xgb_label = {0: 'FAKE', 1: 'REAL'}[xgb_prediction]

xgb_score_sum = xgb_features.iloc[0].sum()
print(f"🧮 XGBoost Feature Sum: {xgb_score_sum:.5f}")

# Landmark model prediction
if landmark_history:
    deltas = compute_deltas(landmark_history[0], landmark_history[-1])
    distances = extract_distances(landmark_history[-1])
    features = deltas + distances + [ears[-1], mars[-1], tilts[-1]]

    print("\n🔬 Landmark Model Features:")
    for i, val in enumerate(features):
        print(f"  - Feature {i+1}: {val:.5f}")

    landmark_score_sum = sum(features)
    print(f"🧮 Landmark Feature Sum: {landmark_score_sum:.5f}")

    features_df = pd.DataFrame([features])
    scaled = scaler_landmark.transform(features_df)
    landmark_prediction = landmark_model.predict(scaled)[0]
    landmark_label = labels_landmark[landmark_prediction]
else:
    landmark_label = "unknown"
    landmark_score_sum = 0

# ------------------------------------------------
# Step 7: Decision Logic
# ------------------------------------------------
final_decision = "REAL" if (xgb_label == "REAL" and (landmark_label == "real" or landmark_label == "monitor")) else "FAKE"
total_score_sum = xgb_score_sum + landmark_score_sum

# ------------------------------------------------
# Step 8: Show Final Result
# ------------------------------------------------
print("\n📊 Results Summary:")
print(f"📈 Tilt Change: {tilt_change:.2f}° (Recommended >6°)")
print(f"👁️ EAR Change: {ear_change:.4f} (Recommended >0.30)")
print(f"👄 MAR Change: {mar_change:.4f} (Recommended >0.25)")
print(f"🧠 Avg Landmark Δ: {avg_movement:.5f}")
print(f"🌀 Sudden Tilt Detected: {'Yes' if sudden_tilt else 'No'}")
print(f"\n🧠 XGBoost Prediction: {xgb_label}")
print(f"🧠 Landmark Model Prediction: {landmark_label}")
print(f"\n🧮 Total Combined Feature Score: {total_score_sum:.5f}")
print(f"\n🎯 Final Liveness Decision: {final_decision}")

# Show result on webcam
cap = cv2.VideoCapture(0)
time.sleep(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    color = (0, 255, 0) if final_decision == "REAL" else (0, 0, 255)
    cv2.putText(frame, f"{final_decision} FACE DETECTED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("🧠 Final Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
