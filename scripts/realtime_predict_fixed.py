# realtime_predict_fixed.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import math
import os
import xgboost as xgb

# ------------------------------------------------
# Step 1: Load Models
# ------------------------------------------------

xgb_model = joblib.load('../models/xgboost_liveness_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# ------------------------------------------------
# Step 2: Initialize Mediapipe and Face Detector
# ------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# ------------------------------------------------
# Step 3: Helper Functions
# ------------------------------------------------

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

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

def compute_deltas(prev, curr):
    return [c - p for p, c in zip(prev, curr)]

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

# ------------------------------------------------
# Step 4: Capture Webcam
# ------------------------------------------------

print("[INFO] Opening webcam...")
cap = cv2.VideoCapture(1)
time.sleep(1)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

max_frames = 300
frame_count = 0

landmark_history = []
tilts = []
ears = []
mars = []
delta_list = []

print("🟡 Capturing 300 frames with DNN Face Detection + Mediapipe...")

prev_landmarks = None

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        continue

    has_face = detect_face_dnn(frame)

    if has_face:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = face_mesh.process(img_rgb)

        if results_mp.multi_face_landmarks:
            for face in results_mp.multi_face_landmarks:
                coords = []
                for lm in face.landmark:
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
                    cv2.putText(frame, f"Tilt: {tilt:.2f} EAR: {ear:.3f} MAR: {mar:.3f}", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("🧠 Capturing Frames", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
time.sleep(1)

print("✅ 300 frames captured!")

# ------------------------------------------------
# Step 5: Calculate Metrics
# ------------------------------------------------

tilt_change = max(tilts) - min(tilts) if tilts else 0
ear_change = max(ears) - min(ears) if ears else 0
mar_change = max(mars) - min(mars) if mars else 0
avg_movement = np.mean(np.abs(delta_list)) if delta_list else 0

# ------------------------------------------------
# Step 6: Predict using XGBoost
# ------------------------------------------------

input_features = pd.DataFrame([{
    'tilt_change': tilt_change,
    'ear_change': ear_change,
    'mar_change': mar_change,
    'avg_movement': avg_movement
}])

final_prediction = xgb_model.predict(input_features)[0]

label_map = {0: 'FAKE', 1: 'REAL'}
final_label = label_map[final_prediction]

# ------------------------------------------------
# Step 7: Print Detailed Summary
# ------------------------------------------------

print("\n📊 Detailed Metrics and Threshold Evaluation:")
print(f"📈 Tilt Change: {tilt_change:.2f}° (Recommended > 6°)")
print(f"👁️ EAR Change: {ear_change:.4f} (Recommended > 0.30)")
print(f"👄 MAR Change: {mar_change:.4f} (Recommended > 0.25)")
print(f"🧠 Landmark Δ (Avg Movement): {avg_movement:.5f} (Recommended > 0.001)")

print("\n📈 Threshold Results:")
print(f"{'PASS' if tilt_change > 6 else 'FAIL'} -> Tilt Change")
print(f"{'PASS' if ear_change > 0.30 else 'FAIL'} -> EAR Change")
print(f"{'PASS' if mar_change > 0.25 else 'FAIL'} -> MAR Change")
print(f"{'PASS' if avg_movement > 0.001 else 'FAIL'} -> Avg Movement")

print(f"\n🎯 Final Prediction by XGBoost Model: {final_label}")

# ------------------------------------------------
# Step 8: Show webcam result
# ------------------------------------------------

print("[INFO] Opening webcam to show final decision...")
cap = cv2.VideoCapture(1)
time.sleep(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    color = (0, 255, 0) if final_label == "REAL" else (0, 0, 255)
    cv2.putText(frame, f"{final_label} FACE DETECTED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("🧠 Final Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
