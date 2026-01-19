"""
realtime_predict.py ─ fully-updated version
───────────────────────────────────────────
Key fixes
• Uses MediaPipe FaceMesh in *streaming* mode (static_image_mode=False).
• Wraps FaceMesh in a context-manager so resources are released cleanly.
• Opens the default webcam (index 0) with DirectShow on Windows.
• Adds a quick version sanity-check so you spot wrong mediapipe/protobuf builds early.
Everything else (feature extraction, scoring, model inference, result overlay, CSV save)
is unchanged.
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import math
import os
import sys

# ------------------------------------------------
# Sanity-check Mediapipe / protobuf combo
# ------------------------------------------------
MIN_MP_VERSION = (0, 10, 14)        # last fully-tested wheel
if tuple(map(int, mp.__version__.split('.'))) < MIN_MP_VERSION:
    sys.exit(f"[FATAL] mediapipe {mp.__version__} too old. Please `pip install mediapipe==0.10.14` or newer.")

try:
    import google.protobuf  # noqa: F401
except ImportError:
    sys.exit("[FATAL] protobuf missing. `pip install 'protobuf<=3.21.12'` (Windows)")

# ------------------------------------------------
# Step 1 : Load our pre-trained models
# ------------------------------------------------
model  = joblib.load('../models/landmark_model.pkl')
scaler = joblib.load('../models/scaler.pkl')
labels = ['mask', 'monitor', 'outline', 'print', 'print_cut', 'real', 'silicone']

# ------------------------------------------------
# Step 2 : Helper geometry functions
# ------------------------------------------------
def dist(p1, p2):               return np.linalg.norm(np.array(p1) - np.array(p2))
def compute_deltas(prev, curr): return [c - p for p, c in zip(prev, curr)]

def extract_distances(landmarks):
    eye_left  = (landmarks[33*2],  landmarks[33*2+1])
    eye_right = (landmarks[263*2], landmarks[263*2+1])
    nose      = (landmarks[1*2],   landmarks[1*2+1])
    mouth     = (landmarks[13*2],  landmarks[13*2+1])
    return [
        dist(eye_left, eye_right),
        dist(nose, mouth),
        dist(eye_left, nose),
        dist(eye_right, mouth)
    ]

def calculate_head_tilt(landmarks):
    left_eye  = np.array([landmarks[33*2],  landmarks[33*2+1]])
    right_eye = np.array([landmarks[263*2], landmarks[263*2+1]])
    dx, dy = right_eye - left_eye
    return np.degrees(math.atan2(dy, dx))

def eye_aspect_ratio(landmarks):
    p1, p2 = np.array([landmarks[159*2], landmarks[159*2+1]]), np.array([landmarks[145*2], landmarks[145*2+1]])
    p3, p4 = np.array([landmarks[33*2],  landmarks[33*2+1]]),  np.array([landmarks[133*2], landmarks[133*2+1]])
    return dist(p1, p2) / dist(p3, p4)

def mouth_aspect_ratio(landmarks):
    top, bottom = np.array([landmarks[13*2],  landmarks[13*2+1]]), np.array([landmarks[14*2],  landmarks[14*2+1]])
    left, right = np.array([landmarks[78*2], landmarks[78*2+1]]), np.array([landmarks[308*2], landmarks[308*2+1]])
    return dist(top, bottom) / dist(left, right)

def detect_face_dnn(frame, net):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104, 177, 123), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > 0.5:
            return True
    return False

def detect_sudden_tilt(tilts, thresh=15.0):
    return any(abs(tilts[i] - tilts[i-1]) > thresh for i in range(1, len(tilts)))

# ------------------------------------------------
# Step 3 : Initialise face-detector & webcam
# ------------------------------------------------
print("[INFO] Loading OpenCV DNN face-detector …")
face_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

print("[INFO] Opening webcam (device 0) …")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    sys.exit("❌ Could not open webcam (index 0).")

time.sleep(1)

# ------------------------------------------------
# Step 4 : Capture loop (300 frames)
# ------------------------------------------------
MAX_FRAMES   = 300
landmark_hist, tilts, ears, mars, delta_list = [], [], [], [], []
prev_landmarks = None
frame_count    = 0

print("🟡 Capturing 300 frames with DNN + FaceMesh …")

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
        static_image_mode=False,   # STREAMING mode -- important!
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        if detect_face_dnn(frame, face_net):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                coords = [coord for lm in face.landmark for coord in (lm.x, lm.y)]

                if len(coords) == 468 * 2:        # Sanity check
                    tilt = calculate_head_tilt(coords)
                    ear  = eye_aspect_ratio(coords)
                    mar  = mouth_aspect_ratio(coords)

                    tilts.append(tilt); ears.append(ear); mars.append(mar)
                    landmark_hist.append(coords)

                    if prev_landmarks is not None:
                        delta_list.append(compute_deltas(prev_landmarks, coords))
                    prev_landmarks = coords

                    frame_count += 1

                    # HUD overlay
                    cv2.putText(frame, f"Frame {frame_count}/{MAX_FRAMES}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                    cv2.putText(frame, f"Tilt {tilt:+6.2f}°  EAR {ear:.3f}  MAR {mar:.3f}",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

        cv2.imshow("🧠 Capturing Live Frames", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("✅ 300 frames captured!")

# ------------------------------------------------
# Step 5 : Metric computation & scoring
# ------------------------------------------------
tilt_change   = max(tilts) - min(tilts) if tilts else 0.0
ear_change    = max(ears)  - min(ears)  if ears  else 0.0
mar_change    = max(mars)  - min(mars)  if mars  else 0.0
avg_movement  = np.mean(np.abs(delta_list)) if delta_list else 0.0
sudden_tilt   = detect_sudden_tilt(tilts)

score = 0.0
if tilt_change >= 6.0 and not sudden_tilt: score += 0.8
if ear_change  > 0.01 or mar_change > 0.01: score += 0.8
if avg_movement > 0.001: score += 0.8

# ------------------------------------------------
# Step 6 : Model inference on last frame
# ------------------------------------------------
predicted_label = "unknown"
if landmark_hist:
    deltas     = compute_deltas(landmark_hist[0], landmark_hist[-1])
    distances  = extract_distances(landmark_hist[-1])
    features   = deltas + distances + [ears[-1], mars[-1], tilts[-1]]

    scaled     = scaler.transform(pd.DataFrame([features]))
    predicted_label = labels[model.predict(scaled)[0]]

    if predicted_label in {"real", "monitor"}:
        score += 0.8

# ------------------------------------------------
# Step 7 : Decision & summary
# ------------------------------------------------
final_decision = "REAL" if score >= 2.4 else "FAKE"
status_color   = (0, 0, 255) if final_decision == "REAL" else (0, 255, 255)

print("\n📊 Detailed Summary")
print(f" Tilt Δ {tilt_change:.2f}°  (≥6 & smooth)      -> {'✔ +0.8' if tilt_change>=6 and not sudden_tilt else '✘ +0.0'}")
print(f" EAR Δ {ear_change:.4f} | MAR Δ {mar_change:.4f}  (>0.01) -> {'✔ +0.8' if ear_change>0.01 or mar_change>0.01 else '✘ +0.0'}")
print(f" Mean |Δlandmark| {avg_movement:.5f}           (>0.001) -> {'✔ +0.8' if avg_movement>0.001 else '✘ +0.0'}")
print(f" Model predicts '{predicted_label}'              -> {'✔ +0.8' if predicted_label in {'real','monitor'} else '✘ +0.0'}")
print(f" → Final score {score:.2f} / 3.2  =>  **{final_decision}**")

# ------------------------------------------------
# Step 8 : Save run metadata
# ------------------------------------------------
os.makedirs('../collected', exist_ok=True)
pd.DataFrame([{
    'tilt_change': tilt_change,
    'ear_change':  ear_change,
    'mar_change':  mar_change,
    'avg_movement': avg_movement,
    'model_pred': predicted_label
}]).to_csv('../collected/temp_run_result.csv', index=False)
print("📁 Saved run summary to ../collected/temp_run_result.csv")

# ------------------------------------------------
# Step 9 : Live overlay with final decision
# ------------------------------------------------
print("[INFO] Re-opening webcam for final overlay …")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"{final_decision} FACE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    cv2.putText(frame, f"Tilt Δ {tilt_change:.2f}°",  (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"EAR Δ {ear_change:.4f} | MAR Δ {mar_change:.4f}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"|Δlm| {avg_movement:.5f}  Score {score:.2f}/3.2", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("🧠 Final Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
