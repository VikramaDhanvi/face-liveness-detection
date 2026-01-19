"""
Liveness-detection pipeline
---------------------------
* FaceMesh runs in streaming mode (static_image_mode=False) for higher FPS.
* A single cv2.VideoCapture handle is reused throughout – we DON’T reopen
  the camera later.
* Tested with:
    mediapipe 0.10.24      (pip install "mediapipe>=0.10.21,<0.11")
    protobuf   3.20.3      (pip install "protobuf>=3.20.3,<4")
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------
import os, time, math, cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import xgboost as xgb

# ------------------------------------------------
# Step 1  |  Load models & scalers
# ------------------------------------------------
xgb_model      = joblib.load('../models/xgboost_liveness_model.pkl')
scaler_xgb     = joblib.load('../models/scaler.pkl')

landmark_model = joblib.load('../models/landmark_model.pkl')
scaler_landmark = joblib.load('../models/scaler.pkl')
labels_landmark = ['mask', 'monitor', 'outline', 'print',
                   'print_cut', 'real', 'silicone']

# ------------------------------------------------
# Step 2  |  Init FaceMesh & DNN face detector
# ------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# ------------------------------------------------
# Helper funcs
# ------------------------------------------------
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_deltas(prev, curr):
    return [c - p for p, c in zip(prev, curr)]

def extract_distances(landmarks):
    eye_left  = (landmarks[ 33*2], landmarks[ 33*2+1])
    eye_right = (landmarks[263*2], landmarks[263*2+1])
    nose      = (landmarks[  1*2], landmarks[  1*2+1])
    mouth     = (landmarks[ 13*2], landmarks[ 13*2+1])
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
    return math.degrees(math.atan2(dy, dx))

def eye_aspect_ratio(landmarks):
    p1 = np.array([landmarks[159*2], landmarks[159*2+1]])
    p2 = np.array([landmarks[145*2], landmarks[145*2+1]])
    p3 = np.array([landmarks[ 33*2], landmarks[ 33*2+1]])
    p4 = np.array([landmarks[133*2], landmarks[133*2+1]])
    return dist(p1, p2) / dist(p3, p4)

def mouth_aspect_ratio(landmarks):
    top    = np.array([landmarks[ 13*2], landmarks[ 13*2+1]])
    bottom = np.array([landmarks[ 14*2], landmarks[ 14*2+1]])
    left   = np.array([landmarks[ 78*2], landmarks[ 78*2+1]])
    right  = np.array([landmarks[308*2], landmarks[308*2+1]])
    return dist(top, bottom) / dist(left, right)

def detect_face_dnn(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 (104,177,123), swapRB=False)
    face_net.setInput(blob)
    det = face_net.forward()
    h, w = frame.shape[:2]
    for i in range(det.shape[2]):
        if det[0,0,i,2] > 0.5:
            return True
    return False

def detect_sudden_tilt(tilts, threshold=15.0):
    return any(abs(t2-t1) > threshold for t1,t2 in zip(tilts, tilts[1:]))

# ------------------------------------------------
# Step 3  |  Open webcam & capture 300 frames
# ------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ Could not open webcam")

MAX_FRAMES = 300
tilts, ears, mars = [], [], []
landmark_history, delta_list = [], []
prev_landmarks = None

print("[INFO] Capturing 300 frames – press Q to abort")

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while len(tilts) < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok: continue

        if detect_face_dnn(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                coords = [c for point in lm for c in (point.x, point.y)]

                if len(coords) == 936:                      # 468 landmarks × 2
                    # basic metrics
                    tilts.append(calculate_head_tilt(coords))
                    ears.append(eye_aspect_ratio(coords))
                    mars.append(mouth_aspect_ratio(coords))
                    landmark_history.append(coords)

                    if prev_landmarks:
                        delta_list.append(compute_deltas(prev_landmarks, coords))
                    prev_landmarks = coords

        # overlay frame counter
        cv2.putText(frame, f"Frame {len(tilts)}/{MAX_FRAMES}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("Capturing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ------------------------------------------------
# Step 4  |  Feature engineering
# ------------------------------------------------
tilt_change = max(tilts)-min(tilts) if tilts else 0
ear_change  = max(ears)-min(ears)   if ears  else 0
mar_change  = max(mars)-min(mars)   if mars  else 0
avg_mov     = np.mean(np.abs(delta_list)) if delta_list else 0
sudden_tilt = detect_sudden_tilt(tilts)

# ------------------------------------------------
# Step 5  |  Model predictions
# ------------------------------------------------
xgb_feats = pd.DataFrame([{
    'tilt_change': tilt_change,
    'ear_change' : ear_change,
    'mar_change' : mar_change,
    'avg_movement': avg_mov
}])
xgb_label = {0:'FAKE', 1:'REAL'}[ int(xgb_model.predict(xgb_feats)[0]) ]

if landmark_history:
    first, last = landmark_history[0], landmark_history[-1]
    deltas  = compute_deltas(first, last)
    dists   = extract_distances(last)
    feats   = deltas + dists + [ears[-1], mars[-1], tilts[-1]]
    scaled  = scaler_landmark.transform(pd.DataFrame([feats]))
    landmark_label = labels_landmark[ int(landmark_model.predict(scaled)[0]) ]
else:
    landmark_label = "unknown"

final_decision = ("REAL" if
                  (xgb_label=="REAL" and
                   landmark_label in ("real","monitor"))
                  else "FAKE")

# ------------------------------------------------
# Step 6  |  Console summary
# ------------------------------------------------
print("\n=== Results ===")
print(f"Tilt Δ   : {tilt_change:6.2f}°")
print(f"EAR  Δ   : {ear_change:6.4f}")
print(f"MAR  Δ   : {mar_change:6.4f}")
print(f"Avg Δ    : {avg_mov:6.5f}")
print(f"SuddenTilt: {'Yes' if sudden_tilt else 'No'}")
print(f"XGBoost  : {xgb_label}")
print(f"Landmark : {landmark_label}")
print(f"FINAL    : {final_decision}")

# ------------------------------------------------
# Step 7  |  Display final decision live
# ------------------------------------------------
while True:
    ok, frame = cap.read()
    if not ok: break
    color = (0,255,0) if final_decision=="REAL" else (0,0,255)
    cv2.putText(frame, f"{final_decision} FACE DETECTED",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Liveness Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


