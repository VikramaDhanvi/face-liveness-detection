cat << 'EOF' > README.md
# Face Liveness Detection System

A real-time face liveness detection system that distinguishes REAL vs FAKE faces using facial landmark motion analysis and hybrid machine learning.

## Core Idea

Instead of relying only on image texture, this system detects natural human micro-movements such as:

- Head tilt variation
- Eye blinking (EAR)
- Mouth movement (MAR)
- Landmark motion between frames

These motion patterns are extremely difficult to spoof using photos, videos, or masks.

---

## Processing Pipeline

Video → Frames → Landmarks → Feature Engineering → Model Training → Real-time Prediction

---

## Feature Engineering

From MediaPipe FaceMesh (468 landmarks):

- Frame-to-frame landmark deltas
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head tilt angle
- Average landmark movement

These form the numerical feature vector.

---

## Models Used

1. Landmark Stacking Model  
   - XGBoost  
   - MLP  
   - Random Forest  
   - Logistic Regression  

2. XGBoost Liveness Model  
   Uses only motion metrics:
   - tilt_change
   - ear_change
   - mar_change
   - avg_movement

3. CNN Model  
   Extracts spatial face features from frames.

4. Hybrid Model  
   Combines landmark + CNN features for final classification.

---

## Decision Logic

Final result = REAL only when:

- Motion thresholds are satisfied
- AND XGBoost predicts REAL
- AND Landmark model predicts real or monitor

Otherwise → FAKE

---

## How to Run

Frame extraction:
python extract_frames.py

Landmark extraction:
python extract_landmark_deltas.py

Model training:
python train_model.py

Real-time prediction:
python realtime_predict.py

---

## Spoof Attacks Detected

- Printed photos
- Mobile screen replay
- Cutouts
- Masks
- Outline faces

---

## Applications

- Face authentication
- Online exam proctoring
- Attendance systems
- Banking security
- Surveillance verification

---

## Author

Paduchuri Vikrama Dhanvi  
B.Tech Graduate, SRM University AP  
GitHub: https://github.com/VikramaDhanvi

---

## Note

Dataset folders and virtual environments are ignored in Git to keep the repository lightweight.

EOF
