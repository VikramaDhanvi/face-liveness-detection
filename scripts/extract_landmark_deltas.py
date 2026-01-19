# Importing necessary libraries
import os  # For working with directories and file paths
import cv2  # OpenCV - to read images
import mediapipe as mp  # Mediapipe - to detect face landmarks
import pandas as pd  # Pandas - to save the extracted features into a CSV file
import numpy as np  # Numpy - for numerical operations
import re  # Regular expressions - for sorting frame files properly

# Define input and output paths
INPUT_FRAMES_DIR = '../video_frames'  # Where all face frames are stored
OUTPUT_CSV = '../data/landmark_deltas.csv'  # Where the extracted CSV will be saved

# Initialize Mediapipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)  # We use static_image_mode because we are working with images (not video)

# -----------------------------------
# Functions for Feature Extraction
# -----------------------------------

# Function to compute the difference (delta) between two frames
def compute_deltas(prev, current):
    return [c - p for p, c in zip(prev, current)]  # Subtract previous landmarks from current landmarks

# Function to calculate important distances between key points on the face
def extract_distances(landmarks):
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))  # Compute straight-line distance between two points

    eye_left = (landmarks[33*2], landmarks[33*2+1])  # Left eye center (x, y)
    eye_right = (landmarks[263*2], landmarks[263*2+1])  # Right eye center (x, y)
    nose = (landmarks[1*2], landmarks[1*2+1])  # Nose tip (x, y)
    mouth = (landmarks[13*2], landmarks[13*2+1])  # Center of mouth (x, y)

    return [
        dist(eye_left, eye_right),  # Distance between two eyes
        dist(nose, mouth),  # Distance between nose and mouth
        dist(eye_left, nose),  # Distance between left eye and nose
        dist(eye_right, mouth)  # Distance between right eye and mouth
    ]

# Function to compute eye or mouth aspect ratio (helps detect blinking or mouth movement)
def aspect_ratio(p1, p2, p3, p4, p5, p6):
    vertical1 = np.linalg.norm(np.array(p2) - np.array(p6))  # Vertical distance 1
    vertical2 = np.linalg.norm(np.array(p3) - np.array(p5))  # Vertical distance 2
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))  # Horizontal distance
    return (vertical1 + vertical2) / (2.0 * horizontal)  # EAR or MAR formula

# Function to calculate Eye Aspect Ratio (EAR)
def compute_ear(landmarks):
    eye = [landmarks[i * 2:i * 2 + 2] for i in [362, 385, 387, 263, 373, 380]]  # Take specific points around the right eye
    return aspect_ratio(*eye)  # Calculate EAR

# Function to calculate Mouth Aspect Ratio (MAR)
def compute_mar(landmarks):
    mouth = [landmarks[i * 2:i * 2 + 2] for i in [61, 81, 13, 311, 308, 402]]  # Take specific points around the mouth
    return aspect_ratio(*mouth)  # Calculate MAR

# Function to calculate the tilt of the head using eye positions
def calculate_tilt(landmarks):
    left = np.array([landmarks[33 * 2], landmarks[33 * 2 + 1]])  # Left eye
    right = np.array([landmarks[263 * 2], landmarks[263 * 2 + 1]])  # Right eye
    dx = right[0] - left[0]  # Horizontal difference
    dy = right[1] - left[1]  # Vertical difference
    return np.degrees(np.arctan2(dy, dx))  # Calculate tilt angle in degrees

# Function to sort frame files naturally (e.g., frame_1.jpg, frame_2.jpg, ..., frame_10.jpg)
def sorted_numerically(files):
    return sorted(files, key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else -1)

# -----------------------------------
# Main Feature Extraction Loop
# -----------------------------------

rows = []  # List to hold feature rows for the CSV

# Loop through each label folder (e.g., real, mask, monitor)
for label in os.listdir(INPUT_FRAMES_DIR):
    label_path = os.path.join(INPUT_FRAMES_DIR, label)
    if not os.path.isdir(label_path):
        continue  # Skip if it's not a directory (safety check)

    # Inside each label, loop through each session folder (example: session1, session2)
    for folder in os.listdir(label_path):
        folder_path = os.path.join(label_path, folder)
        prev_landmarks = None  # To store previous frame landmarks
        tilt_list = []  # List to collect tilt angles
        ear_list = []  # List to collect EARs
        mar_list = []  # List to collect MARs

        frame_files = sorted_numerically(os.listdir(folder_path))  # Sort frames numerically

        # Process each frame image
        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            img = cv2.imread(frame_path)  # Read the image
            if img is None:
                continue  # Skip if the image is unreadable

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (for Mediapipe)
            results = face_mesh.process(rgb)  # Detect face landmarks

            if results.multi_face_landmarks:  # If face detected
                for face_landmarks in results.multi_face_landmarks:
                    current_landmarks = []
                    for lm in face_landmarks.landmark:
                        current_landmarks.extend([lm.x, lm.y])  # Flatten (x, y) for all 468 landmarks

                    if len(current_landmarks) != 936:
                        continue  # If not exactly 936 points, skip (safety)

                    if prev_landmarks:
                        deltas = compute_deltas(prev_landmarks, current_landmarks)  # Frame-to-frame movement
                        distances = extract_distances(current_landmarks)  # Key distances between important points
                        tilt = calculate_tilt(current_landmarks)  # Head tilt
                        ear = compute_ear(current_landmarks)  # Eye aspect ratio
                        mar = compute_mar(current_landmarks)  # Mouth aspect ratio
                        tilt_list.append(tilt)  # Save tilt
                        ear_list.append(ear)  # Save EAR
                        mar_list.append(mar)  # Save MAR
                        row = deltas + distances + [tilt, ear, mar, label]  # Full feature vector
                        rows.append(row)  # Add row to dataset

                    prev_landmarks = current_landmarks  # Set current frame as previous for next frame

# -----------------------------------
# After Collecting All Features
# -----------------------------------

# Column names for the CSV
delta_cols = [f'dx{i}' if i % 2 == 0 else f'dy{i//2}' for i in range(936)]  # dx0, dy0, dx2, dy1, ..., for 468 points
dist_cols = ['eye_distance', 'nose_mouth_distance', 'eye_to_nose', 'eye_to_mouth']  # Important distances
extra_cols = ['tilt_deg', 'avg_ear', 'avg_mar']  # Extra features: tilt, EAR, MAR
columns = delta_cols + dist_cols + extra_cols + ['label']  # Full set of column names

# Convert collected data to a Pandas DataFrame
df = pd.DataFrame(rows, columns=columns)

# Make sure output directory exists
os.makedirs('./data', exist_ok=True)

# Save DataFrame as a CSV file
df.to_csv(OUTPUT_CSV, index=False)

# Final printout
print(f"[✓] Extracted features saved to {OUTPUT_CSV} with {len(df)} samples.")
