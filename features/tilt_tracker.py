import numpy as np
import math

def compute_tilt_angle(landmarks):
    left_eye = np.array([landmarks[33 * 2], landmarks[33 * 2 + 1]])
    right_eye = np.array([landmarks[263 * 2], landmarks[263 * 2 + 1]])
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return np.degrees(np.arctan2(dy, dx))
