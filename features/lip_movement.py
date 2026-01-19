import numpy as np

# Lip indices
UPPER_LIP = 13
LOWER_LIP = 14

def compute_mouth_ratio(landmarks):
    upper = np.array([landmarks[UPPER_LIP * 2], landmarks[UPPER_LIP * 2 + 1]])
    lower = np.array([landmarks[LOWER_LIP * 2], landmarks[LOWER_LIP * 2 + 1]])
    return np.linalg.norm(upper - lower)
