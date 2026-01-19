import numpy as np

# Eye landmark indices for EAR calculation (left/right eye)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(landmarks):
    def eye_aspect_ratio(eye_indices):
        p = [np.array([landmarks[i * 2], landmarks[i * 2 + 1]]) for i in eye_indices]
        A = euclidean_dist(p[1], p[5])
        B = euclidean_dist(p[2], p[4])
        C = euclidean_dist(p[0], p[3])
        return (A + B) / (2.0 * C)

    left_ear = eye_aspect_ratio(LEFT_EYE)
    right_ear = eye_aspect_ratio(RIGHT_EYE)
    return (left_ear + right_ear) / 2.0
