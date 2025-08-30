# utils.py
import numpy as np
import math

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def landmarks_to_points(face_landmarks, image_shape):
    """
    Convert MediaPipe normalized landmarks to list of pixel (x,y) tuples.
    """
    h, w = image_shape[:2]
    pts = []
    for lm in face_landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts

def calculate_EAR(landmarks, eye_indices):
    """
    landmarks: list of (x,y) pixels
    eye_indices: 6 indices [p1, p2, p3, p4, p5, p6] from MediaPipe
    Returns EAR (float)
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    # vertical distances
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    # horizontal distance
    h = euclidean(p1, p4)
    if h == 0:
        return 0.0
    ear = (v1 + v2) / (2.0 * h)
    return ear

def eye_center(landmarks, eye_indices):
    xs = [landmarks[i][0] for i in eye_indices]
    ys = [landmarks[i][1] for i in eye_indices]
    return (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))

def head_direction(landmarks, left_eye_idx, right_eye_idx, nose_idx=1):
    """
    Simple heuristic for head yaw/pitch:
    - compute eye centers and nose x/y
    - normalize displacement by distance between eyes (face scale)
    Returns: (direction_string, dx_norm, dy_norm)
    dx_norm > threshold => nose right (head right)
    dx_norm < -threshold => nose left (head left)
    dy_norm > threshold => nose down (head down)
    """
    left_center = eye_center(landmarks, left_eye_idx)
    right_center = eye_center(landmarks, right_eye_idx)
    eye_mid_x = (left_center[0] + right_center[0]) / 2.0
    eye_mid_y = (left_center[1] + right_center[1]) / 2.0

    # nose point
    try:
        nose = landmarks[nose_idx]
    except Exception:
        nose = (eye_mid_x, eye_mid_y)

    # face scale ~ distance between eye centers
    face_scale = euclidean(left_center, right_center)
    if face_scale == 0:
        face_scale = 1.0

    dx_norm = (nose[0] - eye_mid_x) / face_scale
    dy_norm = (nose[1] - eye_mid_y) / face_scale

    # basic direction
    dir_str = "forward"
    if dx_norm < -0.18:
        dir_str = "left"
    elif dx_norm > 0.18:
        dir_str = "right"
    elif dy_norm > 0.20:
        dir_str = "down"
    elif dy_norm < -0.18:
        dir_str = "up"

    return dir_str, dx_norm, dy_norm
