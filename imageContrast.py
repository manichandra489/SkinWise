import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def analyze_face_contrast_bgr(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    if not detections:
        return 0.0, "No face"

    # largest face
    faces = sorted(detections, key=lambda d: d["box"][2] * d["box"][3], reverse=True)
    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_roi = gray[y:y + h, x:x + w]

    lap = cv2.Laplacian(face_roi, cv2.CV_64F)
    focus_score = lap.var()
    dynamic_range = float(face_roi.max() - face_roi.min())

    contrast = float(
        0.7 * min(focus_score / 100.0, 1.0) +
        0.3 * min(dynamic_range / 128.0, 1.0)
    ) * 100.0

    if focus_score < 15 or dynamic_range < 20:
        status = "Low"
    elif contrast < 40:
        status = "Medium"
    else:
        status = "High"

    return contrast, status

