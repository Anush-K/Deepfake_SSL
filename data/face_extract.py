import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceExtractor:
    def __init__(self, target_size=224):
        self.target_size = target_size

        # Initialize RetinaFace model via InsightFace
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0)

    def extract_face(self, frame):
        faces = self.app.get(frame)

        if len(faces) == 0:
            return None

        # Choose largest face
        max_area = 0
        best_face = None

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                best_face = (x1, y1, x2, y2)

        if best_face is not None:
            x1, y1, x2, y2 = best_face

            # Ensure valid bounds
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (self.target_size, self.target_size))
            return face

        return None