import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceExtractor:
    def __init__(self, target_size=224, gpu_id=0):
        self.target_size = target_size

        # Initialize RetinaFace model via InsightFace
        # gpu_id=0 for GPU, gpu_id=-1 to force CPU
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=gpu_id)

    def extract_face(self, frame):
        faces = self.app.get(frame)

        if len(faces) == 0:
            return None

        # Choose largest face by bounding box area
        best_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        x1, y1, x2, y2 = best_face.bbox.astype(int)

        # Clamp to frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            return None

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (self.target_size, self.target_size))
        return face