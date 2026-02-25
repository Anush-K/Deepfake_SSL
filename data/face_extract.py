import cv2
import numpy as np
from retinaface import RetinaFace


class FaceExtractor:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def extract_face(self, frame):
        """
        Detect and crop largest face using RetinaFace.
        Returns resized face or None.
        """
        detections = RetinaFace.detect_faces(frame)

        if isinstance(detections, dict):
            # Select largest detected face
            max_area = 0
            best_face = None

            for key in detections:
                x1, y1, x2, y2 = detections[key]["facial_area"]
                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                    best_face = (x1, y1, x2, y2)

            if best_face is not None:
                x1, y1, x2, y2 = best_face
                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, (self.target_size, self.target_size))
                return face

        return None