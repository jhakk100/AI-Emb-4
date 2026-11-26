"""Face Mesh extraction for already captured photos; no movement policy here."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from models import ensure_landmarker


@dataclass(frozen=True)
class Pose:
    yaw: float
    pitch: float
    roll: float
    box: tuple
    landmarks: np.ndarray | None = None


class FaceMeshEstimator:
    """One reusable Tasks model. VIDEO is synchronous: no callback backlog/thread races."""
    def __init__(self, progress=lambda text: None):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("Face Mesh가 없습니다. setup.bat 또는 requirements.txt로 설치하세요.") from exc
        path = ensure_landmarker(progress)
        self.mp = mp
        self.timestamp = -1
        self.model = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2, min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6, min_tracking_confidence=0.6,
            output_facial_transformation_matrixes=True))

    def observe(self, frame, now):
        self.timestamp = max(self.timestamp + 1, int(now * 1000))
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.model.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp)
        if len(result.face_landmarks) != 1 or len(result.facial_transformation_matrixes) != 1:
            return None
        matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return None
        rotation = matrix[:3, :3]
        # Remove small scale/orthogonality errors before Euler decomposition (Rz Ry Rx).
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            return None
        yaw = math.degrees(math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0])))
        pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
        roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        points = np.array([(p.x, p.y) for p in result.face_landmarks[0]])
        if not np.isfinite(points).all():
            return None
        height, width = frame.shape[:2]
        low, high = points.min(axis=0), points.max(axis=0)
        box = (low[0] * width, low[1] * height,
               (high[0] - low[0]) * width, (high[1] - low[1]) * height)
        landmarks = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        return Pose(yaw, pitch, roll, box, landmarks)

    def close(self):
        self.model.close()
