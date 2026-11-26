"""Single-camera challenge/response. This is not a physical depth sensor or PAD certification."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import secrets

import cv2
import numpy as np

from models import ROOT, ensure_landmarker
from standby import overlap_ratio


@dataclass(frozen=True)
class LivenessSettings:
    total_timeout: float = 25.0
    step_timeout: float = 7.0
    max_gap: float = 0.7
    hold_seconds: float = 0.25
    min_samples: int = 3
    front_yaw: float = 10.0
    turn_yaw: float = 20.0
    max_yaw: float = 45.0
    max_pitch: float = 25.0
    max_roll: float = 15.0
    embedding_interval: float = 0.3
    yaw_sign: int = 1

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"잘못된 동작 검사 설정: {field.name}")
        if (not 0 < self.hold_seconds < self.max_gap < self.step_timeout <= self.total_timeout <= 120
                or not 1 <= self.min_samples <= 30 or type(self.min_samples) is not int
                or not 0 < self.front_yaw < self.turn_yaw < self.max_yaw < 80
                or not 0 < self.max_pitch < 60 or not 0 < self.max_roll < 60
                or not 0 < self.embedding_interval < self.max_gap
                or self.yaw_sign not in (-1, 1)):
            raise ValueError("동작 검사 설정의 범위 또는 시간 순서가 잘못되었습니다.")


def load_settings(path=ROOT / "liveness.json"):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return LivenessSettings(**data)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"동작 검사 설정을 읽을 수 없습니다: {path} ({exc})") from exc


@dataclass(frozen=True)
class Pose:
    yaw: float
    pitch: float
    roll: float
    box: tuple


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
        return Pose(yaw, pitch, roll, box)

    def close(self):
        self.model.close()


class RotationChallenge:
    """Bounded, continuous observations; a terminal result is consumed by the pipeline once."""
    def __init__(self, settings, now, choose=None):
        self.settings = settings
        choose = choose or secrets.choice
        self.steps = (0, choose((-1, 1)), 0, choose((-1, 1)), 0)
        self.index = 0
        self.started = self.step_started = now
        self.last_seen = None
        self.previous_box = None
        self.hold_started = None
        self.samples = 0
        self.complete = False
        self.failure = None

    @property
    def prompt(self):
        if self.failure:
            return self.failure
        if self.complete:
            return "동작 확인 완료 · 정면 본인 재확인"
        target = self.steps[self.index]
        action = ("정면을 보고 잠시 멈춰주세요" if target == 0 else
                  "화면 오른쪽으로 코를 돌려주세요" if target > 0 else
                  "화면 왼쪽으로 코를 돌려주세요")
        return f"동작 검사 {self.index + 1}/{len(self.steps)} · {action}"

    def fail(self, reason):
        self.failure = reason
        return False

    def update(self, pose, now):
        if self.failure or self.complete:
            return self.complete and not self.failure
        s = self.settings
        if now - self.started >= s.total_timeout or now - self.step_started >= s.step_timeout:
            return self.fail("동작 검사 시간 초과 · 다시 시도하세요")
        if pose is None:
            return self.fail("얼굴 추적 중단 · 다시 시도하세요")
        if (not all(math.isfinite(v) for v in (pose.yaw, pose.pitch, pose.roll, *pose.box))
                or min(pose.box[2:]) <= 0):
            return self.fail("얼굴 자세 데이터 오류")
        if self.last_seen is not None and (now <= self.last_seen or now - self.last_seen > s.max_gap):
            return self.fail("연속 영상이 끊겼습니다 · 다시 시도하세요")
        if self.previous_box is not None and overlap_ratio(self.previous_box, pose.box) < 0.25:
            return self.fail("얼굴 위치가 급변했습니다 · 다시 시도하세요")
        self.last_seen, self.previous_box = now, pose.box
        yaw = pose.yaw * s.yaw_sign
        target = self.steps[self.index]
        valid = (abs(pose.pitch) <= s.max_pitch and abs(pose.roll) <= s.max_roll
                 and (abs(yaw) <= s.front_yaw if target == 0 else
                      s.turn_yaw <= yaw * target <= s.max_yaw))
        if not valid:
            self.hold_started, self.samples = None, 0
            return False
        if self.hold_started is None:
            self.hold_started = now
        self.samples += 1
        if self.samples >= s.min_samples and now - self.hold_started >= s.hold_seconds:
            self.index += 1
            self.hold_started, self.samples = None, 0
            self.step_started = now
            self.complete = self.index == len(self.steps)
        return self.complete
