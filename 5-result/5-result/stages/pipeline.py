"""One camera stream; stage-2 crops and stage-3 embeddings share record IDs."""

from dataclasses import dataclass, replace
import math
from time import monotonic

import cv2

from authenticate import authenticate_face
from capture_control import ACTIVE_PROFILE, STANDBY_PROFILE
from database import initialize_database
from face_core import DEFAULT_THRESHOLD
from standby import FaceWatcher


@dataclass
class FrameResult:
    faces: list
    ready: bool
    status: str
    saved_count: int
    saved_this_frame: int
    next_save_in: float
    detected_count: int
    embedding_errors: int
    authenticated_name: str | None = None
    match_score: float | None = None
    power_mode: str = "active"
    processing_size: tuple = (640, 480)
    target_fps: int = 20
    detection_boxes: list | None = None
    capture_size: tuple | None = None
    capture_fps: float | None = None


class FacePipeline:
    def __init__(self, engine, watcher=None, store=None, source="camera:0",
                 save_interval=1.0, progress=None, threshold=DEFAULT_THRESHOLD,
                 start_in_standby=True):
        self.engine = engine
        self.watcher = watcher if watcher is not None else FaceWatcher()
        self.gate_settings = self.watcher.settings
        self.mode = "standby" if start_in_standby else "active"
        self.store = store if store is not None else initialize_database()
        self.source = source
        self.saved_count = self.store.count_observations()
        self.save_interval = save_interval
        self.last_saved_at = -float("inf")
        self.progress = progress or (lambda text: None)
        self.threshold = threshold

    @property
    def profile(self):
        return STANDBY_PROFILE if self.mode == "standby" else ACTIVE_PROFILE

    def reset_standby(self):
        self.mode = "standby"
        self.watcher = FaceWatcher(self.gate_settings)

    @property
    def save_interval(self):
        return self._save_interval

    @save_interval.setter
    def save_interval(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("저장 간격은 0 이상의 유한한 숫자여야 합니다.")
        self._save_interval = value

    def process(self, frame):
        profile = self.profile
        self.progress("1단계 · 절전 얼굴 검출" if self.mode == "standby" else "2단계 · 얼굴 검출 / Crop 영역 계산")
        detections = self.engine.detect(frame, size=profile.size)
        self.progress("1단계 · 위치·정면·유지 시간 확인")
        if len(detections) == 1:
            self.watcher.update(frame, detections)
        else:
            self.watcher.ready = False
            self.watcher.reset_hold()
            self.watcher.reason = ("얼굴 대기 중" if len(detections) == 0
                                   else "인증하려면 화면에 얼굴이 한 명만 나오도록 해주세요.")
        # Standby only runs YuNet. SFace inference and DB access wait for wake-up.
        if profile.name == "standby" or not self.watcher.ready:
            reason = self.watcher.reason
            if profile.name == "standby" and self.watcher.ready:
                self.mode = "active"
                self.watcher = FaceWatcher(replace(self.gate_settings, hold_seconds=0))
                reason += " · 다음 프레임부터 고해상도 인증"
            elif profile.name == "active" and self.watcher.valid_since is None:
                self.reset_standby()
            self.progress("1단계 · " + ("절전 대기" if self.mode == "standby" else "고해상도 인증 전환"))
            return FrameResult([], False, reason, self.saved_count, 0, 0, len(detections), 0,
                               power_mode=profile.name, processing_size=profile.size,
                               target_fps=profile.fps,
                               detection_boxes=[tuple(float(v) for v in row[:4]) for row in detections])
        faces, errors = [], 0
        self.progress("3단계 · 얼굴 정렬 / 128차원 임베딩 생성")
        for detection in detections:
            try:
                face = self.engine.analyze(frame, [detection])[0]
            except (cv2.error, ValueError, IndexError) as exc:
                errors += 1
            else:
                faces.append(face)
        ready = self.watcher.ready and not errors
        name, score, saved = None, None, 0
        if ready and len(faces) == 1:
            self.progress("4단계 · 등록 사용자 인증")
            # Enrollment/deletion happens on the UI connection; refresh first.
            self.store.refresh()
            name, score = authenticate_face(self.store, faces[0].embedding, self.threshold)
            now = monotonic()
            if name is not None and now - self.last_saved_at >= self.save_interval:
                self.progress("4단계 · 인증 성공 얼굴을 DB에 저장")
                sample_id = self.store.record_authenticated(
                    frame, detections[0], faces[0].embedding, self.source,
                    gate_passed=True, threshold=self.threshold)
                if sample_id is not None:
                    saved = 1
                    self.last_saved_at = now
        self.saved_count += saved
        status = self.watcher.reason
        if errors:
            status += f" · 임베딩 실패 {errors}개"
        if not len(detections):
            self.progress("1단계 · 얼굴 검출 대기")
        else:
            self.progress("4단계 · 인증 성공 얼굴 저장 완료" if saved else
                          ("4단계 · 인증 성공 / 다음 저장 대기" if name is not None
                           else "4단계 · 인증 미통과 / DB 저장 안 함"))
        next_save_in = max(0.0, self.save_interval - (monotonic() - self.last_saved_at))
        return FrameResult(faces, ready, status, self.saved_count,
                           saved, next_save_in, len(detections), errors, name, score,
                           power_mode=profile.name, processing_size=profile.size, target_fps=profile.fps)
