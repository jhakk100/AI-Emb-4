"""One camera, one authentication transaction, no unlock before challenge completion."""

from dataclasses import dataclass, replace
import math
from time import monotonic

import cv2

from authenticate import authenticate_face
from capture_control import ACTIVE_PROFILE, STANDBY_PROFILE
from database import initialize_database
from face_core import DEFAULT_THRESHOLD
from liveness import FaceMeshEstimator, RotationChallenge, load_settings
from standby import FaceWatcher, overlap_ratio


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
    authentication_attempted: bool = False
    liveness_passed: bool = False
    liveness_step: str = ""
    yaw: float | None = None
    session_id: int = 0


class FacePipeline:
    def __init__(self, engine, watcher=None, store=None, source="camera:0",
                 save_interval=1.0, progress=None, threshold=DEFAULT_THRESHOLD,
                 start_in_standby=True, mesh=None, settings=None, purpose="authenticate"):
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
        self._threshold = DEFAULT_THRESHOLD
        self.threshold = threshold
        self.settings = settings if settings is not None else load_settings()
        self.mesh = mesh
        self.challenge = None
        self.session_id = 0
        self.purpose = purpose
        self._clear_attempt()
        if purpose not in ("authenticate", "enroll", "preview", "departure"):
            raise ValueError("잘못된 카메라 사용 목적")

    def _clear_attempt(self):
        self.challenge = None
        self.identity = None
        self.reference_embedding = None
        self.last_identity_at = -float("inf")
        self.terminal = False
        self.active_started = None

    @property
    def profile(self):
        return STANDBY_PROFILE if self.mode == "standby" else ACTIVE_PROFILE

    def reset_standby(self):
        self.mode = "standby"
        self.watcher = FaceWatcher(self.gate_settings)
        self._clear_attempt()

    def set_purpose(self, purpose, session_id=None):
        if purpose not in ("authenticate", "enroll", "preview", "departure"):
            raise ValueError("잘못된 카메라 사용 목적")
        self.purpose = purpose
        if session_id is not None:
            self.session_id = session_id
        self.reset_standby()

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("인식 기준은 0~1 사이여야 합니다.")
        if value != self._threshold:
            self._threshold = value
            if hasattr(self, "challenge"):
                self.reset_standby()

    @property
    def save_interval(self):
        return self._save_interval

    @save_interval.setter
    def save_interval(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("저장 간격은 0 이상의 유한한 숫자여야 합니다.")
        self._save_interval = value

    def _result(self, profile, detections, status, **kwargs):
        self.progress(status)
        return FrameResult([], False, status, self.saved_count, 0, 0,
                           len(detections), 0, power_mode=profile.name,
                           processing_size=profile.size, target_fps=profile.fps,
                           detection_boxes=[tuple(float(v) for v in row[:4]) for row in detections],
                           session_id=self.session_id, **kwargs)

    def _failure(self, profile, detections, message, score=None):
        self.terminal = True
        result = self._result(profile, detections, message)
        result.ready = True
        result.match_score = score
        result.authentication_attempted = True
        return result

    def _analyze(self, frame, detections):
        faces = self.engine.analyze(frame, detections)
        if len(faces) != 1 or faces[0].sharpness < 50 or min(faces[0].box[2:]) < 90:
            raise ValueError("얼굴이 작거나 흐립니다. 가까이에서 잠시 멈춰주세요.")
        return faces[0]

    def process(self, frame):
        profile = self.profile
        if self.terminal:
            return self._result(profile, [], "결과 처리 대기")
        detections = self.engine.detect(frame, size=profile.size)
        now = monotonic()
        if self.purpose == "departure":
            return self._result(profile, detections, "화면에서 벗어나 주세요")
        if self.mode == "standby":
            if len(detections) == 1:
                self.watcher.update(frame, detections)
            else:
                self.watcher.ready = False
                self.watcher.reset_hold()
                self.watcher.reason = "얼굴 대기 중" if not len(detections) else "얼굴은 한 명만 보여주세요"
            reason = self.watcher.reason
            if self.watcher.ready:
                self.mode = "active"
                self.active_started = now
                self.watcher = FaceWatcher(replace(self.gate_settings, hold_seconds=0))
                reason = "준비 완료 · 다음 프레임부터 활성 검사"
            return self._result(profile, detections, reason)
        if self.active_started is None:
            self.active_started = now
        if len(detections) != 1:
            if self.challenge:
                return self._failure(profile, detections, "얼굴 소실 또는 여러 얼굴 · 검사 실패")
            self.reset_standby()
            return self._result(profile, detections, "얼굴은 한 명만 보여주세요")
        if self.challenge is None:
            self.watcher.update(frame, detections)
            if not self.watcher.ready:
                self.reset_standby()
                return self._result(profile, detections, "정면에서 촬영 준비를 다시 확인합니다")
        try:
            if self.purpose in ("preview", "enroll"):
                face = self._analyze(frame, detections)
                result = self._result(profile, detections, "등록 준비 완료 · 선명한 정면 얼굴")
                result.faces, result.ready = [face], True
                return result
            if self.identity is None:
                self.store.refresh()
                if not self.store.people:
                    face = self._analyze(frame, detections)
                    result = self._result(profile, detections, "등록된 사용자가 없습니다 · 사용자 등록 탭을 열어주세요")
                    result.faces, result.ready = [face], True
                    return result
            # Check deadlines before model work; no stale success can survive a long frame gap.
            if self.challenge and (now - self.challenge.started >= self.settings.total_timeout
                    or now - self.challenge.step_started >= self.settings.step_timeout):
                return self._failure(profile, detections, "동작 검사 시간 초과")
            if now - self.active_started >= self.settings.total_timeout:
                return self._failure(profile, detections, "활성 인증 시간 초과")
            if self.mesh is None:
                self.mesh = FaceMeshEstimator(self.progress)
                # Model startup/download is outside the user's response window.
                now = monotonic()
                self.active_started = now
            pose = self.mesh.observe(frame, now)
            if pose is None or overlap_ratio(tuple(detections[0][:4]), pose.box) < 0.25:
                if self.challenge:
                    return self._failure(profile, detections, "Face Mesh 추적 중단 또는 얼굴 불일치")
                return self._result(profile, detections, "얼굴 특징점을 확인하고 있습니다")
            final_front = self.challenge is not None and self.challenge.index == len(self.challenge.steps) - 1
            due = self.identity is None or final_front or now - self.last_identity_at >= self.settings.embedding_interval
            face, score = None, None
            if due:
                face = self._analyze(frame, detections)
                self.store.refresh()
                name, score = authenticate_face(self.store, face.embedding, self.threshold)
                if name is None:
                    return self._failure(profile, detections, "등록된 얼굴과 일치하지 않습니다", score)
                if self.identity is not None and (name != self.identity or
                        float(self.reference_embedding @ face.embedding) < self.threshold):
                    return self._failure(profile, detections, "검사 중 얼굴이 변경되었습니다", score)
                if self.identity is None:
                    self.identity, self.reference_embedding = name, face.embedding.copy()
                self.last_identity_at = now
            if self.challenge is None:
                self.challenge = RotationChallenge(self.settings, now)
            self.challenge.update(pose, now)
            if self.challenge.failure:
                return self._failure(profile, detections, self.challenge.failure)
            result = self._result(profile, detections, self.challenge.prompt)
            result.yaw, result.liveness_step = pose.yaw, self.challenge.prompt
            if not self.challenge.complete:
                return result
            # Completion happens only on a freshly matched, frontal frame.
            if face is None:
                return self._failure(profile, detections, "최종 본인 확인 누락")
            result.ready, result.authentication_attempted, result.liveness_passed = True, True, True
            result.faces, result.authenticated_name, result.match_score = [face], self.identity, score
            if now - self.last_saved_at >= self.save_interval:
                sample_id = self.store.record_authenticated(frame, detections[0], face.embedding,
                    self.source, gate_passed=True, threshold=self.threshold)
                if sample_id is None:
                    return self._failure(profile, detections, "최종 등록 정보가 변경되었습니다")
                result.saved_this_frame = 1
                self.saved_count += 1
                self.last_saved_at = now
            result.saved_count = self.saved_count
            result.status = "본인 확인 및 무작위 동작 검사 통과"
            self.terminal = True
            return result
        except (cv2.error, ValueError, IndexError) as exc:
            if self.challenge:
                result = self._failure(profile, detections, f"검사 실패: {exc}")
            else:
                result = self._result(profile, detections, f"영상 품질 확인: {exc}")
                if now - self.active_started >= self.settings.total_timeout:
                    result = self._failure(profile, detections, "영상 품질 확인 시간 초과")
            result.embedding_errors = 1
            return result

    def close(self):
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None
