"""Stages 1-4: YuNet direction guidance, snapshot analysis, directional DB matching."""

from dataclasses import dataclass, replace
import math
from time import monotonic

import cv2

from authenticate import authenticate_face
from capture_control import ACTIVE_PROFILE, STANDBY_PROFILE, ENROLLMENT_PROFILE
from database import initialize_database
from detection import crop_face
from face_core import DEFAULT_THRESHOLD
from guided_capture import (ARROWS, CaptureSample, GuidedCapture, compare_sample,
                            load_guided_settings, normalized_mesh, yunet_pose)
from liveness import FaceMeshEstimator, Pose
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
    power_mode: str = 'active'
    processing_size: tuple = (640, 480)
    target_fps: int = 20
    detection_boxes: list | None = None
    capture_size: tuple | None = None
    capture_fps: float | None = None
    authentication_attempted: bool = False
    liveness_passed: bool = False
    liveness_step: str = ''
    yaw: float | None = None
    pitch: float | None = None
    direction: str | None = None
    arrow: str = ''
    session_id: int = 0
    enrollment_count: int = 0
    enrollment_completed: str | None = None
    enrollment_error: str | None = None


class FacePipeline:
    def __init__(self, engine, watcher=None, store=None, source='camera:0',
                 save_interval=1., progress=None, threshold=DEFAULT_THRESHOLD,
                 start_in_standby=True, mesh=None, settings=None, purpose='authenticate',
                 enrollment_name=None, pose_estimator=yunet_pose, session_valid=None):
        self.engine, self.mesh = engine, mesh
        self.watcher = watcher if watcher is not None else FaceWatcher()
        self.gate_settings = self.watcher.settings
        self.store = store if store is not None else initialize_database()
        self.saved_count = self.store.count_observations()
        self.source, self.progress = source, progress or (lambda text: None)
        self.save_interval = save_interval  # retained for existing UI; an auth pair is always atomic
        self._threshold = float(threshold)
        if not math.isfinite(self._threshold) or not 0 <= self._threshold <= 1:
            raise ValueError('인식 기준은 0~1 사이여야 합니다.')
        self.settings = settings if settings is not None else load_guided_settings()
        self.pose_estimator = pose_estimator
        self.session_valid = session_valid or (lambda: True)
        self.session_id = 0
        self.set_purpose(purpose, enrollment_name=enrollment_name)
        if purpose != 'enroll':
            self.mode = 'standby' if start_in_standby else 'active'

    @property
    def profile(self):
        if self.purpose == 'enroll':
            return ENROLLMENT_PROFILE
        return STANDBY_PROFILE if self.mode == 'standby' else ACTIVE_PROFILE

    @property
    def enrollment_locked(self):
        return getattr(self, 'purpose', None) == 'enroll' and not getattr(self, 'terminal', False)

    def _reset_state(self):
        self.mode = 'enrollment' if self.purpose == 'enroll' else 'standby'
        self.watcher = FaceWatcher(self.gate_settings)
        self.challenge = None
        self.identity = self.reference_embedding = None
        self.terminal = False
        self.active_started = None
        self.pose_origin = None

    def reset_standby(self):
        if self.purpose == 'enroll':
            return False
        self._reset_state()
        return True

    def set_purpose(self, purpose, session_id=None, enrollment_name=None, cancel_enrollment=False):
        if self.enrollment_locked and not cancel_enrollment:
            return False
        if purpose not in ('authenticate', 'enroll', 'preview', 'departure'):
            raise ValueError('잘못된 카메라 사용 목적')
        if purpose == 'enroll' and (not isinstance(enrollment_name, str)
                                  or not 1 <= len(enrollment_name.strip()) <= 40):
            raise ValueError('방향별 등록을 시작하려면 이름이 필요합니다.')
        self.purpose, self.enrollment_name = purpose, enrollment_name
        self.enrollment_requested_at = monotonic() if purpose == 'enroll' else None
        if session_id is not None:
            self.session_id = session_id
        self._reset_state()
        return True

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('인식 기준은 0~1 사이여야 합니다.')
        if value != self._threshold:
            if self.enrollment_locked:
                return
            self._threshold = value
            self.reset_standby()

    @property
    def save_interval(self):
        return self._save_interval

    @save_interval.setter
    def save_interval(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('저장 간격은 0 이상의 유한한 숫자여야 합니다.')
        self._save_interval = value

    def _result(self, profile, detections, status):
        self.progress(status)
        result = FrameResult([], False, status, self.saved_count, 0, 0, len(detections), 0,
            power_mode=profile.name, processing_size=profile.size, target_fps=profile.fps,
            detection_boxes=[tuple(float(v) for v in row[:4]) for row in detections],
            session_id=self.session_id)
        if self.challenge:
            result.direction = self.challenge.target
            result.arrow = ARROWS.get(result.direction, '')
            result.liveness_step = self.challenge.prompt
            if self.purpose == 'enroll':
                result.enrollment_count = len(self.challenge.samples)
        return result

    def _failure(self, profile, detections, message, score=None):
        self.terminal = True
        result = self._result(profile, detections, message)
        if self.purpose == 'enroll':
            result.enrollment_error = message
        else:
            result.ready = result.authentication_attempted = True
            result.match_score = score
        return result

    def _analyze(self, snapshot, detections):
        faces = self.engine.analyze(snapshot, detections)
        if len(faces) != 1 or faces[0].sharpness < 50 or min(faces[0].box[2:]) < 90:
            raise ValueError('얼굴이 작거나 흐립니다. 가까이에서 잠시 멈춰주세요.')
        return faces[0]

    def _snapshot(self, frame, detection, pose, now):
        # Freeze first; neither feature model ever reads a later camera frame.
        snapshot = frame.copy()
        face = self._analyze(snapshot, [detection])
        estimate = self.mesh.observe(snapshot, now)
        if estimate is None or overlap_ratio(tuple(detection[:4]), estimate.box) < .25:
            raise ValueError('촬영 사진의 Mesh와 YuNet 얼굴이 일치하지 않습니다.')
        size = snapshot.shape[1], snapshot.shape[0]
        geometry = normalized_mesh(estimate.landmarks, size)
        ok, jpeg = cv2.imencode('.jpg', crop_face(snapshot, detection[:4]))
        if not ok:
            raise ValueError('사진 인코딩 실패')
        index = sum(s.direction == self.challenge.target for s in self.challenge.samples)
        return face, CaptureSample(self.challenge.target, index, now, face.embedding.copy(),
            estimate.landmarks.copy(), geometry, pose, size, detection.copy(), jpeg.tobytes())

    def process(self, frame):
        profile, now = self.profile, monotonic()
        if self.terminal or not self.session_valid():
            return self._result(profile, [], '결과 처리 또는 새 세션 대기')
        if (self.purpose == 'enroll' and self.enrollment_requested_at is not None
                and now-self.enrollment_requested_at >= self.settings.enroll_timeout):
            return self._failure(profile, [], '등록 촬영 시간 초과 · 다시 시작하세요')
        if self.purpose == 'enroll' and (frame.shape[1], frame.shape[0]) != ENROLLMENT_PROFILE.size:
            raise RuntimeError('얼굴등록단계는 1920×1080 FHD 영상에서만 진행합니다.')
        # Keep the captured/retained photo FHD; YuNet maps its cheaper input back
        # to original FHD coordinates, including all five landmarks.
        detection_size = ACTIVE_PROFILE.size if self.purpose == 'enroll' else profile.size
        detections = self.engine.detect(frame, size=detection_size)
        if self.purpose == 'departure':
            return self._result(profile, detections, '화면에서 벗어나 주세요')
        if self.mode == 'standby':
            if len(detections) == 1:
                self.watcher.update(frame, detections)
            else:
                self.watcher.ready = False
                self.watcher.reset_hold()
                self.watcher.reason = '얼굴 대기 중' if not len(detections) else '얼굴은 한 명만 보여주세요'
            reason = self.watcher.reason
            if self.watcher.ready:
                self.mode = 'active'
                self.active_started = now
                self.watcher = FaceWatcher(replace(self.gate_settings, hold_seconds=0))
                reason = '촬영 준비 완료 · 활성 해상도로 전환'
            return self._result(profile, detections, reason)
        if len(detections) != 1:
            if self.challenge:
                return self._failure(profile, detections, '얼굴 소실 또는 여러 얼굴 · 다시 시작하세요')
            if self.purpose != 'enroll':
                self.reset_standby()
            return self._result(profile, detections, '얼굴은 한 명만 보여주세요')
        try:
            if self.active_started is None:
                self.active_started = now
            if self.challenge and self.challenge.timed_out(now):
                return self._failure(profile, detections, '방향 촬영 시간 초과 · 다시 시작하세요')
            if (self.challenge is None and self.purpose == 'authenticate'
                    and now-self.active_started >= self.settings.auth_timeout):
                return self._failure(profile, detections, '본인 확인 사진 품질 검사 시간 초과')
            if self.challenge is None:
                self.watcher.update(frame, detections)
                if not self.watcher.ready:
                    if self.purpose != 'enroll':
                        self.reset_standby()
                    return self._result(profile, detections, '정면에서 촬영 준비를 다시 확인합니다')
                if self.purpose == 'preview':
                    result = self._result(profile, detections, '등록 준비 완료 · 이름을 입력하고 등록하세요')
                    result.faces, result.ready = [self._analyze(frame, detections)], True
                    return result
                if self.purpose == 'authenticate':
                    self.store.refresh()
                    face = self._analyze(frame.copy(), detections)
                    if not self.store.people:
                        result = self._result(profile, detections, '등록된 사용자가 없습니다 · 사용자 등록 탭을 열어주세요')
                        result.faces, result.ready = [face], True
                        return result
                    name, score = authenticate_face(self.store, face.embedding, self.threshold)
                    if name is None:
                        return self._failure(profile, detections, '등록된 얼굴과 일치하지 않습니다', score)
                    if not self.store.has_directional_profile(name):
                        return self._failure(profile, detections, '정면·상·하·좌·우 3장씩 재등록이 필요합니다', score)
                    self.identity, self.reference_embedding = name, face.embedding.copy()
                # Five-point PnP has person/camera-dependent pitch bias. Calibrate at the
                # YuNet frontal gate; guide subsequent rotations relative to this pose.
                baseline = self.pose_estimator(detections[0], (frame.shape[1], frame.shape[0]))
                self.pose_origin = baseline.yaw, baseline.pitch
                # Prepare model once, before starting user-facing direction deadlines.
                if self.mesh is None:
                    self.mesh = FaceMeshEstimator(self.progress)
                    now = monotonic()
                self.challenge = GuidedCapture(self.purpose, self.settings, now)
            raw_pose = self.pose_estimator(detections[0], (frame.shape[1], frame.shape[0]))
            pose = Pose(raw_pose.yaw-self.pose_origin[0], raw_pose.pitch-self.pose_origin[1],
                        raw_pose.roll, raw_pose.box)
            due = self.challenge.candidate_due(pose, now)
            if self.challenge.failure:
                return self._failure(profile, detections, self.challenge.failure)
            if not due:
                result = self._result(profile, detections, self.challenge.prompt)
                result.yaw, result.pitch = pose.yaw, pose.pitch
                return result
            face, sample = self._snapshot(frame, detections[0], pose, now)
            if self.reference_embedding is not None and float(self.reference_embedding @ face.embedding) < self.threshold:
                return self._failure(profile, detections, '촬영 도중 얼굴이 변경되었습니다')
            if self.reference_embedding is None:
                self.reference_embedding = face.embedding.copy()
            proof = None
            if self.purpose == 'authenticate':
                proof = compare_sample(sample, self.store.directional_samples(self.identity, sample.direction),
                                       self.threshold, self.settings)
                if proof is None:
                    return self._failure(profile, detections, '해당 방향의 임베딩 또는 Mesh 형상 불일치')
            self.challenge.accept(sample, now)
            result = self._result(profile, detections, self.challenge.prompt)
            result.faces, result.yaw, result.pitch = [face], pose.yaw, pose.pitch
            if not self.challenge.complete:
                return result
            if not self.session_valid():
                self.terminal = True
                return self._result(profile, detections, '등록 또는 인증이 취소되었습니다')
            if self.purpose == 'enroll':
                self.store.enroll_directional(self.enrollment_name, self.challenge.samples, self.settings)
                result.enrollment_completed = self.enrollment_name
                result.enrollment_count = 15
                result.status = '5방향 15장 등록 완료 · 사진·Mesh·임베딩 DB 저장'
            else:
                proofs = self.store.record_directional_pair(self.identity, self.challenge.samples,
                                                          self.source, self.threshold, self.settings)
                if proofs is None:
                    return self._failure(profile, detections, '최종 DB 비교 실패 · 등록 정보 변경 또는 형상 불일치')
                result.ready = result.authentication_attempted = result.liveness_passed = True
                result.authenticated_name = self.identity
                result.match_score = min(p[0] for p in proofs)
                self.saved_count += 2
                result.saved_count, result.saved_this_frame = self.saved_count, 2
                result.status = '무작위 두 방향의 사진·임베딩·Mesh 비교 통과'
            self.terminal = True
            return result
        except (cv2.error, ValueError, IndexError) as exc:
            result = self._result(profile, detections, f'촬영 사진 확인: {exc}')
            result.embedding_errors = 1
            if self.challenge:
                self.challenge.hold_started, self.challenge.frames = None, 0
            return result

    def close(self):
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None
