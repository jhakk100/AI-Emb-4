"""Exclusive FHD enrollment. This stage has no authentication or door transitions."""

import math
from time import monotonic

import cv2

from capture_control import ACTIVE_PROFILE, ENROLLMENT_PROFILE
from database import initialize_database
from face_core import DEFAULT_THRESHOLD, REGISTRATION_SAMPLES
from guided_capture import ARROWS, LABELS, DIRECTIONS, MANUAL_POSE_BASIS, GuidedCapture, load_guided_settings
from liveness import FaceMeshEstimator
from photo_capture import PhotoCapture
from pipeline import FrameResult

FACE_GUIDE_RADII = (.16, .32)  # fractions of the displayed image width and height


def face_centered(box, size):
    x, y, w, h = box
    width, height = size
    rx, ry = FACE_GUIDE_RADII
    dx = ((x+w/2)/width-.5)/rx
    dy = ((y+h/2)/height-.5)/ry
    return dx*dx+dy*dy <= 1


class EnrollmentPipeline(PhotoCapture):
    purpose = 'enroll'
    mode = 'enrollment'
    profile = ENROLLMENT_PROFILE
    enrollment_locked = True

    def __init__(self, engine, enrollment_name, watcher=None, store=None, mesh=None,
                 progress=None, threshold=DEFAULT_THRESHOLD, settings=None,
                 pose_estimator=None, session_valid=None, session_id=0, source='camera:0'):
        if not isinstance(enrollment_name, str) or not 1 <= len(enrollment_name.strip()) <= 40:
            raise ValueError('방향별 등록을 시작하려면 이름이 필요합니다.')
        self.engine, self.mesh = engine, mesh
        self.enrollment_name = enrollment_name.strip()
        self.store = store if store is not None else initialize_database()
        self.saved_count = self.store.count_observations()
        self.progress = progress or (lambda text: None)
        self._threshold = float(threshold)
        if not math.isfinite(self._threshold) or not 0 <= self._threshold <= 1:
            raise ValueError('인식 기준은 0~1 사이여야 합니다.')
        self.settings = settings if settings is not None else load_guided_settings()
        self.session_valid = session_valid or (lambda: True)
        self.session_id = session_id
        self._start_attempt()

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        # An enrollment attempt keeps the settings it started with.
        pass

    def _start_attempt(self):
        self.challenge = GuidedCapture('enroll', self.settings, monotonic())
        self.reference_embedding = None
        self.terminal = self.completed = False
        self.error = None
        self.manual_request = None
        self.manual_status = None
        self.manual_captured = False

    @property
    def _prompt(self):
        direction = self.challenge.target
        count = sum(sample.direction == direction for sample in self.challenge.samples)
        return f'{ARROWS[direction]} {LABELS[direction]} · 자세를 맞추고 P · {count}/{self.challenge.per_direction}장'

    def request_manual_capture(self, session_id, direction, count, requested_at):
        if (self.terminal or not self.session_valid() or session_id != self.session_id or self.challenge is None
                or direction != self.challenge.target or count != len(self.challenge.samples)):
            self.manual_status = '수동 촬영 미접수 · 현재 촬영 방향과 진행 수를 확인하고 다시 누르세요.'
            return False
        if (not isinstance(requested_at, (int, float)) or isinstance(requested_at, bool)
                or not math.isfinite(requested_at) or not 0 <= monotonic()-requested_at <= .5):
            self.manual_status = '수동 촬영 미접수 · 요청이 오래되었습니다. 다시 눌러주세요.'
            return False
        if self.manual_request is not None:
            self.manual_status = '수동 촬영 요청을 이미 처리하고 있습니다.'
            return False
        self.manual_request = (direction, count, requested_at)
        self.manual_status = '촬영 접수 · 다음 영상을 확인합니다.'
        return True

    def _take_manual_request(self, now):
        request, self.manual_request = self.manual_request, None
        if request is None:
            return False
        direction, count, requested_at = request
        if (not 0 <= now-requested_at <= .5 or self.challenge is None or self.terminal
                or direction != self.challenge.target or count != len(self.challenge.samples)):
            self.manual_status = '수동 촬영 요청 만료/진행 변경 · 현재 자세에서 다시 눌러주세요.'
            return False
        return True

    def restart(self, session_id):
        if not self.terminal or self.completed:
            return False
        self.session_id = session_id
        self._start_attempt()
        return True

    def reset_standby(self):
        return False

    def set_purpose(self, *args, **kwargs):
        # Only CameraSession may destroy this stage and create a different one.
        return False

    def _result(self, detections, status):
        result = FrameResult([], False, status, self.saved_count, 0, 0,
            len(detections), 0, power_mode='enrollment', processing_size=self.profile.size,
            target_fps=self.profile.fps, session_id=self.session_id, purpose='enroll',
            detection_boxes=[tuple(float(v) for v in row[:4]) for row in detections])
        if self.challenge:
            result.direction = self.challenge.target
            result.arrow = ARROWS.get(result.direction, '')
            if result.direction is not None:
                result.liveness_step = f'{LABELS[result.direction]} · 가이드에 얼굴을 맞추고 P 키를 눌러주세요'
            result.enrollment_count = len(self.challenge.samples)
            result.grid_counts = {direction: sum(sample.direction == direction for sample in self.challenge.samples)
                                  for direction in DIRECTIONS}
        result.enrollment_error = self.error
        result.manual_capture_status = self.manual_status
        result.manual_capture_pending = self.manual_request is not None
        result.manual_captured = self.manual_captured
        if self.completed:
            result.enrollment_completed = self.enrollment_name
            result.grid_ready = True
            result.fused_ready = True
        return result

    def _failure(self, detections, message):
        self.terminal, self.error = True, message
        self.manual_request = None
        if self.manual_status is not None:
            self.manual_status = f'촬영 중단 · {message}'
        self.challenge = self.reference_embedding = None  # discard the entire incomplete attempt
        return self._result(detections, message)

    def process(self, frame):
        now = monotonic()
        self.manual_captured = False
        if self.terminal or not self.session_valid():
            self.manual_request = None
            return self._result([], self.error or '등록 결과 확인 대기')
        if (frame.shape[1], frame.shape[0]) != self.profile.size:
            raise RuntimeError('얼굴등록단계는 1920×1080 FHD 영상에서만 진행합니다.')
        if not self._take_manual_request(now):
            return self._result([], self._prompt)
        detections = []
        try:
            detections = self.engine.detect(frame, size=ACTIVE_PROFILE.size)
            if len(detections) != 1:
                raise ValueError('얼굴 한 명을 가이드 안에 보여주세요.')
            if not face_centered(detections[0][:4], self.profile.size):
                raise ValueError('얼굴 중심을 가이드 안에 맞춰주세요.')
            if self.mesh is None:
                self.mesh = FaceMeshEstimator(self.progress)
            face, sample = self._snapshot(frame, detections[0], None, now)
            if not self.session_valid():
                return self._result([], '등록 취소 확인 중')
            if self.reference_embedding is not None and float(self.reference_embedding @ face.embedding) < self.threshold:
                return self._failure(detections, '촬영 도중 얼굴이 변경되었습니다 · 다시 촬영을 눌러주세요')
            if self.reference_embedding is None:
                self.reference_embedding = face.embedding.copy()
            sample.capture_mode = 'manual'
            sample.pose_basis = MANUAL_POSE_BASIS.copy()
            self.challenge.accept(sample, monotonic())
            self.manual_captured = True
            self.manual_status = f'수동 촬영 완료 · {LABELS[sample.direction]} · 전체 {len(self.challenge.samples)}/{REGISTRATION_SAMPLES}장'
            if self.challenge.complete:
                if not self.session_valid():
                    return self._result([], '등록 취소 확인 중')
                try:
                    self.store.enroll_directional(self.enrollment_name, self.challenge.samples, self.settings)
                except ValueError as exc:
                    return self._failure(detections, f'등록 저장 검사 실패: {exc}')
                self.terminal = self.completed = True
            result = self._result(detections, '4방향 grid·통합 3D 모델·12장 저장 완료 · 나가기를 눌러주세요'
                                  if self.completed else self._prompt)
            result.faces = [face]
            result.mesh_landmarks = sample.landmarks.copy()
            return result
        except (cv2.error, ValueError, IndexError) as exc:
            self.manual_status = f'수동 촬영 거절 · {exc}'
            result = self._result(detections, f'촬영 사진 확인: {exc}')
            result.embedding_errors = 1
            return result
