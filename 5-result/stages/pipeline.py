"""Authentication, preview and departure only. Enrollment has its own processor."""

from dataclasses import dataclass, replace
import math
from time import monotonic, perf_counter

import cv2

from capture_control import ACTIVE_PROFILE, STANDBY_PROFILE, CALIBRATED_ACTIVE_PROFILE
from database import initialize_database
from face_grid import MeshPoseGuide
from face_core import DEFAULT_THRESHOLD
from guided_capture import (ARROWS, DIRECTIONS, GuidedCapture, compare_sample, direction_matches,
                            load_guided_settings, mesh_errors, normalized_mesh, yunet_pose)
from liveness import FaceMeshEstimator, Pose
from standby import CHECK_FPS, DETECT_SIZE, FaceWatcher, overlap_ratio
from photo_capture import PhotoCapture
from pose_defaults import DefaultPoseGuide, load_default_preset


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
    purpose: str = 'authenticate'
    mesh_landmarks: object | None = None
    grid_counts: dict | None = None
    grid_ready: bool = False
    pose_preset: str | None = None
    fused_ready: bool = False
    manual_capture_status: str | None = None
    manual_capture_pending: bool = False
    manual_captured: bool = False
    diagnostic_only: bool = False
    pose_hint: str | None = None
    auth_attempt_id: int | None = None
    front_threshold: float | None = None


class PoseHintDisplay:
    """Debounce guidance from fresh pose measurements without affecting capture."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.visible = self._candidate = None
        self._candidate_since = self._visible_since = None

    def update(self, hint, now):
        if hint == self.visible:
            self._candidate_since = None
            return
        if self._candidate_since is None or hint != self._candidate:
            self._candidate = hint
            self._candidate_since = now
            return
        delay = .10 if self.visible is None else .35 if hint is None else .25
        if now-self._candidate_since < delay:
            return
        if self._visible_since is not None and now-self._visible_since < .60:
            return
        self.visible = hint
        self._visible_since = now if hint is not None else None
        self._candidate_since = None


class FacePipeline(PhotoCapture):
    def __init__(self, engine, watcher=None, store=None, source='camera:0',
                 save_interval=1., progress=None, threshold=DEFAULT_THRESHOLD,
                 start_in_standby=True, mesh=None, settings=None, purpose='authenticate',
                 enrollment_name=None, pose_estimator=yunet_pose, session_valid=None, adaptive_front=True):
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
        preset = load_default_preset() if pose_estimator is yunet_pose else None
        self.pose_guide = DefaultPoseGuide(preset, source) if preset else None
        self.session_valid = session_valid or (lambda: True)
        self.adaptive_front = adaptive_front
        self.auth_attempt = None
        self.auth_trace = None
        self.session_id = 0
        self.set_purpose(purpose, enrollment_name=enrollment_name)
        self.mode = 'standby' if start_in_standby else 'active'

    @property
    def profile(self):
        if self.mode != 'standby' and self.purpose == 'authenticate':
            if self.manual_names is None:
                self.store.refresh()
                self.manual_names = self.store.manual_auth_names()
            if self.manual_names or self.pose_guide is not None:
                return CALIBRATED_ACTIVE_PROFILE
        return STANDBY_PROFILE if self.mode == 'standby' else ACTIVE_PROFILE

    def _reset_state(self):
        if self.pose_guide is not None:
            self.pose_guide.reset()
        self.mode = 'standby'
        self.watcher = FaceWatcher(self.gate_settings)
        self.challenge = None
        self.identity = self.reference_embedding = None
        self.terminal = False
        self.active_started = None
        self.pose_origin = None
        self.mesh_pose_guide = None
        self.manual_names = None
        self.last_mesh_frame = -math.inf
        self.front_check_status = None
        self.hint_display = PoseHintDisplay()
        self.next_standby_check = -math.inf
        self.standby_detections = []

    def reset_standby(self):
        if self.auth_attempt is not None and not self.auth_attempt.diagnostic:
            self.finish_auth_attempt('interrupted', '대기 상태로 돌아감')
        self._reset_state()
        return True

    def set_purpose(self, purpose, session_id=None, enrollment_name=None, cancel_enrollment=False):
        if purpose not in ('authenticate', 'preview', 'departure'):
            raise ValueError('잘못된 카메라 사용 목적')
        self.finish_auth_attempt('interrupted', '카메라 사용 단계 변경')
        self.purpose, self.enrollment_name = purpose, enrollment_name
        if session_id is not None:
            self.session_id = session_id
        self._reset_state()
        return True

    @property
    def threshold(self):
        return self._threshold

    @property
    def front_threshold(self):
        return self.auth_attempt.threshold if self.auth_attempt else self.threshold

    def begin_auth_attempt(self, diagnostic=False):
        if self.purpose == 'authenticate' and self.auth_attempt is None:
            self.auth_attempt = self.store.calibration.begin(self.source, self.threshold, diagnostic,
                                                             self.adaptive_front)

    def finish_auth_attempt(self, outcome, reason, matched_name=None):
        if self.auth_attempt is not None:
            self.store.calibration.finish(self.auth_attempt, outcome, reason, matched_name)
            self.auth_attempt = None

    def close(self):
        try:
            self.finish_auth_attempt('interrupted', '카메라 세션 종료')
        finally:
            super().close()

    def _record_front_scores(self, embedding):
        if self.auth_attempt is not None:
            self.store.calibration.record_scores(self.auth_attempt, embedding)

    @threshold.setter
    def threshold(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('인식 기준은 0~1 사이여야 합니다.')
        if value != self._threshold:
            self.finish_auth_attempt('interrupted', '유사도 기준 변경')
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
            session_id=self.session_id, purpose=self.purpose,
            pose_preset=(self.pose_guide.preset.name
                         if self.pose_guide is not None and self.mesh_pose_guide is None else None))
        if self.auth_attempt is not None:
            result.auth_attempt_id = self.auth_attempt.id
            result.front_threshold = self.auth_attempt.threshold
        if self.challenge:
            result.direction = self.challenge.target
            result.arrow = ARROWS.get(result.direction, '')
            result.liveness_step = self.challenge.prompt
        return result

    def _failure(self, profile, detections, message, score=None):
        self.hint_display.reset()
        self.terminal = True
        result = self._result(profile, detections, message)
        result.ready = result.authentication_attempted = True
        result.match_score = score
        self.finish_auth_attempt('rejected', message)
        return result

    def _trace(self, event, **fields):
        if self.auth_trace is not None:
            self.auth_trace(event, **fields)

    def _photo_mesh(self, snapshot, detection, now):
        if self.auth_trace is None:
            return super()._photo_mesh(snapshot, detection, now)
        started = perf_counter()
        passed = False
        try:
            estimate = super()._photo_mesh(snapshot, detection, now)
            passed = True
            return estimate
        finally:
            self._trace('mesh_inference', ms=round((perf_counter()-started)*1000, 2), passed=passed,
                        direction=self.challenge.target if self.challenge else 'front')

    def _begin_challenge_trace(self):
        guide = self.mesh_pose_guide or self.pose_guide
        profile = (guide.profile if self.mesh_pose_guide is not None else
                   guide.preset.profile if guide is not None else None)
        self._trace('challenge', directions=self.challenge.directions,
                    pose_source='mesh' if self.mesh_pose_guide else 'yunet',
                    centers=dict(zip(profile.directions, profile.centers.tolist())) if profile else None,
                    radii=dict(zip(profile.directions, profile.radii.tolist())) if profile else None,
                    scale=profile.scale.tolist() if profile else None,
                    pitch_scales=dict(zip(profile.directions, profile.pitch_scales.tolist())) if profile else None,
                    mesh_neutral=self.mesh_pose_guide.neutral if self.mesh_pose_guide else None)

    def _pose_due(self, pose, now):
        challenge = self.challenge
        if self.auth_trace is not None:
            old_start, old_frames = challenge.hold_started, challenge.frames
            last_seen, previous_box = challenge.last_seen, challenge.previous_box
        due = challenge.candidate_due(pose, now)
        raw_hint = (self.mesh_pose_guide.correction_hint(challenge.target, pose)
                    if self.mesh_pose_guide is not None and not challenge.failure and not challenge.frames else None)
        if due or challenge.failure:
            self.hint_display.reset()
        else:
            self.hint_display.update(raw_hint, now)
        if self.auth_trace is None:
            return due
        profile = (self.mesh_pose_guide.profile if self.mesh_pose_guide is not None else
                   self.pose_guide.preset.profile if self.pose_guide is not None else None)
        target_ratio, nearest, detected = None, None, None
        angles = [float(pose.yaw), float(pose.pitch), float(pose.roll)]
        if profile and all(math.isfinite(angle) for angle in angles):
            nearest_distances, acceptance_distances = profile.distances(pose)
            nearest = profile.directions[int(nearest_distances.argmin())]
            target = profile.directions.index(challenge.target)
            target_ratio = float(acceptance_distances[target]/profile.radii[target])
            detected, _ = profile.classify(pose)
        elif all(math.isfinite(angle) for angle in angles):
            detected = next((direction for direction in DIRECTIONS if direction_matches(direction, pose, self.settings)), None)
        block = (challenge.failure or ('capture' if due else
                 'direction' if challenge.frames == 0 else
                 'hold' if challenge.frames < self.settings.min_frames or now-challenge.hold_started < self.settings.hold_seconds
                 else 'sample_interval'))
        self._trace('pose_check', requested=challenge.target, detected=detected, nearest=nearest,
                    angles=[angle if math.isfinite(angle) else None for angle in angles],
                    target_distance_ratio=target_ratio, block=block, hold_frames=challenge.frames,
                    pose_hint=self.hint_display.visible, raw_pose_hint=raw_hint,
                    hold_ms=round((now-challenge.hold_started)*1000, 2) if challenge.hold_started is not None else 0,
                    reset=old_frames > 0 and not challenge.frames,
                    previous_hold_ms=round((now-old_start)*1000, 2) if old_start is not None else 0,
                    frame_gap_ms=round((now-last_seen)*1000, 2) if last_seen is not None else None,
                    box_overlap=float(overlap_ratio(previous_box, pose.box)) if previous_box is not None else None)
        return due

    def _start_mesh_auth(self, profile, frame, detections, now):
        if self.mesh is None:
            started = perf_counter()
            self.mesh = FaceMeshEstimator(self.progress)
            self._trace('mesh_load', ms=round((perf_counter()-started)*1000, 2))
        row = detections[0]
        snapshot = frame.copy()
        face = self._analyze(snapshot, detections)
        self._record_front_scores(face.embedding)
        estimate = self._photo_mesh(snapshot, row, now)
        mesh = normalized_mesh(estimate.landmarks, (frame.shape[1], frame.shape[0]))
        candidates = []
        best_hint, best_hint_score = None, -math.inf
        best_front_score = -math.inf
        self.front_check_status = '정면 얼굴이 등록된 Mesh·임베딩과 일치하지 않습니다.'
        for index, name in enumerate(sorted(self.manual_names)):
            grids = self.store.face_grids(name)
            front = grids.get('front')
            if front is None:
                continue
            score = float(face.embedding @ front['embedding'])
            best_front_score = max(best_front_score, score)
            error, depth = mesh_errors(mesh, front['vertices'])
            self._trace('front_compare', candidate=index, score=score, shape_error=error, depth_error=depth)
            if (score < self.front_threshold or error > self.settings.mesh_limit or depth > self.settings.depth_limit
                    or not self.store.has_directional_profile(name) or not self.store.has_grid_profile(name)
                    or not self.store.has_fused_model(name) or not self.store.grid_basis_matches(name, None)):
                continue
            try:
                guide = MeshPoseGuide(grids, self.settings)
            except ValueError:
                self.front_check_status = '등록 Mesh의 방향 차이가 부족합니다. 방향을 구분해 다시 등록하세요.'
                continue
            pose = guide.pose(mesh, face.box)
            classified, _ = guide.profile.classify(pose)
            hint = guide.correction_hint('front', pose) if classified != 'front' else None
            self._trace('front_pose', candidate=index, angles=[float(pose.yaw), float(pose.pitch), float(pose.roll)],
                        classified=classified, raw_pose_hint=hint)
            if classified == 'front':
                candidates.append((score, name, guide, pose))
            else:
                self.front_check_status = '정면 Mesh를 확인 중입니다. 카메라를 정면으로 봐주세요.'
                if hint and score > best_hint_score:
                    best_hint, best_hint_score = hint, score
        if not candidates:
            if best_front_score < self.front_threshold and math.isfinite(best_front_score):
                self.front_check_status = '정면 유사도가 현재 기준보다 낮습니다.'
            self.hint_display.update(best_hint, now)
            return None
        self.hint_display.reset()
        _, self.identity, self.mesh_pose_guide, pose = max(candidates, key=lambda candidate: candidate[0])
        self.mesh_pose_guide.set_neutral(pose)
        self.reference_embedding = face.embedding.copy()
        self.last_mesh_frame = now
        self.challenge = GuidedCapture('authenticate', self.settings, monotonic(), matcher=self.mesh_pose_guide.matches)
        if self.auth_trace:
            self._begin_challenge_trace()
        result = self._result(profile, detections, self.challenge.prompt)
        result.faces = [face]
        result.mesh_landmarks = estimate.landmarks
        return result

    def _continue_mesh_auth(self, profile, frame, detections, now):
        # Keep standby cheap and bound active Mesh inference during a direction request.
        if now-self.last_mesh_frame < .10:
            result = self._result(profile, detections, self.challenge.prompt)
            result.pose_hint = self.hint_display.visible
            return result
        self.last_mesh_frame = now
        row = detections[0]
        estimate = self._photo_mesh(frame, row, now)
        mesh = normalized_mesh(estimate.landmarks, (frame.shape[1], frame.shape[0]))
        pose = self.mesh_pose_guide.pose(mesh, tuple(row[:4]))
        due = self._pose_due(pose, now)
        if self.challenge.failure:
            return self._failure(profile, detections, self.challenge.failure)
        if due:
            face, sample = self._snapshot(frame, row, pose, now, mesh_estimate=estimate)
            sample.pose_basis = self.mesh_pose_guide.basis
            return self._confirm_photo(profile, detections, face, sample, monotonic())
        result = self._result(profile, detections, self.challenge.prompt)
        result.yaw, result.pitch = pose.yaw, pose.pitch
        result.mesh_landmarks = estimate.landmarks
        result.pose_hint = self.hint_display.visible
        return result

    def _confirm_photo(self, profile, detections, face, sample, now):
        started = perf_counter()
        score = float(self.reference_embedding @ face.embedding)
        trace = None
        if self.auth_trace:
            def trace(stage, **fields):
                self._trace('comparison', check=stage, direction=sample.direction, **fields)
            trace('identity', score=score, passed=score >= self.threshold)
        if score < self.threshold:
            return self._failure(profile, detections, '촬영 도중 얼굴이 변경되었습니다')
        proof = compare_sample(sample, self.store.directional_samples(self.identity, sample.direction),
                               self.threshold, self.settings, trace)
        if proof is None:
            return self._failure(profile, detections, '해당 방향의 임베딩 또는 Mesh 형상 불일치')
        if self.store.compare_face_grid(self.identity, sample, self.threshold, self.settings, trace) is None:
            return self._failure(profile, detections, '해당 방향의 얼굴 grid 형상/상대 깊이 불일치')
        if self.store.compare_face_model(self.identity, sample, self.threshold, self.settings, trace) is None:
            return self._failure(profile, detections, '통합 3D 얼굴 모델의 형상/상대 깊이 불일치')
        finished = max(now, monotonic())
        self.challenge.accept(sample, finished)
        self._trace('photo_accepted', direction=sample.direction, count=len(self.challenge.samples),
                    comparison_ms=round((perf_counter()-started)*1000, 2))
        self.challenge.last_seen = finished
        if self.mesh_pose_guide is None and self.pose_guide is not None:
            self.pose_guide.after_photo(finished)
        result = self._result(profile, detections, self.challenge.prompt)
        result.faces, result.yaw, result.pitch = [face], sample.pose.yaw, sample.pose.pitch
        result.mesh_landmarks = sample.landmarks
        if not self.challenge.complete:
            return result
        if not self.session_valid():
            self.terminal = True
            self.finish_auth_attempt('interrupted', '인증이 취소되었습니다')
            return self._result(profile, detections, '인증이 취소되었습니다')
        started = perf_counter()
        proofs = self.store.record_directional_pair(self.identity, self.challenge.samples,
                                                    self.source, self.threshold, self.settings)
        self._trace('db_confirmation', passed=proofs is not None, ms=round((perf_counter()-started)*1000, 2))
        if proofs is None:
            return self._failure(profile, detections, '최종 DB 비교 실패 · 등록 정보 변경 또는 형상 불일치')
        result.ready = result.authentication_attempted = result.liveness_passed = True
        result.authenticated_name = self.identity
        result.grid_ready = result.fused_ready = True
        result.match_score = min(proof[0] for proof in proofs)
        self.saved_count += 2
        result.saved_count, result.saved_this_frame = self.saved_count, 2
        result.status = '두 방향의 Face Mesh·임베딩·통합 3D 모델 비교 통과'
        self.terminal = True
        self.finish_auth_attempt('accepted', result.status, self.identity)
        return result

    def process(self, frame):
        profile, now = self.profile, monotonic()
        if self.terminal or not self.session_valid():
            return self._result(profile, [], '결과 처리 또는 새 세션 대기')
        if self.mode != 'standby' and self.purpose == 'authenticate':
            self.begin_auth_attempt()
        if profile == CALIBRATED_ACTIVE_PROFILE and (frame.shape[1], frame.shape[0]) != profile.size:
            raise RuntimeError('등록 영상과 같은 FHD 인증 영상이 필요합니다.')
        if self.mode == 'standby':
            if now >= self.next_standby_check:
                self.next_standby_check = now + 1 / CHECK_FPS
                self.standby_detections = self.engine.detect(frame, size=DETECT_SIZE)
                if self.purpose != 'departure':
                    if len(self.standby_detections) == 1:
                        self.watcher.update(frame, self.standby_detections)
                    else:
                        self.watcher.ready = False
                        self.watcher.reset_hold()
                        self.watcher.reason = ('얼굴은 한 명만 보여주세요' if len(self.standby_detections)
                                               else '얼굴 대기 중')
                    if self.auth_trace:
                        single_face = len(self.standby_detections) == 1
                        self._trace('standby_check', reason=self.watcher.reason, ready=self.watcher.ready,
                                    faces=len(self.standby_detections),
                                    confidence=self.watcher.score if single_face else None,
                                    face_ratio=self.watcher.face_ratio if single_face else None,
                                    front_error=self.watcher.front_error if single_face else None,
                                    hold_ms=round(self.watcher.hold_elapsed*1000, 2))
            reason = '화면에서 벗어나 주세요' if self.purpose == 'departure' else self.watcher.reason
            if self.watcher.ready and self.purpose != 'departure':
                self.mode = 'active'
                self.active_started = now
                if self.purpose == 'authenticate':
                    self.begin_auth_attempt()
                self.watcher = FaceWatcher(replace(self.gate_settings, hold_seconds=0))
                reason = '촬영 준비 완료 · 얼굴 확인 시작'
            return self._result(profile, self.standby_detections, reason)
        detection_size = ACTIVE_PROFILE.size
        detections = self.engine.detect(frame, size=detection_size)
        if len(detections) != 1:
            if self.challenge:
                return self._failure(profile, detections, '얼굴 소실 또는 여러 얼굴 · 다시 시작하세요')
            if self.auth_attempt is not None and not self.auth_attempt.diagnostic:
                self.finish_auth_attempt('interrupted', '얼굴 소실 또는 여러 얼굴')
            self.reset_standby()
            return self._result(profile, detections, '얼굴은 한 명만 보여주세요')
        try:
            if self.active_started is None:
                self.active_started = now
            if self.challenge and self.challenge.timed_out(now):
                return self._failure(profile, detections, '방향 촬영 시간 초과 · 다시 시작하세요')
            if (self.challenge is None and self.purpose == 'authenticate'
                    and now-self.active_started >= self.settings.auth_timeout):
                return self._failure(profile, detections, self.front_check_status or '본인 확인 사진 품질 검사 시간 초과')
            if self.mesh_pose_guide is not None:
                return self._continue_mesh_auth(profile, frame, detections, now)
            if self.purpose == 'authenticate' and self.challenge is None:
                if self.manual_names and now-self.last_mesh_frame >= .10:
                    self.last_mesh_frame = now
                    result = self._start_mesh_auth(profile, frame, detections, now)
                    if result is not None:
                        return result
                if self.manual_names and self.store.people.keys() <= self.manual_names:
                    result = self._result(profile, detections, self.front_check_status or '정면 Face Mesh 확인 중')
                    result.pose_hint = self.hint_display.visible
                    return result
                self.hint_display.reset()
            use_preset = self.pose_guide is not None and self.purpose == 'authenticate'
            raw_pose = self.pose_guide.observe(detections[0], profile.size, now) if use_preset else None
            if use_preset and raw_pose is None:
                self._trace('pose_unavailable', reason='yunet_stabilizing',
                            interrupted=bool(self.challenge and self.challenge.frames))
                if self.challenge:
                    self.challenge.hold_started, self.challenge.frames = None, 0
                return self._result(profile, detections, '5점 좌표 안정화 중 · 인증 자세 확인 대기')
            if self.challenge is None:
                self.watcher.update(frame, detections)
                if not self.watcher.ready:
                    self.reset_standby()
                    return self._result(profile, detections, '정면에서 촬영 준비를 다시 확인합니다')
                if self.purpose == 'preview':
                    result = self._result(profile, detections, '등록 준비 완료 · 이름을 입력하고 등록하세요')
                    result.faces, result.ready = [self._analyze(frame, detections)], True
                    return result
                if self.purpose == 'authenticate':
                    face = self._analyze(frame.copy(), detections)
                    self._record_front_scores(face.embedding)
                    if not self.store.people:
                        result = self._result(profile, detections, '등록된 사용자가 없습니다 · 사용자 등록 탭을 열어주세요')
                        result.faces, result.ready = [face], True
                        return result
                    name, score = self.store.match(face.embedding, self.front_threshold, exclude=self.manual_names)
                    if name is None:
                        return self._failure(profile, detections, '등록된 얼굴과 일치하지 않습니다', score)
                    basis = self.pose_guide.preset.basis if use_preset else None
                    if (not self.store.has_directional_profile(name) or not self.store.has_grid_profile(name)
                            or not self.store.has_fused_model(name) or not self.store.grid_basis_matches(name, basis)):
                        return self._failure(profile, detections, '4방향 얼굴 grid를 위해 정면·위·좌·우 3장씩 재등록이 필요합니다', score)
                    self.identity, self.reference_embedding = name, face.embedding.copy()
                # Five-point PnP has person/camera-dependent pitch bias. Calibrate at the
                # YuNet frontal gate; guide subsequent rotations relative to this pose.
                baseline = raw_pose if use_preset else self.pose_estimator(detections[0], (frame.shape[1], frame.shape[0]))
                self.pose_origin = baseline.yaw, baseline.pitch
                if use_preset:
                    self.pose_guide.calibrate_neutral(baseline)
                # Prepare model once, before starting user-facing direction deadlines.
                if self.mesh is None:
                    self.mesh = FaceMeshEstimator(self.progress)
                    now = monotonic()
                self.challenge = (GuidedCapture(self.purpose, self.settings, now, matcher=self.pose_guide.matches)
                                  if use_preset else GuidedCapture(self.purpose, self.settings, now))
                if self.auth_trace:
                    self._begin_challenge_trace()
            if not use_preset:
                raw_pose = self.pose_estimator(detections[0], (frame.shape[1], frame.shape[0]))
            pose = (self.pose_guide.align(raw_pose) if use_preset else
                    Pose(raw_pose.yaw-self.pose_origin[0], raw_pose.pitch-self.pose_origin[1], raw_pose.roll, raw_pose.box))
            due = self._pose_due(pose, now)
            if self.challenge.failure:
                return self._failure(profile, detections, self.challenge.failure)
            if not due:
                result = self._result(profile, detections, self.challenge.prompt)
                result.yaw, result.pitch = pose.yaw, pose.pitch
                return result
            face, sample = self._snapshot(frame, detections[0], pose, now)
            return self._confirm_photo(profile, detections, face, sample, now)
        except (cv2.error, ValueError, IndexError) as exc:
            self.hint_display.update(None, now)
            self._trace('photo_error', reason=str(exc), direction=self.challenge.target if self.challenge else 'front',
                        interrupted=bool(self.challenge and self.challenge.frames))
            result = self._result(profile, detections, f'촬영 사진 확인: {exc}')
            result.pose_hint = self.hint_display.visible
            result.embedding_errors = 1
            if self.challenge:
                self.challenge.hold_started, self.challenge.frames = None, 0
            return result
