"""YuNet guides five poses; only retained snapshots run Face Mesh and embedding."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import secrets

import cv2
import numpy as np

from face_core import normalize
from liveness import Pose
from models import ROOT
from standby import overlap_ratio

DIRECTIONS = ('front', 'up', 'down', 'left', 'right')
LABELS = {'front': '정면', 'up': '위', 'down': '아래', 'left': '화면 왼쪽', 'right': '화면 오른쪽'}
ARROWS = {'front': '●', 'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
MESH_MODEL = 'mediapipe-face-landmarker-v1-eye-scale-468-v1'
FACE_POINTS = np.array([[-32, -28, 20], [32, -28, 20], [0, 0, 0],
                        [-24, 28, 15], [24, 28, 15]], dtype=np.float64)


@dataclass(frozen=True)
class GuidedSettings:
    front_degrees: float = 12.
    yaw_min: float = 15.
    yaw_max: float = 35.
    pitch_min: float = 12.
    pitch_max: float = 30.
    cross_degrees: float = 12.
    roll_max: float = 15.
    hold_seconds: float = .25
    min_frames: int = 3
    sample_interval: float = .4
    max_gap: float = .7
    enroll_timeout: float = 120.
    enroll_step_timeout: float = 25.
    auth_timeout: float = 25.
    auth_step_timeout: float = 10.
    mesh_limit: float = .12
    depth_limit: float = .08
    yaw_sign: int = 1
    pitch_sign: int = 1

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f'잘못된 방향 촬영 설정: {f.name}')
        if not (0 < self.front_degrees < self.yaw_min < self.yaw_max < 60
                and 0 < self.pitch_min < self.pitch_max < 60 and 0 < self.cross_degrees < 30
                and 0 < self.roll_max < 45 and 0 < self.hold_seconds < self.max_gap
                and 0 < self.sample_interval < 5 and type(self.min_frames) is int and self.min_frames >= 2
                and 0 < self.enroll_step_timeout <= self.enroll_timeout <= 300
                and 0 < self.auth_step_timeout <= self.auth_timeout <= 120
                and 0 < self.mesh_limit < 1 and 0 < self.depth_limit < 1
                and self.yaw_sign in (-1, 1) and self.pitch_sign in (-1, 1)):
            raise ValueError('방향 촬영 설정의 범위가 잘못되었습니다.')


def load_guided_settings(path=ROOT / 'guided_capture.json'):
    try:
        return GuidedSettings(**json.loads(Path(path).read_text(encoding='utf-8-sig')))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f'방향 촬영 설정 오류: {exc}') from exc


def yunet_pose(detection, size):
    """Approximate head angles, not calibrated depth, from YuNet's five 2D landmarks."""
    row = np.asarray(detection, dtype=float)
    if row.shape != (15,) or not np.isfinite(row).all() or min(row[2:4]) <= 0:
        raise ValueError('YuNet 특징점 데이터 오류')
    eyes = sorted(row[4:8].reshape(2, 2).tolist())
    mouth = sorted(row[10:14].reshape(2, 2).tolist())
    image_points = np.array([*eyes, row[8:10], *mouth], dtype=np.float64)
    width, height = size
    focal = float(max(size))
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=float)
    ok, rvec, tvec = cv2.solvePnP(FACE_POINTS, image_points, camera, np.zeros(4),
                                flags=cv2.SOLVEPNP_SQPNP)
    if not ok or float(tvec[2, 0]) <= 0:
        raise ValueError('얼굴 방향을 계산할 수 없습니다.')
    rotation, _ = cv2.Rodrigues(rvec)
    yaw = -math.degrees(math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0])))
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return Pose(yaw, pitch, roll, tuple(row[:4]))


def direction_matches(direction, pose, s):
    yaw, pitch = pose.yaw * s.yaw_sign, pose.pitch * s.pitch_sign
    if not all(math.isfinite(v) for v in (yaw, pitch, pose.roll)) or abs(pose.roll) > s.roll_max:
        return False
    if direction == 'front':
        return abs(yaw) <= s.front_degrees and abs(pitch) <= s.front_degrees
    if direction in ('left', 'right'):
        signed = yaw if direction == 'right' else -yaw
        return s.yaw_min <= signed <= s.yaw_max and abs(pitch) <= s.cross_degrees
    if direction in ('up', 'down'):
        signed = pitch if direction == 'down' else -pitch
        return s.pitch_min <= signed <= s.pitch_max and abs(yaw) <= s.cross_degrees
    raise ValueError('알 수 없는 촬영 방향')


def normalized_mesh(landmarks, size):
    raw = np.asarray(landmarks, dtype=np.float32)
    if raw.shape != (478, 3) or not np.isfinite(raw).all():
        raise ValueError('478개 Mesh 좌표가 필요합니다.')
    width, height = size
    points = raw[:468] * np.array([width, height, width], dtype=np.float32)
    scale = float(np.linalg.norm(points[33] - points[263]))
    if scale < 1 or width <= 0 or height <= 0:
        raise ValueError('Mesh 눈 사이 거리 또는 영상 크기 오류')
    return (points - points.mean(axis=0)) / scale


def mesh_errors(query, template):
    a, b = np.asarray(query, dtype=float), np.asarray(template, dtype=float)
    if a.shape != (468, 3) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError('정규화 Mesh 형식 오류')
    # Rigid alignment only: no reflection, no fitted nonrigid depth/shape deformation.
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    residual = a @ (u @ correction @ vt) - b
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))), float(np.sqrt(np.mean(residual[:, 2] ** 2)))


@dataclass
class CaptureSample:
    direction: str
    index: int
    captured_at: float
    embedding: np.ndarray
    landmarks: np.ndarray
    mesh: np.ndarray
    pose: Pose
    size: tuple
    detection: np.ndarray
    crop_jpeg: bytes
    model: str = MESH_MODEL


def compare_sample(sample, templates, threshold, settings):
    if not direction_matches(sample.direction, sample.pose, settings):
        return None
    candidates = []
    for template in templates:
        if template['model'] != sample.model:
            continue
        score = float(np.clip(normalize(sample.embedding) @ normalize(template['embedding']), -1, 1))
        error, depth = mesh_errors(sample.mesh, template['mesh'])
        if score >= threshold and error <= settings.mesh_limit and depth <= settings.depth_limit:
            candidates.append((score, error, depth))
    return max(candidates, default=None, key=lambda item: item[0])


class GuidedCapture:
    def __init__(self, purpose, settings, now, directions=None):
        self.purpose, self.settings = purpose, settings
        self.directions = tuple(directions or (DIRECTIONS if purpose == 'enroll' else
                                 secrets.SystemRandom().sample(DIRECTIONS, 2)))
        self.per_direction = 3 if purpose == 'enroll' else 1
        self.samples = []
        self.index = 0
        self.started = self.step_started = now
        self.last_seen = self.previous_box = self.hold_started = None
        self.last_capture = -float('inf')
        self.frames = 0
        self.failure = None
        self.complete = False

    @property
    def target(self):
        return None if self.complete else self.directions[self.index]

    @property
    def prompt(self):
        if self.failure:
            return self.failure
        if self.complete:
            return '5방향 등록 완료' if self.purpose == 'enroll' else '두 방향 사진 검사 완료'
        count = sum(sample.direction == self.target for sample in self.samples)
        action = ('정면을 바라봐주세요' if self.target == 'front' else
                  f'고개를 {LABELS[self.target]} 방향으로 돌려주세요')
        return f'{ARROWS[self.target]} {action} · {count}/{self.per_direction}장 · 전체 {len(self.samples)}/{len(self.directions)*self.per_direction}'

    def fail(self, message):
        self.failure = message
        return False

    def candidate_due(self, pose, now):
        s = self.settings
        if self.failure or self.complete:
            return False
        if self.timed_out(now):
            return self.fail('방향 촬영 시간 초과 · 다시 시작하세요')
        if pose is None:
            return self.fail('얼굴 소실 또는 여러 얼굴 · 다시 시작하세요')
        if self.last_seen is not None and (now <= self.last_seen or now-self.last_seen > s.max_gap):
            return self.fail('카메라 추적이 끊겼습니다 · 다시 시작하세요')
        if self.previous_box is not None and overlap_ratio(self.previous_box, pose.box) < .25:
            return self.fail('얼굴 위치가 급변했습니다 · 다시 시작하세요')
        self.last_seen, self.previous_box = now, pose.box
        if not direction_matches(self.target, pose, s):
            self.hold_started, self.frames = None, 0
            return False
        if self.hold_started is None:
            self.hold_started = now
        self.frames += 1
        return (self.frames >= s.min_frames and now-self.hold_started >= s.hold_seconds
                and now-self.last_capture >= s.sample_interval)

    def timed_out(self, now):
        s = self.settings
        timeout = s.enroll_timeout if self.purpose == 'enroll' else s.auth_timeout
        step = s.enroll_step_timeout if self.purpose == 'enroll' else s.auth_step_timeout
        return now-self.started >= timeout or now-self.step_started >= step

    def accept(self, sample, now):
        if self.failure or self.complete or sample.direction != self.target:
            raise ValueError('현재 요청과 다른 방향의 샘플입니다.')
        expected = sum(s.direction == self.target for s in self.samples)
        if sample.index != expected:
            raise ValueError('샘플 순서 오류')
        self.samples.append(sample)
        self.last_capture = now
        self.hold_started, self.frames = None, 0
        if expected + 1 == self.per_direction:
            self.index += 1
            self.step_started = now
            self.complete = self.index == len(self.directions)
