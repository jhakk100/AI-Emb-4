"""Four user-labelled poses, with read compatibility for existing five-pose calibrations."""

from collections import deque
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ORDER = ('front', 'left', 'right', 'up')
LEGACY_ORDER = (*ORDER, 'down')
MAX_PROFILE_ROLL = 15.


class PoseProfile:
    def __init__(self, centers, *, pitch_scales=None):
        self.centers = np.asarray(centers, dtype=float).copy()
        if (self.centers.shape not in ((4, 3), (5, 3)) or not np.isfinite(self.centers).all()
                or np.abs(self.centers).max() > 180):
            raise ValueError('정면·좌·우·위의 유효한 기준 4개가 필요합니다.')
        self.directions = ORDER if len(self.centers) == len(ORDER) else LEGACY_ORDER
        pitch_scales = pitch_scales or {}
        self.pitch_scales = np.array([pitch_scales.get(direction, 1.) for direction in self.directions])
        positions = self.centers[:, :2]
        distances = np.linalg.norm(positions[:, None]-positions[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < 8:
            raise ValueError('자세 기준이 너무 비슷합니다. 고개를 더 돌려서 다시 기록하세요.')
        if np.linalg.svd(positions-positions.mean(axis=0), compute_uv=False)[-1] < 4:
            raise ValueError('좌우와 상하의 차이가 부족합니다. 상하 자세를 더 분명히 기록하세요.')
        self.scale = np.maximum(np.ptp(positions, axis=0), 8.)
        self.normalized = (positions-positions[0])/self.scale
        gaps = np.linalg.norm(self.normalized[:, None]-self.normalized[None, :], axis=2)
        np.fill_diagonal(gaps, np.inf)
        # Each direction gets its own radius, accounting for left/right asymmetry.
        self.radii = np.minimum(.35, gaps.min(axis=1)*.4)

    def distances(self, pose):
        query = (np.array([pose.yaw, pose.pitch])-self.centers[0, :2])/self.scale
        offsets = self.normalized-query
        nearest_distances = np.linalg.norm(offsets, axis=1)
        # Keep direction boundaries fixed. Stretch only the acceptance region along pitch.
        acceptance_distances = np.where(self.pitch_scales == 1., nearest_distances,
                                        np.hypot(offsets[:, 0], offsets[:, 1]/self.pitch_scales))
        return nearest_distances, acceptance_distances

    def classify(self, pose):
        if pose is None or not np.isfinite([pose.yaw, pose.pitch, pose.roll]).all():
            return None, None
        if abs(pose.roll-self.centers[0, 2]) > MAX_PROFILE_ROLL:
            return None, None
        nearest_distances, acceptance_distances = self.distances(pose)
        index = int(np.argmin(nearest_distances))
        ratio = float(acceptance_distances[index]/self.radii[index])
        # Retain legacy normalization/boundaries without accepting its down pose.
        direction = self.directions[index]
        return direction if direction in ORDER and acceptance_distances[index] <= self.radii[index] else None, ratio

    def save(self, path, context):
        path = Path(path)
        payload = dict(version=1, directions=list(self.directions), centers=self.centers.tolist(), context=context)
        fd, temporary = tempfile.mkstemp(prefix='.pose-calibration-', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path, context):
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if (not isinstance(payload, dict) or payload.get('version') != 1
                or payload.get('directions') not in (list(ORDER), list(LEGACY_ORDER))
                or payload.get('context') != context):
            raise ValueError('카메라·해상도·좌표 보정 설정이 다릅니다. 보정을 다시 시작하세요.')
        profile = cls(payload['centers'])
        if list(profile.directions) != payload['directions']:
            raise ValueError('보정 방향 수와 기준 각도 수가 다릅니다.')
        return profile


class PoseCalibration:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = 'idle'
        self.records = {}
        self.profile = None
        self.recent = deque(maxlen=15)
        self.index = 0
        self.results = []
        self._reset_hold()

    def _reset_hold(self):
        self.hold_started = self.last_test_seen = None
        self.hold_frames = 0

    @property
    def target(self):
        return ORDER[self.index] if self.mode in ('collect', 'test') else None

    def start(self):
        self.reset()
        self.mode = 'collect'

    def observe(self, pose, now, max_gap=.7):
        if pose is None or not np.isfinite([pose.yaw, pose.pitch, pose.roll, now]).all():
            self.recent.clear()
            self._reset_hold()
            return
        if self.recent and not 0 < now-self.recent[-1][0] <= max_gap:
            self.recent.clear()
            self._reset_hold()
        self.recent.append((now, pose))

    def capture(self, now):
        if self.mode != 'collect':
            raise ValueError('보정 시작을 누른 후 요청 자세에서 P 키를 누르세요.')
        samples = list(self.recent)[-5:]
        if (len(samples) < 5 or not 0 <= now-samples[-1][0] <= .5
                or samples[-1][0]-samples[0][0] < .1):
            raise ValueError('최근 얼굴 영상 5프레임이 필요합니다. 자세를 유지한 후 P 키를 다시 누르세요.')
        values = np.array([[p.yaw, p.pitch, p.roll] for _, p in samples])
        if np.ptp(values, axis=0).max() > 10:
            raise ValueError('자세가 흔들리고 있습니다. 잠시 멈춘 뒤 P 키를 다시 누르세요.')
        center = np.median(values, axis=0)
        if self.records:
            for previous in self.records.values():
                if np.linalg.norm(center[:2]-previous[:2]) < 8:
                    raise ValueError('이전 자세와 너무 비슷합니다. 요청 방향으로 더 돌린 후 P 키를 누르세요.')
            if abs(center[2]-self.records['front'][2]) > 15:
                raise ValueError('고개를 옆으로 기울이지 말고 요청 방향을 바라보세요.')
        records = dict(self.records, **{self.target: center})
        profile = PoseProfile([records[name] for name in ORDER]) if len(records) == len(ORDER) else None
        captured = self.target
        self.records = records
        self.recent.clear()  # A held/repeated P cannot record the same frames for the next direction.
        self.index += 1
        if profile is not None:
            self.profile, self.mode = profile, 'ready'
        return captured

    def back(self):
        if self.mode != 'collect' or self.index == 0:
            return False
        self.index -= 1
        self.records.pop(ORDER[self.index])
        self.recent.clear()
        return True

    def start_test(self):
        if self.profile is None:
            raise ValueError('먼저 정면·좌·우·위 보정을 모두 완료하세요.')
        self.mode, self.index, self.results = 'test', 0, []
        self._reset_hold()

    def test_frame(self, pose, now, hold_seconds=.4, min_frames=3, max_gap=.7):
        if self.mode != 'test':
            return False
        match, _ = self.profile.classify(pose)
        if (pose is None or match != self.target
                or self.last_test_seen is not None and not 0 < now-self.last_test_seen <= max_gap):
            self._reset_hold()
            if pose is None or match != self.target:
                return False
        self.last_test_seen = now
        if self.hold_started is None:
            self.hold_started = now
        self.hold_frames += 1
        if self.hold_frames < min_frames or now-self.hold_started < hold_seconds:
            return False
        self.results.append(self.target)
        self.index += 1
        self._reset_hold()
        if self.index == len(ORDER):
            self.mode = 'tested'
        return True

    def cancel(self):
        if self.mode == 'collect':
            self.reset()
        elif self.mode == 'test':
            self.mode = 'ready'
            self._reset_hold()
