"""Manual YuNet pose tuning using the production direction and hold checks."""

from collections import deque
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from guided_capture import GuidedCapture, direction_matches, load_guided_settings
from liveness import Pose

TUNABLE = ('front_degrees', 'yaw_min', 'yaw_max', 'pitch_min', 'pitch_max',
           'cross_degrees', 'roll_max', 'hold_seconds', 'min_frames', 'max_gap',
           'yaw_sign', 'pitch_sign')


class PoseTuner:
    def __init__(self, settings):
        self.settings = settings
        self.origin = None
        self.recent = deque(maxlen=15)
        self.trial = self.target = None

    def reset_baseline(self):
        self.origin = None
        self.recent.clear()
        self.trial = self.target = None

    def configure(self, **values):
        if set(values)-set(TUNABLE):
            raise ValueError('방향 검사 설정만 변경할 수 있습니다.')
        updated = replace(self.settings, **values)
        if updated != self.settings:
            self.settings = updated
            self.trial = None

    def observe(self, raw, now):
        if raw is None:
            self.recent.clear()
            self.trial = None
            return None
        if self.recent and now-self.recent[-1][0] > self.settings.max_gap:
            self.recent.clear()
        self.recent.append((now, raw))
        if self.origin is None:
            return None
        return Pose(raw.yaw-self.origin[0], raw.pitch-self.origin[1], raw.roll, raw.box)

    def calibrate(self, now):
        if len(self.recent) < 5 or now-self.recent[-1][0] > .5:
            raise ValueError('정면을 보고 잠시 멈춘 뒤 기준 설정을 누르세요. 최근 영상 5장 이상이 필요합니다.')
        angles = np.array([[p.yaw, p.pitch, p.roll] for _, p in self.recent])
        if not np.isfinite(angles).all() or np.ptp(angles, axis=0).max() > 5:
            raise ValueError('고개를 움직이지 않고 정면을 유지한 뒤 다시 눌러주세요.')
        self.origin = tuple(angles.mean(axis=0)[:2])
        self.trial = None
        return self.origin

    def check(self, target, pose, now):
        if pose is None:
            self.trial = None
            return False, False
        if self.trial is None or self.target != target or self.trial.timed_out(now) or self.trial.failure:
            self.target = target
            self.trial = GuidedCapture('enroll', self.settings, now, directions=(target,))
        held = self.trial.candidate_due(pose, now)
        return direction_matches(target, pose, self.settings), held

    def save(self, path):
        # Preserve non-tuning values, including Mesh limits and authentication deadlines.
        current = load_guided_settings(path)
        updated = replace(current, **{name: getattr(self.settings, name) for name in TUNABLE})
        path = Path(path)
        fd, temporary = tempfile.mkstemp(prefix='.pose-', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                json.dump(asdict(updated), stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
