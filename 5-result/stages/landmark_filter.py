"""Filter YuNet's five measured points before pose solving; never predict missing faces."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from standby import overlap_ratio


@dataclass(frozen=True)
class LandmarkFilterSettings:
    smoothing_seconds: float = .18
    jump_ratio: float = .12
    min_confidence: float = .8
    max_gap: float = .7
    warmup_frames: int = 3

    def __post_init__(self):
        for name in ('smoothing_seconds', 'jump_ratio', 'min_confidence', 'max_gap'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('좌표 안정화 설정은 유한한 숫자여야 합니다.')
        if not (.02 <= self.smoothing_seconds <= 1 and .02 <= self.jump_ratio <= .5
                and .5 <= self.min_confidence <= .99 and .05 <= self.max_gap <= 5
                and type(self.warmup_frames) is int and 3 <= self.warmup_frames <= 15):
            raise ValueError('반응 시간 0.02~1초 / 튐 제한 2~50% / 신뢰도 0.5~0.99 범위를 확인하세요.')


@dataclass
class FilteredLandmarks:
    row: np.ndarray | None
    reason: str
    reset_baseline: bool = False
    raw_step: float = 0.
    filtered_step: float = 0.


class LandmarkFilter:
    def __init__(self, settings=None):
        self.settings = settings or LandmarkFilterSettings()
        self.reset()

    def reset(self):
        self.previous = self.smoothed = self.size = None
        self.last_seen = self.last_accepted = None
        self.history = deque(maxlen=3)
        self.streak = self.rejections = 0

    def update(self, rows, size, now):
        s = self.settings
        rows = np.asarray(rows, dtype=float)
        if rows.size == 0 or rows.ndim != 2 or rows.shape != (1, 15):
            self.reset()
            return FilteredLandmarks(None, '얼굴 소실/여러 얼굴 · 좌표와 영점 초기화', True)
        row = rows[0]
        width, height = size
        if (not math.isfinite(now) or not np.isfinite(row).all() or min(row[2:4]) <= 0
                or min(size) <= 0 or not 0 <= row[14] <= 1):
            self.reset()
            return FilteredLandmarks(None, '유효하지 않은 특징점 · 좌표와 영점 초기화', True)
        if row[14] < s.min_confidence:
            self.reset()
            return FilteredLandmarks(None, f'신뢰도 {int(row[14]*100)}% 미달 · 판정 중지', True)
        points = row[4:14].reshape(5, 2)
        if (np.any(points < [-.05*width, -.05*height])
                or np.any(points > [1.05*width, 1.05*height])):
            self.reset()
            return FilteredLandmarks(None, '특징점이 영상 밖에 있습니다 · 판정 중지', True)
        restarted = False
        if self.last_seen is not None and (now <= self.last_seen or now-self.last_seen > s.max_gap
                or now-self.last_accepted > s.max_gap or size != self.size
                or overlap_ratio(self.previous[:4], row[:4]) < .25):
            self.reset()
            restarted = True
        elapsed = now-self.last_seen if self.last_seen is not None else 0.
        self.last_seen, self.size = now, size
        scale = max(float(row[2]), 1.)
        raw_step = 0.
        if self.previous is not None:
            delta = points-self.previous[4:14].reshape(5, 2)
            translation = np.median(delta, axis=0)
            residual = np.linalg.norm(delta-translation, axis=1).max()/scale
            raw_step = float(np.linalg.norm(delta, axis=1).max()/scale)
            if residual > s.jump_ratio or np.linalg.norm(translation)/scale > 2*s.jump_ratio:
                self.history.clear()
                self.smoothed = None
                self.streak = 0
                self.rejections += 1
                if self.rejections >= 3:
                    self.reset()
                    return FilteredLandmarks(None, '좌표가 연속으로 급변 · 재검출/영점 설정 필요', True,
                                             raw_step=raw_step)
                return FilteredLandmarks(None, '좌표 급변 프레임 제외 · 새 연속 프레임 대기',
                                         raw_step=raw_step)
        self.rejections = 0
        self.previous, self.last_accepted = row.copy(), now
        self.history.append(points.copy())
        self.streak += 1
        median = np.median(np.stack(self.history), axis=0)
        filtered_step = 0.
        if self.smoothed is None:
            self.smoothed = median
        else:
            movement = float(np.linalg.norm(median-self.smoothed, axis=1).max()/scale)
            # Follow deliberate motion faster; stationary point noise gets the full smoothing time.
            motion = float(np.clip((movement-.015)/.06, 0, 1))
            tau = s.smoothing_seconds*(1-.8*motion)
            alpha = 1-math.exp(-elapsed/tau)
            previous = self.smoothed.copy()
            self.smoothed += alpha*(median-self.smoothed)
            filtered_step = float(np.linalg.norm(self.smoothed-previous, axis=1).max()/scale)
        if self.streak < s.warmup_frames:
            return FilteredLandmarks(None, f'좌표 수집 {self.streak}/{s.warmup_frames} · 판정 대기',
                                     restarted, raw_step, filtered_step)
        filtered = row.copy()
        filtered[4:14] = self.smoothed.reshape(-1)
        return FilteredLandmarks(filtered, '좌표 안정화 적용', restarted, raw_step, filtered_step)
