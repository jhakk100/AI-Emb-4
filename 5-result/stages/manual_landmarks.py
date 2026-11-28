"""Temporary manual offsets for the five measured landmarks in the pose tester."""

import numpy as np


class ManualLandmarks:
    def __init__(self):
        self.offsets = None
        self.cancel()

    @property
    def active(self):
        return self.offsets is not None

    @property
    def editing(self):
        return self.source is not None

    @staticmethod
    def _row(row):
        row = np.asarray(row, dtype=float)
        if row.shape != (15,) or not np.isfinite(row).all() or min(row[2:4]) <= 0:
            raise ValueError('조정할 얼굴 특징점이 올바르지 않습니다.')
        return row

    @staticmethod
    def _validate(points, source, size, frontal=False):
        width, height = size
        if (points.shape != (5, 2) or not np.isfinite(points).all()
                or np.any(points < 0) or np.any(points >= [width, height])):
            raise ValueError('5개 점을 영상 안에 배치하세요.')
        original = source[4:14].reshape(5, 2)
        if np.any(np.abs((points-original)/source[2:4]) > .3):
            raise ValueError('자동 검출 위치에서 얼굴 크기의 30% 이내로 조정하세요.')
        if not frontal:
            return
        # Preserve point correspondence and the frontal eye/nose/mouth layout.
        for first, second in ((0, 1), (3, 4)):
            sign = np.sign(original[second, 0]-original[first, 0])
            if sign*(points[second, 0]-points[first, 0]) < source[2]*.03:
                raise ValueError('양쪽 눈 또는 입꼬리가 서로 뒤바뀌거나 겹치지 않도록 배치하세요.')
        if not (max(points[:2, 1]) < points[2, 1] < min(points[3:, 1])):
            raise ValueError('정면에서 눈 → 코끝 → 입꼬리 순서로 배치하세요.')

    def begin(self, row, size):
        source = self._row(row).copy()
        current = self.apply(source, size)
        self.source, self.size = source, tuple(size)
        self.points = current[4:14].reshape(5, 2).copy()

    def move(self, index, x, y):
        if not self.editing or not 0 <= index < 5 or not np.isfinite([x, y]).all():
            raise ValueError('조정할 점을 먼저 선택하세요.')
        self.points[index] = np.clip([x, y], [0, 0], np.array(self.size)-1)

    def commit(self):
        if not self.editing:
            raise ValueError('먼저 수동 위치 조정을 시작하세요.')
        self._validate(self.points, self.source, self.size, frontal=True)
        offsets = (self.points-self.source[4:14].reshape(5, 2))/self.source[2:4]
        self.offsets = offsets.copy()
        self.cancel()

    def apply(self, row, size):
        output = self._row(row).copy()
        if self.active:
            points = output[4:14].reshape(5, 2)+self.offsets*output[2:4]
            self._validate(points, output, size)
            output[4:14] = points.reshape(-1)
        return output

    def cancel(self):
        self.source = self.points = self.size = None

    def clear(self):
        self.offsets = None
        self.cancel()

    def labels(self):
        if not self.editing:
            return ()
        return self.labels_for(self.source)

    @staticmethod
    def labels_for(row):
        labels = ['']*5
        for indices, names in (((0, 1), ('왼쪽 눈', '오른쪽 눈')),
                               ((3, 4), ('왼쪽 입꼬리', '오른쪽 입꼬리'))):
            ordered = sorted(indices, key=lambda index: row[4+index*2])
            for index, name in zip(ordered, names):
                labels[index] = name
        labels[2] = '코끝'
        return tuple(labels)
