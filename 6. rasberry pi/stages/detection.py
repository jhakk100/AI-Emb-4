"""Stage 2: letterbox detection and original-image crops (from 2_face)."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    path = Path(path)
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def detect_faces(detector, image, size=(640, 480)):
    """Return YuNet rows with box AND five landmarks in original coordinates."""
    height, width = image.shape[:2]
    target_w, target_h = size
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, min(target_w, round(width * scale)))
    resized_h = max(1, min(target_h, round(height * scale)))
    pad_x, pad_y = (target_w - resized_w) // 2, (target_h - resized_h) // 2
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = cv2.resize(
        image, (resized_w, resized_h))
    detector.setInputSize(size)
    _, rows = detector.detect(padded)
    if rows is None:
        return np.empty((0, 15), dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32).copy()
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 2] > 0) & (rows[:, 3] > 0)]
    scale_x, scale_y = resized_w / width, resized_h / height
    rows[:, 0] = (rows[:, 0] - pad_x) / scale_x
    rows[:, 1] = (rows[:, 1] - pad_y) / scale_y
    rows[:, 2] /= scale_x
    rows[:, 3] /= scale_y
    rows[:, 4:14:2] = (rows[:, 4:14:2] - pad_x) / scale_x
    rows[:, 5:14:2] = (rows[:, 5:14:2] - pad_y) / scale_y
    # Ignore detections entirely inside padding; every remaining row has a crop.
    valid = ((rows[:, 0] < width) & (rows[:, 1] < height)
             & (rows[:, 0] + rows[:, 2] > 0) & (rows[:, 1] + rows[:, 3] > 0))
    return rows[valid]


def crop_face(image, box):
    height, width = image.shape[:2]
    x, y, w, h = box
    x1, y1 = max(0, min(width, int(np.floor(x)))), max(0, min(height, int(np.floor(y))))
    x2, y2 = max(0, min(width, int(np.ceil(x + w)))), max(0, min(height, int(np.ceil(y + h))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("얼굴 영역이 영상 밖에 있습니다.")
    return image[y1:y2, x1:x2].copy()
