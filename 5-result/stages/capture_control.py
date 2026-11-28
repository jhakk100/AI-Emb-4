"""Capture requests plus an enforced software resolution and processing rate."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptureProfile:
    name: str
    size: tuple
    fps: int


CAMERA_SIZE = (1920, 1080)
CAMERA_FPS = 30
STANDBY_PROFILE = CaptureProfile("standby", CAMERA_SIZE, CAMERA_FPS)
ACTIVE_PROFILE = CaptureProfile("active", (640, 480), 20)
CALIBRATED_ACTIVE_PROFILE = CaptureProfile("active", CAMERA_SIZE, CAMERA_FPS)
ENROLLMENT_PROFILE = CaptureProfile("enrollment", CAMERA_SIZE, CAMERA_FPS)


class EnrollmentResolutionGate:
    """Reject low-resolution frames when the camera is expected to supply FHD."""
    def __init__(self, max_misses=10):
        self.max_misses = max_misses
        self.misses = 0

    def reset(self):
        self.misses = 0

    def accept(self, frame, profile):
        if profile.size != ENROLLMENT_PROFILE.size:
            self.reset()
            return True
        height, width = frame.shape[:2]
        if width >= 1920 and height >= 1080:
            self.reset()
            return True
        self.misses += 1
        if self.misses >= self.max_misses:
            raise RuntimeError(f'FHD 얼굴 처리 영상이 필요합니다. 카메라 실제 입력은 {width}×{height}입니다. '
                               '1920×1080 이상을 지원하는 카메라와 설정을 확인하세요.')
        return False


def configure_capture(capture, size, fps):
    """A device may reject these requests; software limits remain mandatory."""
    if hasattr(capture, "configure_profile"):
        capture.configure_profile(size, fps)
        return
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def prepare_frame(frame, size):
    """Letterbox to the exact processing resolution, even on unsupported webcams."""
    height, width = frame.shape[:2]
    target_w, target_h = size
    if (width, height) == size:
        return frame
    scale = min(target_w / width, target_h / height)
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    output = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    x, y = (target_w - resized_w) // 2, (target_h - resized_h) // 2
    output[y:y+resized_h, x:x+resized_w] = cv2.resize(frame, (resized_w, resized_h))
    return output


def frame_delay(started, now, fps):
    return max(0.0, 1 / fps - (now - started))
