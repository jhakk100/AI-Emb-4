"""Capture requests plus an enforced software resolution and processing rate."""

from dataclasses import dataclass

import cv2
import numpy as np

from standby import CHECK_FPS, DETECT_SIZE


@dataclass(frozen=True)
class CaptureProfile:
    name: str
    size: tuple
    fps: int


STANDBY_PROFILE = CaptureProfile("standby", DETECT_SIZE, CHECK_FPS)
ACTIVE_PROFILE = CaptureProfile("active", (640, 480), 20)


def configure_capture(capture, size, fps):
    """A device may reject these requests; software limits remain mandatory."""
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
