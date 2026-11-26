"""Picamera2 adapter for CSI modules; one configuration change per capture profile."""

import cv2


class PiCamera:
    def __init__(self, camera, size, fps):
        self.camera = camera
        self.started = False
        self.closed = False
        try:
            self.configure_profile(size, fps)
        except BaseException:
            self.release()
            raise

    def configure_profile(self, size, fps):
        if self.started:
            self.camera.stop()
            self.started = False
        # Picamera2 RGB888 arrays have B,G,R byte order, as required by OpenCV.
        config = self.camera.create_video_configuration(
            main={"size": tuple(size), "format": "RGB888"},
            controls={"FrameRate": float(fps)}, buffer_count=3)
        self.camera.configure(config)
        self.camera.start()
        self.started = True
        self.size, self.fps = tuple(size), float(fps)

    def read(self):
        frame = self.camera.capture_array("main")
        return frame is not None and bool(frame.size), frame

    def grab(self):
        return self.read()[0]

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.size[0],
                cv2.CAP_PROP_FRAME_HEIGHT: self.size[1],
                cv2.CAP_PROP_FPS: self.fps}.get(prop, 0.)

    def release(self):
        if self.closed:
            return
        try:
            if self.started:
                self.camera.stop()
        finally:
            self.started, self.closed = False, True
            self.camera.close()


def try_open_pi_camera(index, size, fps):
    try:
        from picamera2 import Picamera2
    except ModuleNotFoundError as exc:
        if exc.name != 'picamera2':
            raise
        return None
    info = Picamera2.global_camera_info()
    if not 0 <= index < len(info):
        return None
    return PiCamera(Picamera2(index), size, fps)
