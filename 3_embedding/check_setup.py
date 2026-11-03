"""Offline model diagnostics; optionally test webcam and the Tkinter interface."""

import argparse
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from face_core import Face, FaceEngine, FaceStore, normalize, open_camera
from models import ensure_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, help="Briefly read this camera; no images are saved")
    parser.add_argument("--gui", action="store_true", help="Smoke test GUI rendering without starting the camera")
    args = parser.parse_args()
    engine = FaceEngine(*ensure_models())
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert engine.analyze(blank) == []
    embedding = normalize(engine.recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8)))
    assert embedding.shape == (128,)
    assert abs(float(embedding @ embedding) - 1) < 1e-5
    print(f"PASS: OpenCV {cv2.__version__}, model integrity, inference, normalized 128D embedding")
    if args.gui:
        import tkinter as tk
        from app import FaceApp
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                ui = FaceApp(root, store=FaceStore(Path(directory) / "test.json"))
                ui._handle_frame(blank, [Face((120, 100, 120, 120), embedding, 100)], time.monotonic(), 20)
                root.update_idletasks()
                assert ui.result.get() == "미등록"
                ui.store.enroll("테스트 얼굴", [embedding] * 15)
                ui._handle_frame(blank, [Face((120, 100, 120, 120), embedding, 100)], time.monotonic(), 20)
                assert "테스트 얼굴" in ui.result.get()
            finally:
                root.destroy()
        print("PASS: Tkinter interface, Pillow preview, Korean face labels, recognition display")
    if args.camera is not None:
        capture = open_camera(args.camera)
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera read failed")
            height, width = frame.shape[:2]
            scale = min(640 / width, 480 / height, 1)
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            faces = engine.analyze(cv2.flip(frame, 1))
            print(f"PASS: camera {args.camera}, frame {width}x{height}, {len(faces)} face(s) detected; no images saved")
        finally:
            capture.release()


if __name__ == "__main__":
    main()
