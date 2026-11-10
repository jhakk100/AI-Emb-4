"""Offline model diagnostics; optionally test webcam and the Tkinter interface."""

import argparse
from pathlib import Path
import tempfile
import time
from unittest.mock import Mock, patch

import cv2
import numpy as np

from face_core import Face, FaceEngine, FaceStore, normalize, open_camera
from models import ensure_models
from database import initialize_database
from pipeline import FrameResult


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
                store = initialize_database(Path(directory) / "test.sqlite3")
                ui = FaceApp(root, store=store)
                ui._handle_frame(blank, [Face((120, 100, 120, 120), embedding, 100)], time.monotonic(), 20)
                root.update_idletasks()
                assert ui.result.get() == "미등록 · 저장 안 함"
                low = FrameResult([], False, "절전 대기", 0, 0, 0, 1, 0,
                                  power_mode="standby", processing_size=(320, 180), target_fps=5,
                                  detection_boxes=[(110, 40, 100, 110)], capture_size=(640, 360), capture_fps=5)
                ui._handle_frame(np.zeros((180, 320, 3), dtype=np.uint8), [], time.monotonic(), 5,
                                 low.status, False, 0, low)
                assert "320×180" in ui.power_text.get() and "640×360" in ui.power_text.get()
                assert "임베딩 중지" in ui.indicators.details[2].get()
                ui.store.enroll("테스트 얼굴", [embedding] * 15)
                ui._handle_frame(blank, [Face((120, 100, 120, 120), embedding, 100)], time.monotonic(), 20)
                assert "테스트 얼굴" in ui.result.get()
                detection = np.array([[120, 100, 120, 120, 145, 130, 195, 130,
                                       170, 160, 150, 190, 190, 190, .95]], dtype=np.float32)
                sample_id = store.record_authenticated(blank, detection[0], embedding, gate_passed=True)
                ui._refresh_history()
                ui.history.selection_set(str(sample_id))
                ui._show_crop()
                assert ui.crop_photo is not None
                assert "1" in ui.saved_text.get()
                ui._handle_frame(blank, [Face((120, 100, 120, 120), embedding, 100)],
                                 time.monotonic(), 20, "조건 확인 중", False, 1)
                assert not ui.gate_ready
                assert "테스트 얼굴" not in ui.result.get()
                telemetry = FrameResult([Face((120, 100, 120, 120), embedding, 100)], True,
                                        "준비 완료", 1, 1, 1, 1, 0, "테스트 얼굴", 1.0)
                ui.worker = Mock()
                ui.worker.stop_event.is_set.return_value = False
                ui._handle_frame(blank, telemetry.faces, time.monotonic(), 20, telemetry.status,
                                 True, 1, telemetry)
                assert ui.indicators.color == "green"
                assert ui.door.phase == "open"
                assert ui.door_message.get() == "문이 열렸습니다."
                assert "1개 검출" in ui.indicators.details[1].get()
                assert "인증 성공" in ui.indicators.details[3].get()
                with patch("app.time.monotonic", return_value=ui.door.deadline):
                    ui._advance_door()
                assert ui.door.phase == "closed"
                assert ui.door_message.get() == "문이 닫혔습니다."
                with patch("app.time.monotonic", return_value=ui.door.deadline):
                    ui._advance_door()
                assert ui.door.phase == "waiting" and not ui.gate_ready
                assert ui.worker.commands.put.call_args.args == (("reset_gate", None),)
                ui.mode.set("once")
                ui.apply_mode()
                ui._handle_frame(blank, telemetry.faces, time.monotonic(), 20, telemetry.status,
                                 True, 1, telemetry)
                with patch("app.time.monotonic", return_value=ui.door.deadline):
                    ui._advance_door()
                with patch("app.time.monotonic", return_value=ui.door.deadline):
                    ui._advance_door()
                assert ui.door.completed and ui.door_message.get() == "문이 닫혔습니다."
                ui.worker.stop_event.set.assert_called()
                ui.save_interval.set("2.0")
                ui.apply_save_interval()
                assert ui.worker.commands.put.call_args.args == (("save_interval", 2.0),)
                ui.apply_threshold("0.60")
                assert ui.worker.commands.put.call_args.args == (("threshold", .60),)
                ui.worker = None
                ui._update_led()
                assert ui.indicators.color == "off"
            finally:
                root.destroy()
        print("PASS: GUI, authenticated crop history, LED, settings, open/close/reset and one-shot completion")
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
