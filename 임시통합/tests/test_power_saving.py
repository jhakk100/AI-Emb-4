"""Standby must actually lower detector input/rate and skip SFace entirely."""

from pathlib import Path
from queue import Queue
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from app import CameraWorker
from capture_control import ACTIVE_PROFILE, STANDBY_PROFILE, configure_capture, frame_delay, prepare_frame
from database import initialize_database
from face_core import Face
from pipeline import FacePipeline
from standby import CHECK_FPS, DETECT_SIZE, FaceWatcher, GateSettings


def row():
    return np.array([110, 40, 100, 110, 135, 70, 185, 70,
                     160, 95, 140, 120, 180, 120, .95], dtype=np.float32)


class PowerSavingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = initialize_database(Path(temporary.name) / "test.sqlite3")
        self.engine = Mock()
        self.engine.detect.return_value = np.array([row()])
        self.embedding = np.zeros(128, dtype=np.float32)
        self.embedding[0] = 1
        self.engine.analyze.return_value = [Face((110, 40, 100, 110), self.embedding, 100)]
        self.low = np.zeros((180, 320, 3), dtype=np.uint8)
        self.high = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_standby_uses_preset_input_and_skips_embedding_and_db_even_when_faces_detected(self):
        pipeline = FacePipeline(self.engine, store=self.store)
        for _ in range(3):
            result = pipeline.process(self.low)
        self.assertEqual(pipeline.profile, STANDBY_PROFILE)
        self.assertEqual(result.processing_size, DETECT_SIZE)
        self.assertEqual(result.target_fps, CHECK_FPS)
        self.engine.detect.assert_called_with(self.low, size=DETECT_SIZE)
        self.engine.analyze.assert_not_called()
        self.assertEqual(self.store.count_observations(), 0)
        self.assertEqual(len(result.detection_boxes), 1)

    def test_wake_uses_next_high_resolution_frame_for_embedding(self):
        pipeline = FacePipeline(self.engine, FaceWatcher(GateSettings(hold_seconds=0)), self.store)
        waking = pipeline.process(self.low)
        self.assertEqual(waking.power_mode, "standby")
        self.assertFalse(waking.ready)
        self.engine.analyze.assert_not_called()
        self.assertEqual(pipeline.profile, ACTIVE_PROFILE)
        result = pipeline.process(self.high)
        self.assertTrue(result.ready)
        self.assertEqual(result.power_mode, "active")
        self.engine.detect.assert_called_with(self.high, size=ACTIVE_PROFILE.size)
        self.assertIs(self.engine.analyze.call_args.args[0], self.high)

    def test_lost_face_returns_to_standby_and_skips_embedding(self):
        pipeline = FacePipeline(self.engine, FaceWatcher(GateSettings(hold_seconds=0)), self.store)
        pipeline.process(self.low)
        pipeline.process(self.high)
        self.engine.analyze.reset_mock()
        self.engine.detect.return_value = np.empty((0, 15), dtype=np.float32)
        pipeline.process(self.high)
        self.assertEqual(pipeline.profile, STANDBY_PROFILE)
        self.engine.analyze.assert_not_called()
        pipeline.process(self.low)
        self.engine.detect.assert_called_with(self.low, size=DETECT_SIZE)

    def test_door_reset_restores_low_profile_and_full_hold_duration(self):
        pipeline = FacePipeline(self.engine, FaceWatcher(GateSettings(hold_seconds=.8)), self.store)
        with patch("standby.time.monotonic", side_effect=[0, .4, .81]):
            for _ in range(3):
                pipeline.process(self.low)
        self.assertEqual(pipeline.profile, ACTIVE_PROFILE)
        pipeline.reset_standby()
        self.assertEqual(pipeline.profile, STANDBY_PROFILE)
        self.assertEqual(pipeline.watcher.settings.hold_seconds, .8)
        with patch("standby.time.monotonic", return_value=1):
            self.assertFalse(pipeline.process(self.low).ready)
        self.assertEqual(pipeline.profile, STANDBY_PROFILE)

    def test_unsupported_camera_is_still_downsampled_to_exact_low_input(self):
        raw = np.full((720, 1280, 3), 50, dtype=np.uint8)
        small = prepare_frame(raw, STANDBY_PROFILE.size)
        self.assertEqual(small.shape, (180, 320, 3))
        portrait = prepare_frame(np.zeros((480, 320, 3), dtype=np.uint8), STANDBY_PROFILE.size)
        self.assertEqual(portrait.shape, (180, 320, 3))

    def test_camera_requests_include_both_low_resolution_and_low_fps(self):
        capture = Mock()
        configure_capture(capture, DETECT_SIZE, CHECK_FPS)
        for property_id, value in ((cv2.CAP_PROP_FRAME_WIDTH, 320), (cv2.CAP_PROP_FRAME_HEIGHT, 180),
                                   (cv2.CAP_PROP_FPS, 5)):
            self.assertIn(((property_id, value),), capture.set.call_args_list)
        self.assertAlmostEqual(frame_delay(10, 10.03, 5), .17)
        self.assertAlmostEqual(frame_delay(10, 10.03, 20), .02)
        self.assertEqual(frame_delay(10, 10.3, 5), 0)

    def test_worker_reads_and_processes_at_five_fps_even_if_device_reports_thirty(self):
        worker = CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1))
        clock, reads = [0.0], []
        capture = Mock()
        capture.get.return_value = 30
        self.engine.detect.return_value = np.empty((0, 15), dtype=np.float32)
        def read():
            reads.append(clock[0])
            clock[0] += .01
            if len(reads) == 5:
                worker.stop_event.set()
            return True, np.zeros((720, 1280, 3), dtype=np.uint8)
        capture.read.side_effect = read
        def wait(delay):
            clock[0] += delay
        def make_pipeline(engine, **kwargs):
            return FacePipeline(engine, store=self.store, **kwargs)
        with patch("app.ensure_models", return_value=("detector", "recognizer")), \
             patch("app.FaceEngine", return_value=self.engine), \
             patch("app.FacePipeline", side_effect=make_pipeline), \
             patch("app.open_camera", return_value=capture) as opening, \
             patch("app.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(worker.stop_event, "wait", side_effect=wait):
            worker.run()
        opening.assert_called_once_with(0, size=DETECT_SIZE, fps=CHECK_FPS)
        np.testing.assert_allclose(reads, [0, .2, .4, .6, .8])
        self.assertEqual(self.engine.detect.call_count, 5)
        for call in self.engine.detect.call_args_list:
            self.assertEqual(call.args[0].shape, (180, 320, 3))
        self.engine.analyze.assert_not_called()
        capture.release.assert_called_once()

    def test_buffered_camera_bursts_cannot_burst_inference_above_five_fps(self):
        worker = CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1))
        capture, clock, inference_times, count = Mock(), [0.0], [], [0]
        capture.get.return_value = 5
        def read():
            count[0] += 1
            if count[0] == 1:
                clock[0] += .2
            if count[0] == 5:
                worker.stop_event.set()
            return True, self.low
        def detect(*args, **kwargs):
            inference_times.append(clock[0])
            return np.empty((0, 15), dtype=np.float32)
        capture.read.side_effect = read
        self.engine.detect.side_effect = detect
        def wait(delay):
            clock[0] += delay
        with patch("app.ensure_models", return_value=("d", "r")), \
             patch("app.FaceEngine", return_value=self.engine), \
             patch("app.FacePipeline", side_effect=lambda engine, **kwargs: FacePipeline(engine, store=self.store, **kwargs)), \
             patch("app.open_camera", return_value=capture), \
             patch("app.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(worker.stop_event, "wait", side_effect=wait):
            worker.run()
        np.testing.assert_allclose(inference_times, [.2, .4, .6, .8, 1.0])


if __name__ == "__main__":
    unittest.main()
