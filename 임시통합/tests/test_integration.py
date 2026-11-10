"""Authenticated-only storage, sampling, and explicit enrollment regressions."""

from contextlib import closing
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from authenticate import authenticate_face
from database import add_user, delete_user, initialize_database
from detection import detect_faces
from face_core import Enrollment, Face, REGISTRATION_SAMPLES, normalize
from pipeline import FacePipeline
from standby import FaceWatcher, GateSettings


def row(x=110, y=40):
    return np.array([x, y, 100, 110, x+25, y+30, x+75, y+30,
                     x+50, y+55, x+30, y+80, x+70, y+80, .95], dtype=np.float32)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "기록" / "faces.sqlite3"
        self.store = initialize_database(self.path)
        self.frame = np.full((180, 320, 3), 100, dtype=np.uint8)
        self.embedding = np.zeros(128, dtype=np.float32)
        self.embedding[0] = 1
        self.query = self.embedding.copy()
        self.engine = Mock()
        self.engine.detect.return_value = np.array([row()])
        self.engine.analyze.side_effect = lambda frame, rows: [Face(tuple(rows[0][:4]), self.query, 100)]

    def enroll(self):
        add_user(self.store, "홍길동", [self.embedding] * REGISTRATION_SAMPLES)

    def pipeline(self, hold=0, interval=0):
        return FacePipeline(self.engine, FaceWatcher(GateSettings(hold_seconds=hold)),
                            self.store, "test", save_interval=interval, start_in_standby=False)

    def test_unregistered_face_never_creates_observations(self):
        pipeline = self.pipeline()
        for _ in range(3):
            result = pipeline.process(self.frame)
        self.assertTrue(result.ready)
        self.assertIsNone(result.authenticated_name)
        self.assertEqual(self.store.count_observations(), 0)
        self.assertEqual(self.store.people, {})

    def test_unknown_face_rejected_even_when_someone_is_enrolled(self):
        self.enroll()
        self.query = np.roll(self.embedding, 1)
        result = self.pipeline().process(self.frame)
        self.assertIsNone(result.authenticated_name)
        self.assertEqual(self.store.count_observations(), 0)

    def test_authenticated_crop_and_embedding_saved_together_with_match_audit(self):
        self.enroll()
        result = self.pipeline().process(self.frame)
        self.assertEqual(result.authenticated_name, "홍길동")
        self.assertEqual(result.saved_this_frame, 1)
        with closing(self.store._connect()) as connection:
            saved = connection.execute("""SELECT crop_jpeg, embedding, gate_passed,
                matched_name, match_score, match_threshold FROM observations""").fetchone()
        jpeg, embedding, gate, name, score, threshold = saved
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR))
        self.assertEqual(len(json.loads(embedding)), 128)
        self.assertEqual((gate, name, score, threshold), (1, "홍길동", 1, .45))

    def test_gate_waiting_and_multiple_faces_not_saved_even_if_known(self):
        self.enroll()
        pipeline = self.pipeline(hold=.8)
        self.assertFalse(pipeline.process(self.frame).ready)
        self.engine.detect.return_value = np.array([row(10), row(190)])
        self.assertFalse(pipeline.process(self.frame).ready)
        self.assertEqual(self.store.count_observations(), 0)

    def test_embedding_failure_never_saves_a_crop(self):
        self.enroll()
        self.engine.analyze.side_effect = ValueError("bad embedding")
        result = self.pipeline().process(self.frame)
        self.assertEqual(result.embedding_errors, 1)
        self.assertFalse(result.ready)
        self.assertEqual(self.store.count_observations(), 0)

    def test_default_one_second_sampling_keeps_inference_running(self):
        self.enroll()
        pipeline = FacePipeline(self.engine, FaceWatcher(GateSettings(hold_seconds=0)), self.store,
                                start_in_standby=False)
        for index in range(60):
            with patch("pipeline.monotonic", return_value=index / 30):
                result = pipeline.process(self.frame)
                self.assertEqual(result.authenticated_name, "홍길동")
        self.assertEqual(self.engine.analyze.call_count, 60)
        self.assertEqual(self.store.count_observations(), 2)

    def test_runtime_threshold_is_used_for_both_matching_and_storage(self):
        self.enroll()
        self.query = normalize(self.embedding * .5 + np.roll(self.embedding, 1) * np.sqrt(.75))
        pipeline = self.pipeline()
        pipeline.threshold = .6
        self.assertIsNone(pipeline.process(self.frame).authenticated_name)
        self.assertEqual(self.store.count_observations(), 0)
        pipeline.threshold = .45
        self.assertEqual(pipeline.process(self.frame).authenticated_name, "홍길동")
        self.assertEqual(self.store.count_observations(), 1)

    def test_worker_store_sees_enrollment_and_deletion_from_ui_connection(self):
        pipeline = FacePipeline(self.engine, FaceWatcher(GateSettings(hold_seconds=0)),
                                initialize_database(self.path), save_interval=0, start_in_standby=False)
        self.assertIsNone(pipeline.process(self.frame).authenticated_name)
        self.enroll()
        self.assertEqual(pipeline.process(self.frame).authenticated_name, "홍길동")
        delete_user(self.store, "홍길동")
        self.assertIsNone(pipeline.process(self.frame).authenticated_name)
        self.assertEqual(self.store.count_observations(), 1)

    def test_database_itself_refuses_unauthenticated_or_gate_failing_writes(self):
        self.assertIsNone(self.store.record_authenticated(self.frame, row(), self.embedding, gate_passed=True))
        self.enroll()
        self.assertIsNone(self.store.record_authenticated(self.frame, row(), self.embedding, gate_passed=False))
        self.assertIsNone(self.store.record_authenticated(self.frame, row(), np.roll(self.embedding, 1), gate_passed=True))
        self.assertEqual(self.store.count_observations(), 0)

    def test_registration_can_enroll_without_recording_unknown_observations(self):
        pipeline = self.pipeline(interval=1)
        enrollment = Enrollment("별도 등록")
        for index in range(REGISTRATION_SAMPLES):
            now = index * .31
            with patch("pipeline.monotonic", return_value=now):
                result = pipeline.process(self.frame)
            enrollment.accept(result.faces, now)
        self.assertEqual(len(enrollment.samples), 15)
        self.assertEqual(self.store.count_observations(), 0)
        self.assertEqual(self.store.people, {})
        add_user(self.store, enrollment.name, enrollment.samples)
        self.assertEqual(pipeline.process(self.frame).authenticated_name, "별도 등록")

    def test_runtime_interval_change_preserves_last_save_clock(self):
        self.enroll()
        pipeline = self.pipeline(interval=1)
        for now in (0, .5):
            with patch("pipeline.monotonic", return_value=now):
                pipeline.process(self.frame)
        pipeline.save_interval = 2
        for now in (1, 1.99, 2):
            with patch("pipeline.monotonic", return_value=now):
                pipeline.process(self.frame)
        self.assertEqual(self.store.count_observations(), 2)

    def test_face_loss_requires_hold_again_before_saving(self):
        self.enroll()
        pipeline = self.pipeline(hold=.8)
        with patch("standby.time.monotonic", side_effect=[0, .4, .81]):
            self.assertFalse(pipeline.process(self.frame).ready)
            self.assertFalse(pipeline.process(self.frame).ready)
            self.assertTrue(pipeline.process(self.frame).ready)
        self.engine.detect.return_value = np.empty((0, 15), dtype=np.float32)
        self.assertFalse(pipeline.process(self.frame).ready)
        self.engine.detect.return_value = np.array([row()])
        with patch("standby.time.monotonic", return_value=.9):
            self.assertFalse(pipeline.process(self.frame).ready)
        self.assertEqual(self.store.count_observations(), 1)


class DetectionGeometryTests(unittest.TestCase):
    def test_letterbox_restores_box_and_all_landmarks(self):
        original = row()
        detector = Mock()
        padded_row = original.copy()
        padded_row[:14] *= 2
        padded_row[[1, 5, 7, 9, 11, 13]] += 60
        detector.detect.return_value = (None, np.array([padded_row]))
        rows = detect_faces(detector, np.zeros((180, 320, 3), dtype=np.uint8))
        np.testing.assert_allclose(rows[0], original, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
