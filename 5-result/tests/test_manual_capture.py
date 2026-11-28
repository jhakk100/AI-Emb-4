"""Manual enrollment takes photos on demand and validates position, identity and freshness."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stages'))

import numpy as np

from camera_session import CameraSession
from database import initialize_database
from enrollment import EnrollmentPipeline
from face_core import Face
from guided_capture import MANUAL_POSE_BASIS
from liveness import Pose
from pipeline import FrameResult
from capture_control import ENROLLMENT_PROFILE
from test_secure_pipeline import detection, landmarks


class ManualCaptureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = initialize_database(Path(temp.name) / 'test.sqlite3')
        self.frame = np.zeros((1080, 1920, 3), np.uint8)
        self.row = detection()*np.array([3., 2.25]*7+[1.], np.float32)
        self.box = tuple(self.row[:4])
        self.vector = np.zeros(128, np.float32)
        self.vector[0] = 1
        self.engine = Mock()
        self.engine.detect.return_value = np.array([self.row])
        self.engine.analyze.return_value = [Face(self.box, self.vector, 100)]
        self.mesh = Mock()
        self.mesh.observe.return_value = Pose(0, 0, 0, self.box, landmarks())
        self.now = 1.
        self.estimate_pose = Mock(side_effect=AssertionError('manual enrollment must not estimate pose'))
        with patch('enrollment.monotonic', return_value=self.now):
            self.pipeline = EnrollmentPipeline(self.engine, 'user', store=self.store, mesh=self.mesh,
                                               pose_estimator=self.estimate_pose, session_id=4)
        self.addCleanup(self.pipeline.close)
        self.tick()

    def tick(self, advance=.05):
        self.now += advance
        with patch('enrollment.monotonic', return_value=self.now):
            return self.pipeline.process(self.frame)

    def request(self, **overrides):
        values = dict(session_id=4, direction=self.pipeline.challenge.target,
                      count=len(self.pipeline.challenge.samples), requested_at=self.now)
        values.update(overrides)
        with patch('enrollment.monotonic', return_value=self.now):
            return self.pipeline.request_manual_capture(**values)

    def test_p_captures_one_photo_without_waiting_for_auto_hold_and_does_not_repeat(self):
        self.assertTrue(self.request())
        result = self.tick()
        self.assertTrue(result.manual_captured)
        self.assertFalse(result.manual_capture_pending)
        self.assertEqual(result.enrollment_count, 1)
        self.assertEqual(self.pipeline.challenge.samples[0].capture_mode, 'manual')
        self.assertEqual(self.mesh.observe.call_count, 1)
        for _ in range(5):
            result = self.tick()
        self.assertEqual(result.enrollment_count, 1)
        self.assertFalse(result.manual_captured)

    def test_off_center_face_retries_same_slot_without_calling_mesh(self):
        off_center = self.row.copy()
        off_center[0] = 90
        self.engine.detect.return_value = np.array([off_center])
        self.assertTrue(self.request())
        result = self.tick()
        self.assertFalse(result.manual_captured)
        self.assertIn('가이드', result.manual_capture_status)
        self.assertEqual(result.enrollment_count, 0)
        self.tick()
        self.mesh.observe.assert_not_called()
        self.engine.detect.return_value = np.array([self.row])
        self.assertTrue(self.request())
        self.assertEqual(self.tick().enrollment_count, 1)

    def test_duplicate_requests_do_not_replace_the_original_one_shot(self):
        self.assertTrue(self.request())
        self.assertFalse(self.request())
        result = self.tick()
        self.assertEqual(result.enrollment_count, 1)
        self.assertEqual(self.mesh.observe.call_count, 1)

    def test_old_session_count_direction_or_timestamp_is_not_accepted(self):
        for overrides in (dict(session_id=3), dict(count=1), dict(direction='up'),
                          dict(requested_at=self.now-1), dict(requested_at=float('nan')), dict(requested_at=None)):
            self.assertFalse(self.request(**overrides))
        self.assertIsNone(self.pipeline.manual_request)
        self.tick()
        self.mesh.observe.assert_not_called()

    def test_expired_request_cannot_trigger_a_later_shutter(self):
        self.assertTrue(self.request())
        result = self.tick(advance=.6)
        self.assertFalse(result.manual_captured)
        self.assertEqual(result.enrollment_count, 0)
        self.assertIn('만료', result.manual_capture_status)
        self.tick()
        self.mesh.observe.assert_not_called()

    def test_separate_p_requests_do_not_need_a_minimum_spacing(self):
        self.assertTrue(self.request())
        self.tick()
        self.assertTrue(self.request())
        result = self.tick()
        self.assertEqual(result.enrollment_count, 2)
        self.assertTrue(result.manual_captured)

    def test_blurry_photo_is_rejected_then_requires_another_p_to_retry(self):
        self.engine.analyze.return_value = [Face(self.box, self.vector, 1)]
        self.assertTrue(self.request())
        result = self.tick()
        self.assertFalse(result.manual_captured)
        self.assertEqual(result.enrollment_count, 0)
        self.assertIn('수동 촬영 거절', result.manual_capture_status)
        self.mesh.observe.assert_not_called()
        self.engine.analyze.return_value = [Face(self.box, self.vector, 100)]
        self.tick()
        self.mesh.observe.assert_not_called()
        self.assertTrue(self.request())
        self.assertTrue(self.tick().manual_captured)

    def test_missing_face_cannot_be_manually_captured_or_release_the_enrollment_stage(self):
        self.assertTrue(self.request())
        self.engine.detect.return_value = np.empty((0, 15))
        result = self.tick()
        self.assertIsNone(result.enrollment_error)
        self.assertEqual(result.direction, 'front')
        self.assertFalse(result.manual_captured)
        self.assertTrue(self.pipeline.enrollment_locked)
        self.assertEqual(self.store.people, {})

    def test_manual_shutter_cannot_bypass_identity_consistency(self):
        self.assertTrue(self.request())
        self.tick()
        self.engine.analyze.return_value = [Face(self.box, np.roll(self.vector, 1), 100)]
        self.assertTrue(self.request())
        result = self.tick(advance=.45)
        self.assertTrue(result.enrollment_error)
        self.assertIn('변경', result.enrollment_error)
        self.assertFalse(result.manual_captured)
        self.assertEqual(self.store.people, {})

    def test_invalidated_session_discards_pending_manual_request(self):
        self.assertTrue(self.request())
        self.pipeline.session_valid = lambda: False
        self.assertFalse(self.tick().manual_captured)
        self.assertIsNone(self.pipeline.manual_request)
        self.pipeline.session_valid = lambda: True
        self.tick()
        self.mesh.observe.assert_not_called()

    def test_all_twelve_manual_photos_still_build_both_models_and_record_capture_mode(self):
        targets = []
        for count in range(12):
            targets.append(self.pipeline.challenge.target)
            self.assertTrue(self.request())
            result = self.tick()
            self.assertTrue(result.manual_captured, result.status)
            self.assertEqual(result.enrollment_count, count+1)
        self.assertEqual(result.enrollment_completed, 'user')
        self.assertTrue(result.grid_ready and result.fused_ready)
        self.assertEqual(targets, [direction for direction in ('front', 'up', 'left', 'right') for _ in range(3)])
        self.assertEqual(self.mesh.observe.call_count, 12)
        self.estimate_pose.assert_not_called()
        rows = self.store.directional_samples('user')
        self.assertEqual({row['capture_mode'] for row in rows}, {'manual'})
        self.assertTrue(all(row['pose'] is None and row['pose_basis'] == MANUAL_POSE_BASIS for row in rows))
        self.assertTrue(self.store.has_grid_profile('user'))
        self.assertTrue(self.store.has_fused_model('user'))
        self.assertEqual(self.store.count_observations(), 0)
        self.assertFalse(self.pipeline.request_manual_capture(4, 'front', 12, self.now))

    def test_session_router_never_exposes_manual_capture_to_authentication(self):
        session = CameraSession.__new__(CameraSession)
        session.active = Mock(purpose='authenticate')
        self.assertFalse(session.request_manual_capture(4, 'front', 0, self.now))
        session.active.request_manual_capture.assert_not_called()

    def test_camera_wait_and_missing_mesh_do_not_discard_successful_photos(self):
        self.assertTrue(self.request())
        self.tick()
        result = self.tick(advance=300)
        self.assertEqual(result.enrollment_count, 1)
        self.assertIsNone(result.enrollment_error)
        self.mesh.observe.return_value = None
        self.assertTrue(self.request())
        result = self.tick()
        self.assertEqual(result.enrollment_count, 1)
        self.assertFalse(result.manual_captured)
        self.assertEqual(result.direction, 'front')


class ConsoleManualCaptureTests(unittest.TestCase):
    def test_preview_returns_p_without_interpreting_it_as_quit(self):
        import integrate
        preview = integrate.Preview.__new__(integrate.Preview)
        preview.cv2 = Mock()
        preview.cv2.waitKey.return_value = ord('p')
        preview.cv2.getWindowProperty.return_value = 1
        result = FrameResult([], False, 'test', 0, 0, 0, 1, 0, purpose='enroll')
        self.assertEqual(preview.show(np.zeros((480, 640, 3), np.uint8), result), ord('p'))

    def test_console_enrollment_routes_p_to_next_frame_shutter_and_releases_resources(self):
        import integrate
        from capture_control import frame_delay
        stages = Mock()
        stages.face_core.REGISTRATION_SAMPLES = 12
        stages.capture_control.ENROLLMENT_PROFILE = ENROLLMENT_PROFILE
        stages.capture_control.frame_delay = frame_delay
        pipeline = Mock(profile=ENROLLMENT_PROFILE, session_id=4)
        stages.new_pipeline.return_value = pipeline
        pipeline.process.side_effect = [
            FrameResult([], False, 'ready', 0, 0, 0, 1, 0, direction='front', purpose='enroll'),
            FrameResult([], False, 'done', 0, 0, 0, 1, 0, enrollment_completed='user', enrollment_count=12, purpose='enroll')]
        camera = Mock()
        camera.read.return_value = np.zeros((1080, 1920, 3), np.uint8)
        preview = Mock()
        preview.show.return_value = ord('p')
        args = SimpleNamespace(threshold=None, camera=0, no_flip=False, name='user', show=True)
        with patch('integrate.Stages', return_value=stages), patch('integrate.Camera', return_value=camera), \
                patch('integrate.Preview', return_value=preview), patch('integrate.time.monotonic', return_value=1.), \
                patch('integrate.time.sleep'), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            integrate.cmd_enroll(args)
        pipeline.request_manual_capture.assert_called_once_with(4, 'front', 0, 1.)
        camera.release.assert_called_once()
        pipeline.close.assert_called_once()
        preview.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
