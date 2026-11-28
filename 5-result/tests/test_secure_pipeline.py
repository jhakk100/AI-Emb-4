"""Four-pose registration, deferred Mesh analysis and two-photo authentication regressions."""

from pathlib import Path
from contextlib import closing
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stages'))

import cv2
import numpy as np

from database import add_user, delete_user, initialize_database
from face_core import Face, REGISTRATION_SAMPLES
from guided_capture import (CaptureSample, DIRECTIONS, FACE_POINTS, GuidedCapture, GuidedSettings,
                            direction_matches, mesh_errors, normalized_mesh, yunet_pose)
from liveness import Pose
from pipeline import FacePipeline
from enrollment import EnrollmentPipeline
from camera_session import CameraSession
from standby import FaceWatcher, GateSettings
from capture_control import ENROLLMENT_PROFILE, STANDBY_PROFILE

BOX = (240., 120., 160., 190.)
SIZE = (640, 480)
ANGLES = {'front': (0., 0.), 'up': (0., -20.), 'down': (0., 20.),
          'left': (-25., 0.), 'right': (25., 0.)}


def detection():
    x, y, w, h = BOX
    return np.array([x, y, w, h, x+40, y+45, x+120, y+45, x+80, y+90,
                     x+50, y+140, x+110, y+140, .99], dtype=np.float32)


def landmarks():
    t = np.linspace(0, 2*np.pi, 478)
    raw = np.column_stack([.5 + .12*np.cos(t), .5 + .18*np.sin(t), .03*np.cos(t*3)]).astype(np.float32)
    raw[33], raw[263] = [.35, .45, .01], [.65, .45, .01]
    return raw


def sample(direction='front', index=0, embedding=None):
    vector = np.eye(1, 128, dtype=np.float32).reshape(-1) if embedding is None else embedding
    raw = landmarks()
    jpeg = cv2.imencode('.jpg', np.zeros((20, 20, 3), np.uint8))[1].tobytes()
    yaw, pitch = ANGLES[direction]
    return CaptureSample(direction, index, index*.5, vector.copy(), raw,
                         normalized_mesh(raw, SIZE), Pose(yaw, pitch, 0, BOX), SIZE, detection(), jpeg)


def profile(embedding=None):
    return [sample(direction, i, embedding) for direction in DIRECTIONS for i in range(3)]


class GeometryTests(unittest.TestCase):
    def test_yunet_five_landmarks_recover_synthetic_rotation(self):
        camera = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=float)
        for yaw, pitch in ((0, 0), (25, 0), (-25, 0), (0, 20), (0, -20)):
            # Canonical +yaw points the nose left; our API uses screen-right +yaw.
            ry = cv2.Rodrigues(np.array([0., -np.radians(yaw), 0.]))[0]
            rx = cv2.Rodrigues(np.array([np.radians(pitch), 0., 0.]))[0]
            rotation = cv2.Rodrigues(ry @ rx)[0]
            points = cv2.projectPoints(FACE_POINTS, rotation, np.array([0., 0., 600.]), camera,
                                       np.zeros(4))[0].reshape(5, 2)
            result = yunet_pose(np.r_[BOX, points.ravel(), .99], SIZE)
            self.assertAlmostEqual(result.yaw, yaw, places=2)
            self.assertAlmostEqual(result.pitch, pitch, places=2)

    def test_four_active_directions_and_roll_rejection(self):
        settings = GuidedSettings()
        for direction in DIRECTIONS:
            yaw, pitch = ANGLES[direction]
            self.assertTrue(direction_matches(direction, Pose(yaw, pitch, 0, BOX), settings))
            self.assertFalse(direction_matches(direction, Pose(yaw, pitch, 30, BOX), settings))
        self.assertFalse(direction_matches('down', Pose(0., 20., 0., BOX), settings))

    def test_enrollment_and_random_authentication_never_request_down(self):
        self.assertEqual(DIRECTIONS, ('front', 'up', 'left', 'right'))
        enrollment = GuidedCapture('enroll', GuidedSettings(), 0.)
        self.assertEqual(len(enrollment.directions)*enrollment.per_direction, REGISTRATION_SAMPLES)
        for _ in range(100):
            challenge = GuidedCapture('authenticate', GuidedSettings(), 0.)
            self.assertEqual(len(set(challenge.directions)), 2)
            self.assertTrue(set(challenge.directions).issubset(DIRECTIONS))
        with self.assertRaises(ValueError):
            GuidedCapture('authenticate', GuidedSettings(), 0., directions=('left', 'down'))

    def test_mesh_normalization_removes_translation_and_scale(self):
        raw = landmarks()
        changed = raw * 1.2 + np.array([.1, -.1, .2], dtype=np.float32)
        a, b = normalized_mesh(raw, SIZE), normalized_mesh(changed, SIZE)
        self.assertLess(mesh_errors(a, b)[0], 1e-5)
        self.assertLess(mesh_errors(a, b)[1], 1e-5)

    def test_flattened_or_deformed_depth_changes_mesh_error(self):
        raw = landmarks()
        bad = raw.copy()
        bad[:, 2] += np.sin(np.arange(478))*.2
        self.assertGreater(mesh_errors(normalized_mesh(raw, SIZE), normalized_mesh(bad, SIZE))[0], .12)

    def test_invalid_mesh_and_settings_fail_closed(self):
        with self.assertRaises(ValueError):
            normalized_mesh(np.zeros((478, 3)), SIZE)
        with self.assertRaises(ValueError):
            GuidedSettings(yaw_min=2)


class DirectionalDatabaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = initialize_database(Path(temp.name) / 'faces.sqlite3')
        self.store.enroll_directional('owner', profile())

    def test_twelve_crops_meshes_and_embeddings_are_stored(self):
        self.assertTrue(self.store.has_directional_profile('owner'))
        rows = self.store.directional_samples('owner')
        self.assertEqual(len(rows), 12)
        for direction in DIRECTIONS:
            self.assertEqual(sorted(r['index'] for r in rows if r['direction'] == direction), [0, 1, 2])
        self.assertEqual(rows[0]['landmarks'].shape, (478, 3))
        self.assertEqual(rows[0]['mesh'].shape, (468, 3))
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(rows[0]['crop_jpeg'], np.uint8), cv2.IMREAD_COLOR))

    def test_incomplete_reenrollment_does_not_replace_existing_profile(self):
        with self.assertRaises(ValueError):
            self.store.enroll_directional('owner', profile()[:-1])
        self.assertTrue(self.store.has_directional_profile('owner'))

    def test_partial_insert_failure_rolls_back_entire_profile(self):
        samples = profile()
        # Simulate a failure halfway through replacing a profile.
        with closing(self.store._connect()) as connection, connection:
            connection.execute("CREATE TRIGGER reject_sample BEFORE INSERT ON directional_samples "
                               "WHEN NEW.sample_index=2 BEGIN SELECT RAISE(ABORT,'simulated disk failure'); END")
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.enroll_directional('owner', samples)
        self.assertTrue(self.store.has_directional_profile('owner'))

    def test_adding_and_deleting_other_people_preserves_owner_meshes(self):
        vector = np.roll(sample().embedding, 1)
        add_user(self.store, 'legacy', [vector]*REGISTRATION_SAMPLES)
        self.assertTrue(self.store.has_directional_profile('owner'))
        delete_user(self.store, 'legacy')
        self.assertTrue(self.store.has_directional_profile('owner'))
        delete_user(self.store, 'owner')
        self.assertEqual(self.store.directional_samples('owner'), [])

    def test_auth_pair_is_saved_atomically_with_same_session(self):
        result = self.store.record_directional_pair('owner', [sample('left'), sample('up')],
                                                     'test', .45, GuidedSettings())
        self.assertEqual(len(result), 2)
        with closing(self.store._connect()) as connection:
            rows = connection.execute('SELECT direction, auth_session, mesh_error FROM observations').fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], rows[1][1])
        self.assertLess(rows[0][2], 1e-5)

    def test_one_bad_mesh_does_not_save_either_auth_photo(self):
        bad = sample('up')
        bad.mesh = bad.mesh.copy()
        bad.mesh[:, 2] += np.sin(np.arange(468)) * 2
        self.assertIsNone(self.store.record_directional_pair('owner', [sample('left'), bad],
                                                            'test', .45, GuidedSettings()))
        self.assertEqual(self.store.count_observations(), 0)

    def test_direction_labels_cannot_replace_actual_angle_checks(self):
        bad = sample('up')
        bad.pose = Pose(0, 0, 0, BOX)
        self.assertIsNone(self.store.record_directional_pair('owner', [sample('left'), bad],
                                                            'test', .45, GuidedSettings()))
        wrong_profile = profile()
        wrong_profile[-1].pose = Pose(0, 0, 0, BOX)
        with self.assertRaises(ValueError):
            self.store.enroll_directional('new', wrong_profile)
        self.assertNotIn('new', self.store.people)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = initialize_database(Path(temp.name) / 'faces.sqlite3')
        self.store.enroll_directional('owner', profile())
        self.vector = sample().embedding
        self.engine = Mock()
        self.engine.detect.return_value = np.array([detection()])
        self.engine.analyze.return_value = [Face(BOX, self.vector.copy(), 100)]
        self.mesh = Mock()
        self.mesh.observe.return_value = Pose(0, 0, 0, BOX, landmarks())
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.now, self.pose = 0., Pose(0, 0, 0, BOX)
        self.box = BOX
        self.pipeline = FacePipeline(self.engine, FaceWatcher(GateSettings(hold_seconds=0)), self.store,
            mesh=self.mesh, start_in_standby=False, pose_estimator=lambda row, size: self.pose)
        self.addCleanup(self.pipeline.close)
        self.choice_patch = patch('pipeline.GuidedCapture', side_effect=lambda p, s, n:
            GuidedCapture(p, s, n, directions=('left', 'up') if p == 'authenticate' else None))
        self.choice_patch.start()
        self.addCleanup(self.choice_patch.stop)

    def tick(self, direction='front', advance=.15):
        self.now += advance
        yaw, pitch = ANGLES[direction]
        self.pose = Pose(yaw, pitch, 0, self.box)
        with patch('pipeline.monotonic', return_value=self.now), \
                patch('enrollment.monotonic', side_effect=lambda: self.now), \
                patch('standby.time.monotonic', return_value=self.now):
            return self.pipeline.process(self.frame)

    def complete(self):
        result = self.tick()
        for _ in range(200):
            if self.pipeline.terminal:
                return result
            if self.pipeline.purpose == 'enroll':
                result = self.shoot()
            else:
                result = self.tick(self.pipeline.challenge.target or 'front')
        self.fail('directional session did not terminate')

    def shoot(self):
        with patch('enrollment.monotonic', return_value=self.now):
            self.assertTrue(self.pipeline.request_manual_capture(self.pipeline.session_id, self.pipeline.challenge.target,
                                                                 len(self.pipeline.challenge.samples), self.now))
        return self.tick()

    def begin_enrollment(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        row = detection() * np.array([3., 2.25]*7 + [1.], dtype=np.float32)
        self.box = tuple(row[:4])
        self.engine.detect.return_value = np.array([row])
        self.engine.analyze.return_value = [Face(self.box, self.vector.copy(), 100)]
        self.mesh.observe.return_value = Pose(0, 0, 0, self.box, landmarks())
        self.pipeline.close()
        self.mesh = Mock()
        self.mesh.observe.return_value = Pose(0, 0, 0, self.box, landmarks())
        with patch('enrollment.monotonic', return_value=self.now):
            self.pipeline = EnrollmentPipeline(self.engine, 'new',
                watcher=FaceWatcher(GateSettings(hold_seconds=0)), store=self.store, mesh=self.mesh,
                session_id=3, pose_estimator=lambda row, size: self.pose)
        self.addCleanup(self.pipeline.close)

    def test_front_only_does_not_run_mesh_or_authorize(self):
        for _ in range(10):
            result = self.tick()
        self.mesh.observe.assert_not_called()
        self.assertFalse(result.authentication_attempted)
        self.assertEqual(self.store.count_observations(), 0)

    def test_two_random_direction_photos_are_both_required(self):
        result = self.complete()
        self.assertTrue(result.liveness_passed)
        self.assertEqual(result.authenticated_name, 'owner')
        self.assertEqual(self.mesh.observe.call_count, 2)
        self.assertEqual(result.saved_this_frame, 2)
        self.assertEqual(self.store.count_observations(), 2)
        self.assertFalse(self.tick().authentication_attempted)

    def test_enroll_captures_exactly_three_per_direction_before_commit(self):
        self.begin_enrollment()
        self.tick()
        result = None
        while not self.pipeline.terminal:
            self.assertEqual(self.pipeline.profile, ENROLLMENT_PROFILE)
            if len(self.pipeline.challenge.samples) < 12:
                self.assertNotIn('new', self.store.people)
            result = self.shoot()
        self.assertEqual(result.enrollment_completed, 'new')
        self.assertEqual(result.enrollment_count, 12)
        self.assertTrue(self.store.has_directional_profile('new'))
        self.assertEqual(self.mesh.observe.call_count, 12)
        self.assertFalse(result.authentication_attempted)
        self.assertEqual(self.store.count_observations(), 0)
        self.assertEqual({row['size'] for row in self.store.directional_samples('new')}, {(1920, 1080)})

    def test_enrollment_preview_runs_no_inference_before_first_shutter(self):
        self.begin_enrollment()
        for _ in range(20):
            result = self.tick()
        self.assertEqual(result.enrollment_count, 0)
        self.assertEqual(result.direction, 'front')
        self.engine.detect.assert_not_called()
        self.engine.analyze.assert_not_called()
        self.mesh.observe.assert_not_called()

    def test_mesh_and_embedding_read_the_same_frozen_snapshot(self):
        self.tick()
        while not self.pipeline.challenge.samples:
            self.tick('left')
        captured = self.mesh.observe.call_args.args[0]
        self.assertIsNot(captured, self.frame)
        self.assertIs(self.engine.analyze.call_args.args[0], captured)
        self.frame[:] = 255
        self.assertEqual(int(captured.max()), 0)

    def test_wrong_direction_does_not_capture(self):
        self.tick()
        for _ in range(10):
            self.tick('right')
        self.assertEqual(self.mesh.observe.call_count, 0)
        self.assertEqual(self.pipeline.challenge.samples, [])

    def test_missing_face_and_multiple_faces_fail_without_writes(self):
        for count in (0, 2):
            self.pipeline.reset_standby()
            self.pipeline.mode = 'active'
            self.engine.detect.return_value = np.array([detection()])
            self.tick()
            self.engine.detect.return_value = np.array([detection()]*count).reshape(count, 15)
            self.assertTrue(self.tick().authentication_attempted)
        self.assertEqual(self.store.count_observations(), 0)

    def test_person_switch_at_capture_fails(self):
        self.tick()
        self.engine.analyze.return_value = [Face(BOX, np.roll(self.vector, 1), 100)]
        for _ in range(4):
            result = self.tick('left')
            if result.authentication_attempted:
                break
        self.assertTrue(result.authentication_attempted)
        self.assertIn('변경', result.status)

    def test_mesh_mismatch_in_second_photo_does_not_save_first(self):
        self.tick()
        while not self.pipeline.challenge.samples:
            self.tick('left')
        raw = landmarks()
        raw[:, 2] += np.sin(np.arange(478)) * .2
        self.mesh.observe.return_value = Pose(0, 0, 0, BOX, raw)
        for _ in range(5):
            result = self.tick('up')
            if result.authentication_attempted:
                break
        self.assertTrue(result.authentication_attempted)
        self.assertIsNone(result.authenticated_name)
        self.assertEqual(self.store.count_observations(), 0)

    def test_deleted_registration_cannot_finish(self):
        self.tick()
        delete_user(self.store, 'owner')
        for _ in range(4):
            result = self.tick('left')
            if result.authentication_attempted:
                break
        self.assertTrue(result.authentication_attempted)
        self.assertFalse(result.liveness_passed)

    def test_timeout_stops_even_when_pose_estimation_keeps_failing(self):
        self.tick()
        self.pipeline.pose_estimator = Mock(side_effect=ValueError('bad pose'))
        result = self.tick(advance=11)
        self.assertTrue(result.authentication_attempted)
        self.assertIn('시간 초과', result.status)

    def test_preview_standby_and_departure_skip_mesh(self):
        self.pipeline.set_purpose('preview', 7)
        self.pipeline.mode = 'active'
        self.assertTrue(self.tick().ready)
        self.pipeline.set_purpose('departure', 8)
        self.tick()
        self.pipeline.reset_standby()
        self.pipeline.watcher = FaceWatcher(GateSettings(hold_seconds=2))
        self.tick()
        self.mesh.observe.assert_not_called()

    def test_standby_keeps_fhd_preview_and_throttles_small_yunet_checks(self):
        self.pipeline.reset_standby()
        self.frame = np.zeros((1080, 1920, 3), np.uint8)
        self.engine.detect.return_value = np.empty((0, 15))
        for _ in range(15):
            result = self.tick(advance=.04)
        self.assertEqual(result.processing_size, (1920, 1080))
        self.assertEqual(result.target_fps, 30)
        self.assertEqual(self.engine.detect.call_count, 3)
        self.assertTrue(all(call.kwargs['size'] == (320, 180) for call in self.engine.detect.call_args_list))
        self.engine.analyze.assert_not_called()
        self.mesh.observe.assert_not_called()

    def test_legacy_embedding_only_registration_requires_reenrollment(self):
        add_user(self.store, 'owner', [self.vector]*REGISTRATION_SAMPLES)
        result = self.tick()
        self.assertTrue(result.authentication_attempted)
        self.assertIn('재등록', result.status)
        self.mesh.observe.assert_not_called()

    def test_enrollment_loss_keeps_partial_and_cannot_leave_stage_or_commit(self):
        self.begin_enrollment()
        self.shoot()
        self.engine.detect.return_value = np.empty((0, 15))
        result = self.shoot()
        self.assertIsNone(result.enrollment_error)
        self.assertEqual(result.enrollment_count, 1)
        self.assertNotIn('new', self.store.people)
        self.assertFalse(self.pipeline.set_purpose('preview', 4))
        self.assertEqual(self.pipeline.profile, ENROLLMENT_PROFILE)
        self.assertTrue(self.pipeline.enrollment_locked)

    def test_invalidated_session_cannot_commit_or_authorize(self):
        self.tick()
        self.pipeline.session_valid = lambda: False
        result = self.tick('left')
        self.assertFalse(result.authentication_attempted)
        self.assertEqual(self.store.count_observations(), 0)

    def test_neutral_yunet_pitch_bias_is_calibrated_before_direction_requests(self):
        self.pipeline.pose_estimator = lambda row, size: Pose(self.pose.yaw+5, self.pose.pitch+18, 0, BOX)
        result = self.complete()
        self.assertTrue(result.liveness_passed)
        self.assertEqual(self.pipeline.pose_origin, (5, 18))

    def test_enrollment_stage_rejects_implicit_stage_and_threshold_changes(self):
        self.begin_enrollment()
        self.tick()
        challenge = self.pipeline.challenge
        self.assertEqual(self.pipeline.mode, 'enrollment')
        for purpose in ('authenticate', 'preview', 'departure'):
            self.assertFalse(self.pipeline.set_purpose(purpose, 4))
            self.assertEqual(self.pipeline.session_id, 3)
            self.assertEqual(self.pipeline.profile, ENROLLMENT_PROFILE)
        self.assertFalse(self.pipeline.reset_standby())
        self.pipeline.threshold = .7
        self.assertEqual(self.pipeline.threshold, .45)
        self.assertIs(self.pipeline.challenge, challenge)

    def test_exclusive_enrollment_cannot_be_changed_in_place(self):
        self.begin_enrollment()
        self.tick()
        self.assertFalse(self.pipeline.set_purpose('preview', 4, cancel_enrollment=True))
        self.assertTrue(self.pipeline.enrollment_locked)
        self.assertEqual(self.pipeline.profile, ENROLLMENT_PROFILE)
        self.assertNotIn('new', self.store.people)

    def test_non_fhd_frame_never_reaches_enrollment_models(self):
        self.begin_enrollment()
        self.engine.detect.reset_mock()
        with patch('enrollment.monotonic', return_value=self.now+.1), self.assertRaisesRegex(RuntimeError, 'FHD'):
            self.pipeline.process(np.zeros((480, 640, 3), np.uint8))
        self.engine.detect.assert_not_called()
        self.mesh.observe.assert_not_called()

    def test_enrollment_constructor_starts_in_fhd_with_requested_threshold(self):
        pipeline = EnrollmentPipeline(self.engine, 'new', store=self.store,
                                threshold=.6, mesh=Mock())
        self.addCleanup(pipeline.close)
        self.assertEqual(pipeline.profile, ENROLLMENT_PROFILE)
        self.assertEqual(pipeline.mode, 'enrollment')
        self.assertEqual(pipeline.threshold, .6)

    def test_waiting_for_initial_frontal_pose_keeps_enrollment_fhd(self):
        self.begin_enrollment()
        self.engine.detect.return_value = np.empty((0, 15))
        for _ in range(5):
            self.tick()
            self.assertEqual(self.pipeline.profile, ENROLLMENT_PROFILE)
            self.assertEqual(self.pipeline.mode, 'enrollment')
        self.assertNotIn('new', self.store.people)

    def test_failed_enrollment_can_retry_without_leaving_fhd_or_saving_partial(self):
        self.begin_enrollment()
        self.shoot()
        self.engine.analyze.return_value = [Face(self.box, np.roll(self.vector, 1), 100)]
        self.assertTrue(self.shoot().enrollment_error)
        with patch('enrollment.monotonic', return_value=self.now):
            self.assertTrue(self.pipeline.restart(4))
        self.assertFalse(self.pipeline.terminal)
        self.assertEqual(self.pipeline.challenge.target, 'front')
        self.assertEqual(self.pipeline.challenge.samples, [])
        self.assertEqual(self.pipeline.profile, ENROLLMENT_PROFILE)
        self.assertEqual(self.pipeline.session_id, 4)
        self.assertNotIn('new', self.store.people)

    def test_offline_mesh_delay_does_not_interfere_with_the_next_manual_shutter(self):
        self.begin_enrollment()
        estimate = self.mesh.observe.return_value
        def analyze_photo(*args):
            self.now += 2.  # wall time spent computing this already captured photo
            return estimate
        self.mesh.observe.side_effect = analyze_photo
        self.shoot()
        result = self.shoot()
        self.assertFalse(result.enrollment_error)
        self.assertEqual(result.enrollment_count, 2)

    def test_enrollment_database_validation_failure_stays_locked_and_discards_attempt(self):
        self.begin_enrollment()
        with patch.object(self.store, 'enroll_directional', side_effect=ValueError('invalid profile')):
            result = self.complete()
        self.assertTrue(result.enrollment_error)
        self.assertFalse(result.authentication_attempted)
        self.assertTrue(self.pipeline.enrollment_locked)
        self.assertEqual(self.pipeline.profile, ENROLLMENT_PROFILE)
        self.assertIsNone(self.pipeline.challenge)
        self.assertNotIn('new', self.store.people)

    def test_auth_pipeline_rejects_enrollment_purpose(self):
        with self.assertRaises(ValueError):
            self.pipeline.set_purpose('enroll', enrollment_name='new')


class CameraSessionTests(unittest.TestCase):
    def test_enrollment_owns_a_separate_instance_until_explicit_exit_even_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = initialize_database(Path(directory) / 'faces.sqlite3')
            session = CameraSession(Mock(), store=store)
            self.addCleanup(session.close)
            old = session.active
            with patch.object(old, 'close') as close:
                self.assertTrue(session.set_purpose('enroll', 1, enrollment_name='new'))
            close.assert_called_once()
            registration = session.active
            self.assertIsInstance(registration, EnrollmentPipeline)
            self.assertIsNot(registration, old)
            registration._failure([], 'lost')
            for purpose in ('authenticate', 'departure', 'preview', 'enroll'):
                self.assertFalse(session.set_purpose(purpose, 2))
                self.assertIs(session.active, registration)
            session.threshold = .7
            self.assertEqual(session.threshold, .45)
            self.assertFalse(session.reset_standby())
            self.assertTrue(session.restart_enrollment(3))
            self.assertEqual(session.profile, ENROLLMENT_PROFILE)
            self.assertTrue(session.set_purpose('preview', 4, cancel_enrollment=True))
            self.assertIsInstance(session.active, FacePipeline)
            self.assertEqual(session.profile, STANDBY_PROFILE)
            self.assertEqual(store.count_observations(), 0)

    def test_success_does_not_allow_implicit_authentication_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            session = CameraSession(Mock(), purpose='enroll', enrollment_name='new',
                                    store=initialize_database(Path(directory) / 'faces.sqlite3'))
            self.addCleanup(session.close)
            session.active.terminal = session.active.completed = True
            self.assertFalse(session.set_purpose('authenticate', 2))
            self.assertFalse(session.restart_enrollment(2))
            self.assertEqual(session.profile, ENROLLMENT_PROFILE)


if __name__ == '__main__':
    unittest.main()
