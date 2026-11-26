"""Regression tests for challenge continuity, identity binding and single-use results."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stages"))

import numpy as np

from database import add_user, delete_user, initialize_database
from face_core import Face, REGISTRATION_SAMPLES
from liveness import LivenessSettings, Pose, RotationChallenge
from pipeline import FacePipeline
from standby import FaceWatcher, GateSettings


BOX = (240., 120., 160., 190.)


def detection():
    x, y, w, h = BOX
    return np.array([x, y, w, h, x+40, y+45, x+120, y+45,
                     x+80, y+90, x+50, y+140, x+110, y+140, .99], dtype=np.float32)


def pose(yaw=0, pitch=0, roll=0, box=BOX):
    return Pose(yaw, pitch, roll, box)


class ChallengeTests(unittest.TestCase):
    def challenge(self):
        choices = iter((1, -1))
        return RotationChallenge(LivenessSettings(), 0, lambda values: next(choices))

    def test_valid_sequence_requires_front_between_turns(self):
        challenge = self.challenge()
        now = 0
        for yaw in (0, 25, 0, -25, 0):
            for _ in range(4):
                challenge.update(pose(yaw), now)
                now += .15
        self.assertTrue(challenge.complete)
        self.assertIsNone(challenge.failure)

    def test_stationary_face_times_out(self):
        challenge = self.challenge()
        for i in range(80):
            challenge.update(pose(), i * .1)
        self.assertFalse(challenge.complete)
        self.assertIsNotNone(challenge.failure)

    def test_wrong_direction_never_completes_turn(self):
        challenge = self.challenge()
        for i in range(4):
            challenge.update(pose(), i * .15)
        for i in range(4, 20):
            challenge.update(pose(-25), i * .15)
        self.assertEqual(challenge.index, 1)

    def test_single_frame_cannot_complete_step(self):
        challenge = self.challenge()
        challenge.update(pose(), 0)
        challenge.update(pose(25), .3)
        self.assertEqual(challenge.index, 0)

    def test_flat_image_translation_is_not_rotation(self):
        challenge = self.challenge()
        for i in range(40):
            challenge.update(pose(box=(BOX[0] + i, *BOX[1:])), i * .1)
        self.assertEqual(challenge.index, 1)

    def test_image_roll_is_not_yaw(self):
        challenge = self.challenge()
        for i in range(4):
            challenge.update(pose(), i * .15)
        for i in range(4, 20):
            challenge.update(pose(25, roll=30), i * .15)
        self.assertEqual(challenge.index, 1)

    def test_loss_gap_jump_and_invalid_values_fail_closed(self):
        for observation, timestamp in ((None, .2), (pose(), 1),
                (pose(box=(0, 0, 30, 30)), .2), (pose(float('nan')), .2)):
            with self.subTest(observation=observation):
                challenge = self.challenge()
                challenge.update(pose(), 0)
                challenge.update(observation, timestamp)
                self.assertIsNotNone(challenge.failure)
                self.assertFalse(challenge.complete)

    def test_configuration_rejects_nan_unknown_or_inconsistent_limits(self):
        for changes in ({'total_timeout': float('nan')}, {'min_samples': 1.5},
                        {'turn_yaw': 5}, {'embedding_interval': 1}, {'yaw_sign': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                LivenessSettings(**changes)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = initialize_database(Path(self.temp.name) / "faces.sqlite3")
        self.embedding = np.eye(1, 128, dtype=np.float32).reshape(-1)
        add_user(self.store, "owner", [self.embedding] * REGISTRATION_SAMPLES)
        self.engine = Mock()
        self.engine.detect.return_value = np.array([detection()])
        self.engine.analyze.return_value = [Face(BOX, self.embedding.copy(), 100)]
        self.mesh = Mock()
        self.mesh.observe.return_value = pose()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.now = 0.
        self.pipeline = FacePipeline(self.engine, FaceWatcher(GateSettings(hold_seconds=0)),
            self.store, mesh=self.mesh, start_in_standby=False)
        self.addCleanup(self.pipeline.close)

    def process(self, yaw=0, advance=.15):
        self.now += advance
        self.mesh.observe.return_value = pose(yaw)
        with patch('pipeline.monotonic', return_value=self.now), \
                patch('standby.time.monotonic', return_value=self.now):
            return self.pipeline.process(self.frame)

    def complete(self):
        result = self.process()
        for _ in range(50):
            challenge = self.pipeline.challenge
            if challenge.complete:
                return result
            result = self.process(challenge.steps[challenge.index] * 25)
        self.fail("challenge did not terminate")

    def test_known_frontal_face_alone_never_unlocks_or_saves(self):
        result = self.process()
        self.assertFalse(result.authentication_attempted)
        self.assertIsNone(result.authenticated_name)
        self.assertEqual(self.store.count_observations(), 0)

    def test_complete_challenge_authorizes_exactly_once(self):
        result = self.complete()
        self.assertTrue(result.authentication_attempted)
        self.assertTrue(result.liveness_passed)
        self.assertEqual(result.authenticated_name, 'owner')
        self.assertEqual(self.store.count_observations(), 1)
        for _ in range(3):
            replay = self.process()
            self.assertFalse(replay.authentication_attempted)
            self.assertIsNone(replay.authenticated_name)
        self.assertEqual(self.store.count_observations(), 1)

    def test_turn_is_not_blocked_by_standby_front_gate(self):
        self.process()
        self.process()
        self.process()
        self.assertEqual(self.pipeline.challenge.index, 1)
        for _ in range(3):
            self.process(self.pipeline.challenge.steps[1] * 25)
        self.assertEqual(self.pipeline.challenge.index, 2)
        self.assertEqual(self.pipeline.mode, 'active')

    def test_embedding_is_throttled_during_motion(self):
        self.process(advance=.05)
        initial = self.engine.analyze.call_count
        for _ in range(3):
            self.process(advance=.05)
        self.assertEqual(self.engine.analyze.call_count, initial)

    def test_tracking_loss_and_multiple_faces_emit_one_failure(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self.pipeline.reset_standby()
                self.pipeline.mode = 'active'
                self.engine.detect.return_value = np.array([detection()])
                self.process()
                self.engine.detect.return_value = np.array([detection()] * count).reshape(count, 15)
                result = self.process()
                self.assertTrue(result.authentication_attempted)
                self.assertFalse(result.liveness_passed)
                self.assertIsNone(result.authenticated_name)
        self.assertEqual(self.store.count_observations(), 0)

    def test_second_registered_person_cannot_finish_first_person_challenge(self):
        other = np.roll(self.embedding, 1)
        add_user(self.store, 'other', [other] * REGISTRATION_SAMPLES)
        self.process()
        self.engine.analyze.return_value = [Face(BOX, other, 100)]
        result = self.process(advance=.4)
        self.assertTrue(result.authentication_attempted)
        self.assertIn('변경', result.status)
        self.assertEqual(self.store.count_observations(), 0)

    def test_deleted_identity_cannot_finish(self):
        self.process()
        delete_user(self.store, 'owner')
        result = self.process(advance=.4)
        self.assertTrue(result.authentication_attempted)
        self.assertFalse(result.liveness_passed)

    def test_unknown_face_rejected_before_challenge(self):
        self.engine.analyze.return_value = [Face(BOX, np.roll(self.embedding, 1), 100)]
        result = self.process()
        self.assertTrue(result.authentication_attempted)
        self.assertIsNone(self.pipeline.challenge)

    def test_timeout_emits_failure_instead_of_waiting_forever(self):
        self.process()
        result = self.process(advance=8)
        self.assertTrue(result.authentication_attempted)
        self.assertIn('시간 초과', result.status)

    def test_standby_never_runs_mesh_or_embedding(self):
        self.pipeline.reset_standby()
        self.pipeline.watcher = FaceWatcher(GateSettings(hold_seconds=2))
        self.process()
        self.mesh.observe.assert_not_called()
        self.engine.analyze.assert_not_called()

    def test_enrollment_preview_do_not_authenticate_or_write_observations(self):
        for purpose in ('enroll', 'preview'):
            self.pipeline.set_purpose(purpose, 4)
            self.pipeline.mode = 'active'
            result = self.process()
            self.assertTrue(result.ready)
            self.assertTrue(result.faces)
            self.assertFalse(result.authentication_attempted)
            self.assertEqual(result.session_id, 4)
        self.mesh.observe.assert_not_called()
        self.assertEqual(self.store.count_observations(), 0)

    def test_no_users_does_not_latch_a_failure_or_load_mesh(self):
        delete_user(self.store, 'owner')
        result = self.process()
        self.assertTrue(result.ready)
        self.assertFalse(result.authentication_attempted)
        self.mesh.observe.assert_not_called()

    def test_mode_threshold_and_reset_clear_old_challenge(self):
        self.process()
        self.pipeline.threshold = .6
        self.assertIsNone(self.pipeline.challenge)
        self.assertEqual(self.pipeline.mode, 'standby')
        self.pipeline.set_purpose('preview', 7)
        self.assertEqual(self.pipeline.session_id, 7)
        self.assertIsNone(self.pipeline.identity)

    def test_departure_only_detects_and_never_wakes(self):
        self.pipeline.set_purpose('departure')
        for _ in range(4):
            self.process()
        self.assertEqual(self.pipeline.mode, 'standby')
        self.engine.analyze.assert_not_called()
        self.mesh.observe.assert_not_called()

    def test_mesh_disagreement_does_not_authorize(self):
        self.process()
        self.mesh.observe.return_value = None
        self.now += .15
        with patch('pipeline.monotonic', return_value=self.now):
            result = self.pipeline.process(self.frame)
        self.assertTrue(result.authentication_attempted)
        self.assertFalse(result.liveness_passed)


if __name__ == '__main__':
    unittest.main()
