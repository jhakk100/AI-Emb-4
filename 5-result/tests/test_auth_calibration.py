"""Verified observations update the next session's frontal threshold without relabeling attempts."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stages'))

import numpy as np

from auth_calibration import MIN_CONFIRMED, equal_error_threshold
from database import initialize_database
from face_core import Face
from liveness import Pose
from pipeline import FacePipeline
from standby import FaceWatcher, GateSettings
from test_secure_pipeline import BOX, detection, profile


def embedding(score, first=0, second=1):
    vector = np.zeros(128, dtype=np.float32)
    vector[first], vector[second] = score, np.sqrt(1-score**2)
    return vector


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / 'test.sqlite3'
        self.store = initialize_database(self.path)
        self.store.enroll_directional('owner', profile(embedding(1.)))
        self.calibration = self.store.calibration

    def attempt(self, score, truth=None, *, outcome='rejected'):
        attempt = self.calibration.begin('test', .45, diagnostic=True)
        if score is not None:
            self.calibration.record_scores(attempt, embedding(score))
            self.calibration.record_scores(attempt, embedding(.99))  # later frames cannot replace it
        self.calibration.finish(attempt, outcome, 'test')
        if truth:
            self.calibration.review(attempt.id, truth, 'owner' if truth == 'registered' else None)
        return attempt

    def test_discrete_equal_error_crossing(self):
        threshold, far, frr = equal_error_threshold([.51, .80, .90], [.20, .45, .61])
        self.assertAlmostEqual(threshold, .56)
        self.assertAlmostEqual(far, 1/3)
        self.assertAlmostEqual(frr, 1/3)
        threshold, far, frr = equal_error_threshold([.80, .90], [.20, .40])
        self.assertAlmostEqual(threshold, .60)
        self.assertEqual((far, frr), (0., 0.))

    def test_only_independently_tagged_scores_update_the_next_attempt(self):
        for _ in range(MIN_CONFIRMED):
            self.attempt(.8, 'registered')
        self.attempt(None, 'visitor', outcome='interrupted')
        self.attempt(.5, outcome='accepted')
        previous = self.calibration.begin('test', .45)
        self.assertEqual(self.calibration.status()['impostor'], 0)
        self.assertEqual(previous.threshold, .45)
        for _ in range(MIN_CONFIRMED):
            last = self.attempt(.5, 'visitor')
        self.assertEqual(previous.threshold, .45)
        reopened = initialize_database(self.path).calibration
        active = reopened.begin('test', .45)
        self.assertAlmostEqual(active.threshold, .65)
        self.assertEqual(active.threshold_source, 'eer')
        self.assertEqual(reopened.status()['far'], 0.)
        self.assertEqual(reopened.status()['frr'], 0.)
        self.assertEqual(reopened.recent_attempts()[1]['score_candidates'], {'owner': .5})
        reopened.review(last.id, None)
        self.assertIsNone(reopened.status()['threshold'])
        self.assertEqual(reopened.begin('test', .45).threshold, .45)

    def test_registered_team_members_provide_real_cross_identity_scores(self):
        self.store.enroll_directional('team', profile(embedding(1., 1, 0)))
        for _ in range(MIN_CONFIRMED):
            self.attempt(.9, 'registered')
        current = self.calibration.status()
        self.assertEqual((current['genuine'], current['impostor']), (MIN_CONFIRMED, MIN_CONFIRMED))
        self.assertGreater(current['threshold'], .45)
        self.calibration.set_enabled(False)
        self.assertEqual(self.calibration.begin('test', .45).threshold, .45)
        self.calibration.set_enabled(True)
        self.store.enroll_directional('team', profile(embedding(1., 1, 0)))
        self.assertIsNone(self.calibration.status()['threshold'])
        self.assertEqual(self.calibration.begin('test', .45).threshold, .45)

    def test_real_pipeline_uses_calibrated_front_cut_and_saves_rejected_score(self):
        for _ in range(MIN_CONFIRMED):
            self.attempt(.8, 'registered')
            self.attempt(.5, 'visitor')
        engine = Mock()
        engine.detect.return_value = np.array([detection()])
        engine.analyze.return_value = [Face(BOX, embedding(.58), 100)]
        pipeline = FacePipeline(engine, store=self.store, start_in_standby=False,
                                watcher=FaceWatcher(GateSettings(hold_seconds=0)),
                                pose_estimator=lambda *_: Pose(0., 0., 0., BOX))
        self.addCleanup(pipeline.close)
        with patch('standby.time.monotonic', return_value=1.):
            result = pipeline.process(np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(result.authentication_attempted)
        self.assertIsNone(result.authenticated_name)
        self.assertAlmostEqual(result.front_threshold, .65)
        self.assertEqual(pipeline.threshold, .45)  # downstream facial comparisons retain their own cutoff
        row = self.calibration.recent_attempts()[0]
        self.assertEqual(row['id'], result.auth_attempt_id)
        self.assertEqual(row['outcome'], 'rejected')
        self.assertAlmostEqual(row['score_candidates']['owner'], .58)
        self.assertIsNone(row['verified_at'])

    def test_started_diagnostic_can_remain_unscored_until_explicitly_ended(self):
        engine = Mock()
        pipeline = FacePipeline(engine, store=self.store, pose_estimator=lambda *_: None)
        self.addCleanup(pipeline.close)
        pipeline.begin_auth_attempt(diagnostic=True)
        attempt_id = pipeline.auth_attempt.id
        pipeline.reset_standby()
        self.assertEqual(pipeline.auth_attempt.id, attempt_id)
        pipeline.finish_auth_attempt('interrupted', '사용자 중단')
        row, = self.calibration.recent_attempts()
        self.assertTrue(row['diagnostic'])
        self.assertEqual(row['outcome'], 'interrupted')
        self.assertEqual(row['score_candidates'], {})
