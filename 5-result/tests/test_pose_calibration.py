"""User-labelled asymmetric four-pose calibration and verification without manual thresholds."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stages'))

import numpy as np

from liveness import Pose
from pose_calibration import ORDER, PoseCalibration, PoseProfile

BOX = (100, 100, 200, 250)
CENTERS = {'front': (5., 10., 2.), 'left': (-25., 11., 2.), 'right': (65., 9., 3.),
           'up': (7., -15., 1.), 'down': (3., 35., 2.)}


def pose(name):
    return Pose(*CENTERS[name], BOX)


class PoseCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.flow = PoseCalibration()
        self.now = 0.

    def feed(self, name):
        for _ in range(5):
            self.now += .05
            self.flow.observe(pose(name), self.now)

    def complete(self):
        self.flow.start()
        for name in ORDER:
            self.assertEqual(self.flow.target, name)
            self.feed(name)
            self.assertEqual(self.flow.capture(self.now), name)
        return self.flow.profile

    def test_order_is_front_left_right_up_and_each_p_records_its_own_reference(self):
        self.assertEqual(ORDER, ('front', 'left', 'right', 'up'))
        profile = self.complete()
        self.assertEqual(self.flow.mode, 'ready')
        self.assertIsNone(self.flow.target)
        np.testing.assert_array_equal(profile.centers, [CENTERS[name] for name in ORDER])

    def test_side_asymmetry_and_camera_bias_are_learned_without_fixed_signed_thresholds(self):
        profile = self.complete()
        for name in ORDER:
            self.assertEqual(profile.classify(pose(name))[0], name)
        self.assertEqual(profile.centers[2, 0]-profile.centers[0, 0], 60.)
        self.assertEqual(profile.centers[0, 0]-profile.centers[1, 0], 30.)

    def test_repeated_p_cannot_reuse_frames_and_does_not_skip_directions(self):
        self.flow.start()
        self.feed('front')
        self.flow.capture(self.now)
        with self.assertRaises(ValueError):
            self.flow.capture(self.now)
        self.assertEqual(self.flow.target, 'left')
        self.assertEqual(len(self.flow.records), 1)

    def test_stale_missing_and_unstable_samples_cannot_be_recorded(self):
        self.flow.start()
        self.feed('front')
        with self.assertRaises(ValueError):
            self.flow.capture(self.now+1)
        self.flow.observe(None, self.now+.1)
        with self.assertRaises(ValueError):
            self.flow.capture(self.now+.1)
        for i in range(5):
            self.now += .05
            self.flow.observe(Pose(i*10., 0, 0, BOX), self.now)
        with self.assertRaises(ValueError):
            self.flow.capture(self.now)
        self.assertEqual(self.flow.records, {})

    def test_same_pose_under_a_different_request_is_rejected_without_advancing(self):
        self.flow.start()
        self.feed('front')
        self.flow.capture(self.now)
        self.feed('front')
        with self.assertRaisesRegex(ValueError, '비슷'):
            self.flow.capture(self.now)
        self.assertEqual(self.flow.target, 'left')

    def test_previous_step_replaces_only_the_last_record_and_requires_new_frames(self):
        self.flow.start()
        for name in ('front', 'left'):
            self.feed(name)
            self.flow.capture(self.now)
        self.assertTrue(self.flow.back())
        self.assertEqual(self.flow.target, 'left')
        self.assertEqual(list(self.flow.records), ['front'])
        with self.assertRaises(ValueError):
            self.flow.capture(self.now)

    def test_incomplete_or_degenerate_five_pose_sets_cannot_start_a_test(self):
        with self.assertRaises(ValueError):
            self.flow.start_test()
        with self.assertRaises(ValueError):
            PoseProfile(np.zeros((5, 3)))
        with self.assertRaisesRegex(ValueError, '상하'):
            PoseProfile([[0, 0, 0], [-30, 0, 0], [60, 0, 0], [15, 0, 0], [45, 0, 0]])

    def test_midpoints_extreme_angles_roll_and_missing_faces_are_not_mapped_to_a_false_pass(self):
        profile = self.complete()
        for unknown in (None, Pose(-10, 10.5, 2, BOX), Pose(150, 120, 2, BOX), Pose(5, 10, 40, BOX)):
            self.assertIsNone(profile.classify(unknown)[0])

    def test_verification_requires_all_four_targets_and_fresh_hold_for_each(self):
        self.complete()
        self.flow.start_test()
        for target in ORDER:
            self.assertEqual(self.flow.target, target)
            for step in range(6):
                self.now += .1
                passed = self.flow.test_frame(pose(target), self.now)
                if passed:
                    break
            self.assertTrue(passed)
        self.assertEqual(self.flow.results, list(ORDER))
        self.assertEqual(self.flow.mode, 'tested')

    def test_wrong_pose_face_loss_or_gap_resets_test_hold(self):
        self.complete()
        self.flow.start_test()
        self.assertFalse(self.flow.test_frame(pose('front'), 1.))
        self.assertFalse(self.flow.test_frame(pose('left'), 1.2))
        self.assertFalse(self.flow.test_frame(pose('front'), 1.4))
        self.assertFalse(self.flow.test_frame(None, 1.5))
        self.assertFalse(self.flow.test_frame(pose('front'), 1.6))
        self.assertFalse(self.flow.test_frame(pose('front'), 3.))
        self.assertEqual(self.flow.results, [])
        self.assertEqual(self.flow.hold_frames, 1)

    def test_profile_persists_only_angles_and_requires_matching_capture_context(self):
        profile = self.complete()
        context = dict(camera=0, size=[1920, 1080], stabilized=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pose_calibration.json'
            profile.save(path, context)
            loaded = PoseProfile.load(path, context)
            self.assertEqual(loaded.classify(pose('right'))[0], 'right')
            self.assertEqual(set(json.loads(path.read_text(encoding='utf-8'))), {'version', 'directions', 'centers', 'context'})
            with self.assertRaises(ValueError):
                PoseProfile.load(path, dict(context, camera=1))
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])
            path.write_text('[]', encoding='utf-8')
            with self.assertRaises(ValueError):
                PoseProfile.load(path, context)

    def test_legacy_five_pose_file_is_preserved_but_down_is_never_accepted(self):
        directions = [*ORDER, 'down']
        centers = [CENTERS[name] for name in directions]
        context = dict(camera=0, size=[1920, 1080], stabilized=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.json'
            path.write_text(json.dumps(dict(version=1, directions=directions, centers=centers, context=context)),
                            encoding='utf-8')
            original = path.read_bytes()
            loaded = PoseProfile.load(path, context)
            self.assertEqual(path.read_bytes(), original)
            np.testing.assert_array_equal(loaded.centers, centers)
            for name in ORDER:
                self.assertEqual(loaded.classify(pose(name))[0], name)
            self.assertIsNone(loaded.classify(pose('down'))[0])

    def test_new_four_pose_profile_rejects_down_and_mismatched_direction_count(self):
        profile = self.complete()
        self.assertEqual(profile.centers.shape, (4, 3))
        self.assertIsNone(profile.classify(pose('down'))[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            profile.save(path, {})
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['directions'], list(ORDER))
            payload['directions'].append('down')
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                PoseProfile.load(path, {})

    def test_invalid_profile_or_failed_write_cannot_replace_previous_saved_calibration(self):
        with self.assertRaises(ValueError):
            PoseProfile(np.full((5, 3), 1000.))
        profile = self.complete()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pose_calibration.json'
            profile.save(path, {})
            previous = path.read_bytes()
            with patch('pose_calibration.os.replace', side_effect=OSError('unavailable')):
                with self.assertRaises(OSError):
                    profile.save(path, {})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_cancel_collect_discards_partial_records_and_cancel_test_keeps_completed_profile(self):
        self.flow.start()
        self.feed('front')
        self.flow.capture(self.now)
        self.flow.cancel()
        self.assertEqual(self.flow.records, {})
        self.assertIsNone(self.flow.profile)
        self.complete()
        profile = self.flow.profile
        self.flow.start_test()
        self.flow.cancel()
        self.assertEqual(self.flow.mode, 'ready')
        self.assertIs(self.flow.profile, profile)


if __name__ == '__main__':
    unittest.main()
