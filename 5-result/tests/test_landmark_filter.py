"""Point noise, outlier rejection, motion latency and stale-coordinate regressions."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stages'))

import numpy as np

from landmark_filter import LandmarkFilter, LandmarkFilterSettings


def face():
    return np.array([100, 100, 200, 250, 150, 165, 250, 165, 200, 220,
                     160, 285, 240, 285, .99], dtype=float)


class LandmarkFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = LandmarkFilter()

    def update(self, row, now, size=(640, 480)):
        return self.filter.update([row], size, now)

    def warm(self):
        for index in range(3):
            result = self.update(face(), index*.05)
        return result

    def test_no_pose_coordinates_until_three_fresh_frames_and_input_is_not_mutated(self):
        row = face()
        before = row.copy()
        self.assertIsNone(self.update(row, 0).row)
        self.assertIsNone(self.update(row, .05).row)
        result = self.update(row, .1)
        np.testing.assert_allclose(result.row, before)
        np.testing.assert_array_equal(row, before)
        self.assertIsNot(result.row, row)

    def test_stationary_point_noise_is_reduced_before_pose_solving(self):
        rng = np.random.default_rng(42)
        raw, filtered = [], []
        for index in range(400):
            row = face()
            row[4:14] += rng.normal(0, 2., 10)
            result = self.update(row, index*.05)
            if result.row is not None and index > 30:
                raw.append(row[4:14]-face()[4:14])
                filtered.append(result.row[4:14]-face()[4:14])
        raw_rms = np.sqrt(np.mean(np.square(raw)))
        filtered_rms = np.sqrt(np.mean(np.square(filtered)))
        self.assertLess(filtered_rms, raw_rms*.65)

    def test_one_point_spike_is_excluded_and_does_not_reuse_old_valid_output(self):
        self.warm()
        spike = face()
        spike[8] += 40
        result = self.update(spike, .15)
        self.assertIsNone(result.row)
        self.assertIn('제외', result.reason)
        self.assertIsNone(self.update(face(), .2).row)
        self.assertIsNone(self.update(face(), .25).row)
        np.testing.assert_allclose(self.update(face(), .3).row, face())

    def test_sustained_large_jumps_reset_and_require_new_baseline(self):
        self.warm()
        jump = face()
        jump[8] += 40
        for now in (.15, .2):
            self.assertIsNone(self.update(jump, now).row)
        result = self.update(jump, .25)
        self.assertIsNone(result.row)
        self.assertTrue(result.reset_baseline)

    def test_missing_multiple_low_confidence_and_invalid_points_clear_all_coordinates(self):
        invalid = face()
        invalid[8] = np.nan
        low = face()
        low[14] = .6
        for rows in ([], [face(), face()], [invalid], [low]):
            self.filter.reset()
            self.warm()
            result = self.filter.update(rows, (640, 480), .15)
            self.assertIsNone(result.row)
            self.assertTrue(result.reset_baseline)
            self.assertIsNone(self.filter.smoothed)
            self.assertIsNone(self.filter.previous)
            self.assertIsNone(self.update(face(), .2).row)

    def test_capture_gap_duplicate_timestamp_resolution_and_face_jump_reset(self):
        for mode in ('gap', 'duplicate', 'resolution', 'new_face'):
            self.filter.reset()
            self.warm()
            row, size, now = face(), (640, 480), .15
            if mode == 'gap':
                now = 1.
            elif mode == 'duplicate':
                now = .1
            elif mode == 'resolution':
                size = (1920, 1080)
            else:
                row[0] += 240
                row[4:14:2] += 240
            result = self.update(row, now, size)
            self.assertIsNone(result.row, mode)
            self.assertTrue(result.reset_baseline, mode)

    def test_continuous_intended_motion_is_followed_without_freezing_or_overshooting(self):
        self.warm()
        result = None
        for index in range(1, 21):
            row = face()
            row[0] += index*3
            row[4:14:2] += index*3
            result = self.update(row, .1+index*.05)
            self.assertIsNotNone(result.row)
            self.assertFalse(result.reset_baseline)
            self.assertLessEqual(result.row[4], row[4]+1e-6)
        self.assertLess(row[4]-result.row[4], 15.)

    def test_filter_behaves_the_same_at_vga_and_fhd_coordinate_scales(self):
        other = LandmarkFilter()
        rng = np.random.default_rng(12)
        scaling = np.array([3, 2.25]*7+[1])
        for index in range(30):
            row = face()
            row[4:14] += rng.normal(0, 1, 10)
            first = self.update(row, index*.05)
            second = other.update([row*scaling], (1920, 1080), index*.05)
            if first.row is not None:
                np.testing.assert_allclose(second.row/scaling, first.row, atol=.1)

    def test_invalid_filter_settings_are_rejected(self):
        for options in ({'smoothing_seconds': float('nan')}, {'jump_ratio': 0},
                        {'min_confidence': 2}, {'warmup_frames': 1}, {'max_gap': float('inf')}):
            with self.assertRaises(ValueError):
                LandmarkFilterSettings(**options)


if __name__ == '__main__':
    unittest.main()
