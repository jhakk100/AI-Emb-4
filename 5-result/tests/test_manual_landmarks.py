"""Manual point correspondence, offset scaling and atomic edit/cancel behavior."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stages'))

import numpy as np

from manual_landmarks import ManualLandmarks
from test_landmark_filter import face


class ManualLandmarksTests(unittest.TestCase):
    def setUp(self):
        self.manual = ManualLandmarks()

    def test_commit_applies_only_edited_points_without_mutating_detection(self):
        row = face()
        before = row.copy()
        self.manual.begin(row, (640, 480))
        self.manual.move(2, 210.25, 225.75)
        self.assertFalse(self.manual.active)
        self.manual.commit()
        adjusted = self.manual.apply(row, (640, 480))
        np.testing.assert_array_equal(row, before)
        np.testing.assert_allclose(adjusted[8:10], [210.25, 225.75])
        np.testing.assert_array_equal(adjusted[:8], row[:8])
        np.testing.assert_array_equal(adjusted[10:], row[10:])
        self.assertFalse(self.manual.editing)

    def test_offsets_follow_face_translation_and_scale_instead_of_fixed_screen_pixels(self):
        row = face()
        self.manual.begin(row, (640, 480))
        self.manual.move(2, 210, 225)
        self.manual.commit()
        moved = row.copy()
        moved[:4] *= 1.5
        moved[:2] += [15, 20]
        moved[4:14:2] = moved[4:14:2]*1.5+15
        moved[5:14:2] = moved[5:14:2]*1.5+20
        adjusted = self.manual.apply(moved, (960, 720))
        np.testing.assert_allclose(adjusted[8:10]-moved[8:10], [15, 7.5])

    def test_cancelling_reedit_preserves_existing_offsets_and_does_not_accumulate(self):
        row = face()
        self.manual.begin(row, (640, 480))
        self.manual.move(2, 210, 225)
        self.manual.commit()
        expected = self.manual.apply(row, (640, 480))
        self.manual.begin(row, (640, 480))
        self.manual.commit()
        np.testing.assert_array_equal(self.manual.apply(row, (640, 480)), expected)
        self.manual.begin(row, (640, 480))
        self.manual.move(2, 215, 226)
        self.manual.cancel()
        np.testing.assert_array_equal(self.manual.apply(row, (640, 480)), expected)
        self.manual.clear()
        np.testing.assert_array_equal(self.manual.apply(row, (640, 480)), row)

    def test_crossed_eyes_invalid_layout_and_excessive_offset_cannot_be_committed(self):
        row = face()
        for index, point in ((0, (255, 165)), (2, (200, 150)), (2, (300, 220))):
            self.manual.begin(row, (640, 480))
            self.manual.move(index, *point)
            with self.assertRaises(ValueError):
                self.manual.commit()
            self.assertTrue(self.manual.editing)
            self.assertFalse(self.manual.active)

    def test_dragging_clips_to_native_image_and_nonfinite_positions_are_rejected(self):
        self.manual.begin(face(), (640, 480))
        self.manual.move(2, -20, 900)
        np.testing.assert_array_equal(self.manual.points[2], [0, 479])
        with self.assertRaises(ValueError):
            self.manual.move(2, float('nan'), 20)

    def test_screen_left_right_labels_keep_the_original_point_correspondence(self):
        row = face()
        row[4:8] = row[4:8].reshape(2, 2)[::-1].reshape(-1)
        row[10:14] = row[10:14].reshape(2, 2)[::-1].reshape(-1)
        self.manual.begin(row, (640, 480))
        self.assertEqual(self.manual.labels(), ('오른쪽 눈', '왼쪽 눈', '코끝', '오른쪽 입꼬리', '왼쪽 입꼬리'))


if __name__ == '__main__':
    unittest.main()
