import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from face_core import Enrollment, Face, FaceStore, REGISTRATION_SAMPLES, normalize


def vector(index=0):
    result = np.zeros(128, dtype=np.float32)
    result[index] = 1
    return result


class FaceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "data" / "embeddings.json"
        self.store = FaceStore(self.path)

    def enroll(self, name="내 얼굴", index=0):
        self.store.enroll(name, [vector(index)] * REGISTRATION_SAMPLES)

    def test_round_trip_korean_name_and_normalized_embedding(self):
        self.enroll()
        loaded = FaceStore(self.path)
        self.assertEqual(loaded.match(vector()), ("내 얼굴", 1.0))
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["people"]["내 얼굴"]["embedding"]), 128)

    def test_unknown_person_is_rejected(self):
        self.enroll()
        self.assertEqual(self.store.match(vector(1)), (None, 0.0))

    def test_best_match_with_multiple_people(self):
        self.enroll("가", 0)
        self.enroll("나", 1)
        self.assertEqual(self.store.match(vector(1)), ("나", 1.0))

    def test_threshold_changes_decision(self):
        self.enroll()
        query = normalize(vector() * 0.5 + vector(1) * np.sqrt(0.75))
        self.assertEqual(self.store.match(query, 0.45)[0], "내 얼굴")
        self.assertIsNone(self.store.match(query, 0.6)[0])

    def test_delete_survives_reload(self):
        self.enroll()
        self.store.delete("내 얼굴")
        self.assertEqual(FaceStore(self.path).match(vector()), (None, None))

    def test_reenroll_replaces_existing_embedding(self):
        self.enroll()
        self.enroll(index=1)
        self.assertEqual(len(self.store.people), 1)
        self.assertEqual(self.store.match(vector()), (None, 0.0))
        self.assertEqual(self.store.match(vector(1)), ("내 얼굴", 1.0))

    def test_failed_write_keeps_memory_and_file_unchanged(self):
        self.enroll()
        before = self.path.read_bytes()
        with patch("face_core.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.enroll(index=1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.store.match(vector())[0], "내 얼굴")
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_corrupt_database_is_never_silently_overwritten(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("broken json", encoding="utf-8")
        with self.assertRaises(ValueError):
            FaceStore(self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "broken json")

    def test_different_model_is_rejected(self):
        self.enroll()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["model"] = "different-model"
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            FaceStore(self.path)

    def test_invalid_embedding_is_rejected(self):
        for invalid in (np.zeros(128), np.ones(127), np.full(128, np.nan), np.full(128, np.inf)):
            with self.subTest(invalid=invalid.shape), self.assertRaises(ValueError):
                normalize(invalid)

    def test_incomplete_registration_is_not_saved(self):
        with self.assertRaises(ValueError):
            self.store.enroll("me", [vector()])
        self.assertFalse(self.path.exists())


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.enrollment = Enrollment("me")
        self.face = Face((0, 0, 120, 120), vector(), 100)

    def test_no_face_or_multiple_faces_not_collected(self):
        for faces in ([], [self.face, self.face]):
            self.enrollment.accept(faces, 1)
        self.assertEqual(len(self.enrollment.samples), 0)

    def test_small_or_blurry_face_not_collected(self):
        for face in (Face((0, 0, 60, 60), vector(), 100), Face((0, 0, 120, 120), vector(), 5)):
            self.enrollment.accept([face], 1)
        self.assertEqual(len(self.enrollment.samples), 0)

    def test_sample_interval_and_target(self):
        self.enrollment.accept([self.face], 0)
        self.enrollment.accept([self.face], 0.1)
        self.assertEqual(len(self.enrollment.samples), 1)
        for i in range(1, 30):
            self.enrollment.accept([self.face], i * 0.31)
        self.assertEqual(len(self.enrollment.samples), REGISTRATION_SAMPLES)

    def test_other_person_is_not_mixed_into_registration(self):
        self.enrollment.accept([self.face], 0)
        other = Face((0, 0, 120, 120), vector(1), 100)
        self.enrollment.accept([other], 1)
        self.assertEqual(len(self.enrollment.samples), 1)


if __name__ == "__main__":
    unittest.main()
