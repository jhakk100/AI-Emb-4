"""Directional grids and fused 3D models must both pass, atomically, before authentication."""

from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stages'))

import numpy as np
import cv2

from database import initialize_database, add_user, delete_user
from face_grid import (GRID_MODEL, FUSED_MODEL, MeshPoseGuide, build_grids, fuse_grids, compare_grid, compare_fused,
                       grid_triangles, grid_edges, render_grids, render_fused_model, draw_mesh_preview)
from guided_capture import (GuidedSettings, MANUAL_POSE_BASIS, MESH_AUTH_POSE_KIND,
                            mesh_auth_pose_profile)
from liveness import Pose
from pose_calibration import PoseProfile
from test_secure_pipeline import profile, sample, landmarks

ANGLE_CENTERS = [(0., 0., 0.), (-30., -17., 2.), (34., -12., -5.), (1., -24., 0.)]


class FaceGridTests(unittest.TestCase):
    def setUp(self):
        self.settings = GuidedSettings()
        self.samples = profile()
        self.grids = build_grids(self.samples, self.settings)
        self.model = fuse_grids(self.grids)

    def test_four_distinct_grids_and_one_fused_triangular_model_are_built(self):
        self.assertEqual(set(self.grids), {'front', 'left', 'right', 'up'})
        for grid in self.grids.values():
            self.assertEqual(grid['vertices'].shape, (468, 3))
            self.assertEqual(grid['samples'], 3)
            self.assertEqual(grid['grid_model'], GRID_MODEL)
        self.assertEqual(self.model['vertices'].shape, (468, 3))
        self.assertEqual(self.model['model'], FUSED_MODEL)
        self.assertEqual(len(grid_triangles()), 852)
        self.assertGreater(len(grid_edges()), 1000)
        self.assertEqual(self.model['directions'], 4)
        self.assertTrue(np.isfinite(self.model['vertices']).all())

    def test_same_person_passes_grid_and_fused_shape_comparison(self):
        for query in self.samples:
            self.assertIsNotNone(compare_grid(query, self.grids[query.direction], .45, self.settings))
            self.assertIsNotNone(compare_fused(query, self.model, .45, self.settings))

    def test_only_front_and_side_pitch_windows_are_widened(self):
        original = PoseProfile(ANGLE_CENTERS)
        relaxed = mesh_auth_pose_profile(ANGLE_CENTERS, self.settings)
        box = sample().pose.box
        for direction, angles in (('front', (0., -10., 0.)), ('left', (-27., -11., 0.)),
                                  ('right', (34., -4., 0.))):
            pose = Pose(*angles, box)
            self.assertIsNone(original.classify(pose)[0])
            self.assertEqual(relaxed.classify(pose)[0], direction)
        for angles in ((1., -20., 0.), (1., -17.5, 0.), (-49., -17., 0.), (-30., -17., 16.)):
            pose = Pose(*angles, box)
            self.assertEqual(original.classify(pose)[0], relaxed.classify(pose)[0])

    def test_relaxed_front_cannot_move_later_pitch_origin_outside_original_front_window(self):
        guide = MeshPoseGuide.__new__(MeshPoseGuide)
        guide.profile = mesh_auth_pose_profile(ANGLE_CENTERS, self.settings)
        original = PoseProfile(ANGLE_CENTERS)
        box = sample().pose.box
        guide.set_neutral(Pose(1., -3., 2., box))
        self.assertEqual(guide.neutral, (1., -3., 2.))
        tilted = Pose(1., -10., 2., box)
        self.assertEqual(guide.profile.classify(tilted)[0], 'front')
        guide.set_neutral(tilted)
        yaw, pitch, roll = guide.neutral
        self.assertEqual((yaw, roll), (tilted.yaw, tilted.roll))
        self.assertGreater(pitch, tilted.pitch)
        _, distances = original.distances(Pose(yaw, pitch, roll, box))
        self.assertAlmostEqual(distances[0], original.radii[0])

    def test_correction_hint_distinguishes_pitch_turn_and_roll_without_passing_the_pose(self):
        guide = MeshPoseGuide.__new__(MeshPoseGuide)
        guide.profile = mesh_auth_pose_profile(ANGLE_CENTERS, self.settings)
        box = sample().pose.box
        index = guide.profile.directions.index('left')
        yaw, pitch = guide.profile.centers[index, :2]
        radius = guide.profile.radii[index]
        yaw_edge = radius * guide.profile.scale[0]
        pitch_edge = radius * guide.profile.scale[1] * guide.profile.pitch_scales[index]
        cases = (
            (Pose(yaw, pitch+pitch_edge*1.1, 0., box), 'raise_chin'),
            (Pose(yaw, pitch-pitch_edge*1.1, 0., box), 'lower_chin'),
            (Pose(yaw-yaw_edge*1.1, pitch, 0., box), 'turn_right'),
            (Pose(yaw, pitch, 17., box), 'level_head'),
            (Pose(yaw, pitch+pitch_edge*2, 0., box), None),
            (Pose(0., 0., 0., box), None),
        )
        for pose, expected in cases:
            with self.subTest(expected=expected):
                self.assertFalse(guide.matches('left', pose))
                self.assertEqual(guide.correction_hint('left', pose), expected)
        accepted = Pose(yaw, pitch, 0., box)
        self.assertTrue(guide.matches('left', accepted))
        self.assertIsNone(guide.correction_hint('left', accepted))

    def test_each_grid_uses_its_own_direction_reference_before_fusing(self):
        for index, item in enumerate(self.samples):
            rotation = cv2.Rodrigues(np.array([.03*index, -.05*index, .02*index]))[0]
            item.mesh = item.mesh @ rotation.T
        grids = build_grids(list(reversed(self.samples)), self.settings)
        for direction, grid in grids.items():
            reference = next(item.mesh for item in self.samples if item.direction == direction and item.index == 0)
            np.testing.assert_allclose(grid['vertices'], reference, atol=1e-6)
        self.assertGreater(np.linalg.norm(grids['left']['vertices']-grids['front']['vertices']), .1)
        np.testing.assert_allclose(fuse_grids(grids)['vertices'], self.samples[0].mesh, atol=1e-6)

    def test_incomplete_or_mixed_calibration_cannot_create_grids_or_fusion(self):
        with self.assertRaises(ValueError):
            build_grids(self.samples[:-1], self.settings)
        with self.assertRaises(ValueError):
            fuse_grids({key: value for key, value in self.grids.items() if key != 'up'})
        self.samples[0].pose_basis = {'version': 1, 'name': 'different'}
        with self.assertRaises(ValueError):
            build_grids(self.samples, self.settings)

    def test_deformed_depth_or_wrong_embedding_fails_both_comparisons(self):
        query = sample('left')
        query.mesh = query.mesh.copy()
        query.mesh[:, 2] += np.sin(np.arange(468))*.7
        self.assertIsNone(compare_grid(query, self.grids['left'], .45, self.settings))
        self.assertIsNone(compare_fused(query, self.model, .45, self.settings))
        query = sample('left', embedding=np.roll(sample().embedding, 1))
        self.assertIsNone(compare_grid(query, self.grids['left'], .45, self.settings))
        self.assertIsNone(compare_fused(query, self.model, .45, self.settings))

    def test_different_angle_basis_cannot_reuse_old_grid_or_fused_model(self):
        query = sample('left')
        self.grids['left']['pose_basis'] = {'name': 'other'}
        self.model['pose_basis'] = {'name': 'other'}
        self.assertIsNone(compare_grid(query, self.grids['left'], .45, self.settings))
        self.assertIsNone(compare_fused(query, self.model, .45, self.settings))
        with self.assertRaises(ValueError):
            fuse_grids(self.grids)

    def test_rendering_uses_real_mesh_topology_without_changing_original_landmarks(self):
        self.assertEqual(render_grids(self.grids).shape, (380, 570, 3))
        self.assertEqual(render_fused_model(self.model).shape, (380, 570, 3))
        raw = landmarks()
        before = raw.copy()
        preview = np.zeros((360, 640, 3), np.uint8)
        draw_mesh_preview(preview, raw)
        self.assertGreater(np.count_nonzero(preview), 1000)
        np.testing.assert_array_equal(raw, before)


class GridDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'faces.sqlite3'
        self.store = initialize_database(self.path)
        self.settings = GuidedSettings()
        self.store.enroll_directional('owner', profile())

    def test_final_db_checks_accept_wider_mesh_pitch_but_still_require_identity_and_shape(self):
        registered = [replace(item, pose=None, pose_basis=MANUAL_POSE_BASIS, capture_mode='manual') for item in profile()]
        self.store.enroll_directional('owner', registered)
        basis = dict(version=1, kind=MESH_AUTH_POSE_KIND, centers=ANGLE_CENTERS)
        front = replace(sample('front'), pose=Pose(0., -10., 0., sample().pose.box), pose_basis=basis)
        left = replace(sample('left'), pose=Pose(-27., -11., 0., sample().pose.box), pose_basis=basis)
        strict = replace(self.settings, mesh_front_pitch_scale=1, mesh_side_pitch_scale=1)
        db_check = self.store.record_directional_pair
        self.assertIsNone(db_check('owner', [front, left], 'test', .45, strict))
        wrong_identity = replace(left, embedding=np.roll(left.embedding, 1))
        self.assertIsNone(db_check('owner', [front, wrong_identity], 'test', .45, self.settings))
        deformed = left.mesh.copy()
        deformed[:, 2] += np.sin(np.arange(468))*.7
        self.assertIsNone(db_check('owner', [front, replace(left, mesh=deformed)], 'test', .45, self.settings))
        self.assertEqual(self.store.count_observations(), 0)
        self.assertIsNotNone(db_check('owner', [front, left], 'test', .45, self.settings))
        self.assertEqual(self.store.count_observations(), 2)

    def test_both_profiles_are_stored_and_authentication_logs_both_comparisons(self):
        self.assertTrue(self.store.has_grid_profile('owner'))
        self.assertTrue(self.store.has_fused_model('owner'))
        self.assertEqual(len(self.store.face_grids('owner')), 4)
        self.assertIsNotNone(self.store.record_directional_pair('owner', [sample('left'), sample('up')],
                                                               'test', .45, self.settings))
        with closing(self.store._connect()) as connection:
            rows = connection.execute('SELECT grid_model,fused_model,grid_error,fused_error,auth_session FROM observations').fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0:2], (GRID_MODEL, FUSED_MODEL))
        self.assertLess(rows[0][2], 1e-5)
        self.assertLess(rows[0][3], 1e-5)
        self.assertEqual(rows[0][4], rows[1][4])

    def test_grid_failure_rejects_pair_even_when_fused_model_still_matches(self):
        with closing(self.store._connect()) as connection, connection:
            grid = self.store.face_grids('owner', connection)['up']['vertices'].copy()
            grid[:, 2] += np.sin(np.arange(468))*.7
            connection.execute('UPDATE face_grids SET vertices=? WHERE person=? AND direction=?',
                               (json.dumps(grid.tolist()), 'owner', 'up'))
        self.assertIsNotNone(self.store.compare_face_model('owner', sample('up'), .45, self.settings))
        self.assertIsNone(self.store.record_directional_pair('owner', [sample('left'), sample('up')], 'test', .45, self.settings))
        self.assertEqual(self.store.count_observations(), 0)

    def test_fused_failure_rejects_pair_even_when_both_directional_grids_match(self):
        with closing(self.store._connect()) as connection, connection:
            vertices = self.store.face_model('owner', connection)['vertices'].copy()
            vertices[:, 2] += np.sin(np.arange(468))*.7
            connection.execute('UPDATE face_models SET vertices=? WHERE person=?', (json.dumps(vertices.tolist()), 'owner'))
        self.assertIsNotNone(self.store.compare_face_grid('owner', sample('left'), .45, self.settings))
        self.assertIsNone(self.store.record_directional_pair('owner', [sample('left'), sample('up')], 'test', .45, self.settings))
        self.assertEqual(self.store.count_observations(), 0)

    def test_missing_fused_or_incomplete_grid_cannot_authenticate(self):
        with closing(self.store._connect()) as connection, connection:
            connection.execute('DELETE FROM face_models WHERE person=?', ('owner',))
        self.assertIsNone(self.store.record_directional_pair('owner', [sample('left'), sample('up')], 'test', .45, self.settings))
        self.assertEqual(self.store.count_observations(), 0)

    def test_fused_insert_failure_rolls_back_all_twelve_photos_and_five_grids(self):
        previous = self.store.face_model('owner')['vertices'].copy()
        with closing(self.store._connect()) as connection, connection:
            connection.execute("CREATE TRIGGER reject_model BEFORE INSERT ON face_models BEGIN SELECT RAISE(ABORT,'no model'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.enroll_directional('owner', profile())
        self.assertEqual(len(self.store.directional_samples('owner')), 12)
        self.assertTrue(self.store.has_grid_profile('owner'))
        np.testing.assert_array_equal(self.store.face_model('owner')['vertices'], previous)

    def test_compatible_existing_samples_are_migrated_without_changing_people_or_photos(self):
        original = self.store.people.copy()
        with closing(self.store._connect()) as connection, connection:
            connection.execute('DELETE FROM face_grids')
            connection.execute('DELETE FROM face_models')
        migrated = initialize_database(self.path)
        self.assertEqual(migrated.people, original)
        self.assertEqual(len(migrated.directional_samples('owner')), 12)
        self.assertTrue(migrated.has_grid_profile('owner'))
        self.assertTrue(migrated.has_fused_model('owner'))

    def test_old_derived_fused_model_version_is_rebuilt_from_valid_grids(self):
        with closing(self.store._connect()) as connection, connection:
            connection.execute('UPDATE face_models SET model=? WHERE person=?', ('old-version', 'owner'))
        migrated = initialize_database(self.path)
        self.assertTrue(migrated.has_fused_model('owner'))
        self.assertEqual(len(migrated.directional_samples('owner')), 12)

    def make_legacy_five_pose_database(self):
        with closing(self.store._connect()) as connection, connection:
            connection.execute('''INSERT INTO directional_samples
                SELECT person,'down',sample_index,embedding,landmarks,mesh,?,image_size,
                       crop_jpeg,mesh_model,captured_at,pose_basis,capture_mode
                FROM directional_samples WHERE direction='front' ''', ('[0,20,0]',))
            connection.execute('UPDATE people SET samples=15')
            connection.execute('''INSERT INTO face_grids
                SELECT person,'down',vertices,embedding,sample_count,grid_model,mesh_model,pose_basis,updated_at
                FROM face_grids WHERE direction='front' ''')
            connection.execute("UPDATE face_grids SET grid_model='five-pose-face-grid-468-v1'")
            connection.execute("UPDATE face_models SET directions=5,model='fused-five-pose-face-468-v1'")
            schema = connection.execute("SELECT sql FROM sqlite_master WHERE name='face_models'").fetchone()[0]
            legacy = schema.replace('face_models', 'legacy_models').replace('directions IN (4,5)', 'directions=5')
            connection.execute(legacy)
            connection.execute('INSERT INTO legacy_models SELECT * FROM face_models')
            connection.execute('DROP TABLE face_models')
            connection.execute('ALTER TABLE legacy_models RENAME TO face_models')

    def test_old_five_pose_schema_and_photos_convert_to_four_grids_without_losing_originals(self):
        self.make_legacy_five_pose_database()
        with closing(self.store._connect()) as connection:
            original_photos = connection.execute('SELECT * FROM directional_samples ORDER BY direction,sample_index').fetchall()
            original_people = connection.execute('SELECT * FROM people').fetchall()
        migrated = initialize_database(self.path)
        self.assertTrue(migrated.has_grid_profile('owner'))
        self.assertTrue(migrated.has_fused_model('owner'))
        self.assertEqual(set(migrated.face_grids('owner')), {'front', 'up', 'left', 'right'})
        self.assertEqual(migrated.face_model('owner')['directions'], 4)
        with closing(migrated._connect()) as connection:
            self.assertEqual(connection.execute('SELECT * FROM directional_samples ORDER BY direction,sample_index').fetchall(),
                             original_photos)
            self.assertEqual(connection.execute('SELECT * FROM people').fetchall(), original_people)
            self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.assertEqual(len(migrated.directional_samples('owner', 'down')), 3)
        self.assertIsNone(migrated.record_directional_pair('owner', [sample('down'), sample('left')],
                                                         'test', .45, self.settings))
        self.assertEqual(migrated.count_observations(), 0)
        self.assertIsNotNone(migrated.record_directional_pair('owner', [sample('up'), sample('left')],
                                                            'test', .45, self.settings))
        reopened = initialize_database(self.path)
        self.assertTrue(reopened.has_fused_model('owner'))
        self.assertEqual(len(reopened.directional_samples('owner')), 15)
        reopened.enroll_directional('owner', profile())
        self.assertEqual(len(reopened.directional_samples('owner')), 12)
        self.assertEqual(reopened.people['owner']['samples'], 12)

    def test_invalid_legacy_photos_are_preserved_but_cannot_authenticate_as_a_four_pose_model(self):
        self.make_legacy_five_pose_database()
        with closing(self.store._connect()) as connection, connection:
            connection.execute("DELETE FROM directional_samples WHERE direction='up' AND sample_index=2")
        migrated = initialize_database(self.path)
        self.assertFalse(migrated.has_directional_profile('owner'))
        self.assertFalse(migrated.has_grid_profile('owner'))
        self.assertFalse(migrated.has_fused_model('owner'))
        self.assertEqual(len(migrated.directional_samples('owner')), 14)
        self.assertEqual(migrated.face_model('owner')['directions'], 5)
        self.assertIsNone(migrated.record_directional_pair('owner', [sample('up'), sample('left')],
                                                         'test', .45, self.settings))
        self.assertEqual(migrated.count_observations(), 0)

    def test_embedding_only_reenrollment_invalidates_old_grids_and_models_and_delete_cascades(self):
        add_user(self.store, 'owner', [sample().embedding]*15)
        self.assertEqual(self.store.face_grids('owner'), {})
        self.assertIsNone(self.store.face_model('owner'))
        self.store.enroll_directional('owner', profile())
        delete_user(self.store, 'owner')
        self.assertEqual(self.store.face_grids('owner'), {})
        self.assertIsNone(self.store.face_model('owner'))


if __name__ == '__main__':
    unittest.main()
