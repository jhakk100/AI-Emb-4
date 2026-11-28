"""Named default calibration and full FHD grid enrollment/authentication with real five-point PnP."""

from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stages'))

import cv2
import numpy as np

from capture_control import STANDBY_PROFILE, ACTIVE_PROFILE, CALIBRATED_ACTIVE_PROFILE, EnrollmentResolutionGate
from auth_diagnostics import AuthDiagnostics
from database import initialize_database
from enrollment import EnrollmentPipeline
from face_core import Face
from guided_capture import FACE_POINTS, GuidedCapture, GuidedSettings, MANUAL_POSE_BASIS
from liveness import Pose
from pipeline import FacePipeline, PoseHintDisplay
from pose_defaults import DefaultPoseGuide, load_default_preset
from standby import FaceWatcher, GateSettings
from test_secure_pipeline import landmarks


class DefaultPresetTests(unittest.TestCase):
    def test_jonghak_is_selected_and_original_measurements_are_preserved(self):
        preset = load_default_preset()
        self.assertEqual(preset.name, '종학')
        stored = json.loads(preset.path.read_text(encoding='utf-8'))
        np.testing.assert_array_equal(preset.profile.centers, stored['centers'])
        self.assertEqual(preset.context['size'], [1920, 1080])
        self.assertAlmostEqual(preset.profile.centers[1, 0], -40.92141605647862)
        self.assertAlmostEqual(preset.profile.centers[2, 0], 52.87479244253902)

    def test_default_pose_guide_uses_same_filter_and_aligns_neutral_camera_bias(self):
        preset = load_default_preset()
        guide = DefaultPoseGuide(preset)
        front = preset.profile.centers[0]
        neutral = Pose(front[0]+3, front[1]-4, front[2], (0, 0, 100, 100))
        guide.calibrate_neutral(neutral)
        aligned = guide.align(neutral)
        self.assertTrue(guide.matches('front', aligned))
        for index, name in enumerate(('front', 'left', 'right', 'up', 'down')):
            target = preset.profile.centers[index]
            transformed = guide.align(Pose(target[0]+3, target[1]-4, target[2], neutral.box))
            self.assertEqual(guide.matches(name, transformed), name != 'down')
        self.assertEqual(guide.filter.settings.smoothing_seconds, .18)
        with self.assertRaises(ValueError):
            DefaultPoseGuide(preset, source='camera:1')

    def test_default_authentication_requests_true_fhd_and_preview_retains_low_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            store = initialize_database(Path(folder) / 'faces.sqlite3')
            pipeline = FacePipeline(Mock(), store=store)
            self.addCleanup(pipeline.close)
            self.assertEqual(pipeline.profile, STANDBY_PROFILE)
            pipeline.mode = 'active'
            self.assertEqual(pipeline.profile, CALIBRATED_ACTIVE_PROFILE)
            with self.assertRaisesRegex(RuntimeError, 'FHD'):
                pipeline.process(np.zeros((480, 640, 3), np.uint8))
            pipeline.set_purpose('preview')
            pipeline.mode = 'active'
            self.assertEqual(pipeline.profile, ACTIVE_PROFILE)
        gate = EnrollmentResolutionGate(max_misses=2)
        self.assertFalse(gate.accept(np.zeros((480, 640, 3), np.uint8), CALIBRATED_ACTIVE_PROFILE))
        with self.assertRaises(RuntimeError):
            gate.accept(np.zeros((480, 640, 3), np.uint8), CALIBRATED_ACTIVE_PROFILE)

    def test_invalid_default_config_is_not_silently_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(load_default_preset(root))
            (root / 'pose_defaults.json').write_text('[]', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_default_preset(root)
            (root / 'pose_defaults.json').write_text(json.dumps(dict(version=1, enabled=True, preset='../outside.json')), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_default_preset(root)


class PoseHintDisplayTests(unittest.TestCase):
    def test_single_noisy_frame_cannot_show_or_clear_guidance(self):
        display = PoseHintDisplay()
        display.update('raise_chin', 0.)
        display.update(None, .11)
        self.assertIsNone(display.visible)
        display.update('raise_chin', .23)
        display.update('raise_chin', .34)
        self.assertEqual(display.visible, 'raise_chin')
        display.update(None, 1.)
        self.assertEqual(display.visible, 'raise_chin')
        display.update('raise_chin', 1.11)
        self.assertEqual(display.visible, 'raise_chin')
        display.update(None, 1.22)
        display.update(None, 1.60)
        self.assertIsNone(display.visible)

    def test_opposite_hint_waits_and_a_new_direction_clears_immediately(self):
        display = PoseHintDisplay()
        display.update('raise_chin', 0.)
        display.update('raise_chin', .11)
        display.update('lower_chin', .18)
        display.update('lower_chin', .46)
        self.assertEqual(display.visible, 'raise_chin')
        display.update('lower_chin', .74)
        self.assertEqual(display.visible, 'lower_chin')
        display.reset()
        self.assertIsNone(display.visible)
        display.update('lower_chin', .85)
        self.assertIsNone(display.visible)


class CalibratedGridFlowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = initialize_database(Path(temp.name) / 'faces.sqlite3')
        self.preset = load_default_preset()
        self.centers = dict(zip(('front', 'left', 'right', 'up', 'down'), self.preset.profile.centers))
        self.angles = self.centers['front'].copy()
        self.now = 1.
        self.box = (720., 270., 480., 427.5)
        self.frame = np.zeros((1080, 1920, 3), np.uint8)
        self.vector = np.zeros(128, np.float32)
        self.vector[0] = 1.
        self.engine = Mock()
        self.engine.detect.side_effect = lambda frame, size: np.array([self.detection()])
        self.engine.analyze.return_value = [Face(self.box, self.vector, 100)]
        self.mesh = Mock()
        self.mesh.observe.return_value = Pose(0, 0, 0, self.box, landmarks())

    def mesh_pose(self, angles):
        yaw, pitch, roll = np.radians(angles)
        ry = cv2.Rodrigues(np.array([0., -yaw, 0.]))[0]
        rx = cv2.Rodrigues(np.array([pitch, 0., 0.]))[0]
        rz = cv2.Rodrigues(np.array([0., 0., roll]))[0]
        center = np.array([.5, .5, 0.])
        scale = np.array([1920., 1080., 1920.])
        points = (landmarks()-center)*scale @ (rz @ ry @ rx).T / scale + center
        return Pose(0, 0, 0, self.box, points.astype(np.float32))

    def detection(self):
        yaw, pitch, roll = np.radians(self.angles)
        ry = cv2.Rodrigues(np.array([0., -yaw, 0.]))[0]
        rx = cv2.Rodrigues(np.array([pitch, 0., 0.]))[0]
        rz = cv2.Rodrigues(np.array([0., 0., roll]))[0]
        rvec = cv2.Rodrigues(rz @ ry @ rx)[0]
        camera = np.array([[1920., 0, 960.], [0, 1920., 540.], [0, 0, 1.]])
        points = cv2.projectPoints(FACE_POINTS, rvec, np.array([0., 0., 600.]), camera, np.zeros(4))[0].reshape(5, 2)
        return np.r_[self.box, points.reshape(-1), .99].astype(np.float32)

    def advance(self, pipeline):
        target = pipeline.challenge.target if pipeline.challenge and not pipeline.terminal else 'front'
        self.angles += (self.centers[target]-self.angles)*.25
        self.now += .05
        with patch('pipeline.monotonic', return_value=self.now), patch('enrollment.monotonic', return_value=self.now), \
                patch('standby.time.monotonic', return_value=self.now):
            return pipeline.process(self.frame)

    def enroll(self):
        with patch('enrollment.monotonic', return_value=self.now):
            pipeline = EnrollmentPipeline(self.engine, 'owner', store=self.store, mesh=self.mesh)
        self.addCleanup(pipeline.close)
        for count in range(12):
            self.mesh.observe.return_value = self.mesh_pose(self.centers[pipeline.challenge.target])
            with patch('enrollment.monotonic', return_value=self.now):
                self.assertTrue(pipeline.request_manual_capture(pipeline.session_id, pipeline.challenge.target,
                                                                 count, self.now))
            result = self.advance(pipeline)
            self.assertTrue(result.manual_captured, result.status)
        self.assertEqual(result.enrollment_completed, 'owner', result.status)
        self.assertTrue(result.grid_ready and result.fused_ready)
        self.assertIsNone(result.pose_preset)
        return pipeline

    def test_manual_profile_authenticates_using_mesh_even_when_old_pose_guide_cannot_classify(self):
        enrollment = self.enroll()
        self.assertEqual(self.mesh.observe.call_count, 12)
        self.assertEqual(self.store.count_observations(), 0)
        rows = self.store.directional_samples('owner')
        self.assertTrue(all(row['pose_basis'] == MANUAL_POSE_BASIS for row in rows))
        self.assertTrue(self.store.has_grid_profile('owner'))
        self.assertTrue(self.store.has_fused_model('owner'))
        self.assertTrue(self.store.grid_basis_matches('owner', self.preset.basis))
        self.angles = self.centers['front'].copy()
        mesh = Mock()
        mesh.observe.side_effect = lambda *args: self.mesh_pose(self.angles)
        auth = FacePipeline(self.engine, watcher=FaceWatcher(GateSettings(hold_seconds=0)), store=self.store,
                            mesh=mesh, start_in_standby=False)
        self.addCleanup(auth.close)
        trace = []
        diagnostic = AuthDiagnostics(self.store.path.parent / 'diagnostics')
        self.assertTrue(diagnostic.begin())
        self.addCleanup(diagnostic.finish, 'incomplete', 'test cleanup')
        def record(event, **fields):
            trace.append(dict(event=event, **fields))
            diagnostic.write(event, **fields)
        auth.auth_trace = record
        with patch.object(auth.pose_guide, 'observe', side_effect=AssertionError('old guide called')), \
                patch('pipeline.GuidedCapture', side_effect=lambda purpose, settings, now, **kwargs:
                      GuidedCapture(purpose, settings, now, directions=('left', 'up'), **kwargs)):
            for _ in range(1000):
                result = self.advance(auth)
                if auth.terminal:
                    break
        self.assertEqual(result.authenticated_name, 'owner', result.status)
        self.assertTrue(result.grid_ready and result.fused_ready and result.liveness_passed)
        self.assertGreater(mesh.observe.call_count, 2)
        self.assertEqual(self.store.count_observations(), 2)
        checks = [row for row in trace if row['event'] == 'pose_check']
        self.assertTrue(any(row['requested'] == 'up' and row['detected'] == 'up'
                            and row['block'] == 'capture' for row in checks))
        self.assertTrue(any(row['block'] == 'direction' and row['target_distance_ratio'] is not None
                            for row in checks))
        comparisons = [row for row in trace if row['event'] == 'comparison']
        self.assertEqual({row['check'] for row in comparisons}, {'identity', 'sample', 'grid', 'fused'})
        self.assertTrue(all('score' in row and 'passed' in row for row in comparisons))
        diagnostic.finish('success', result.status)
        self.assertIsNone(diagnostic.error)
        with closing(self.store._connect()) as connection:
            rows = connection.execute('SELECT grid_model,fused_model FROM observations').fetchall()
        self.assertTrue(all(grid is not None and fused is not None for grid, fused in rows))
        self.assertTrue(all(call.kwargs['size'] == ACTIVE_PROFILE.size for call in self.engine.detect.call_args_list))

    def test_manual_profile_does_not_capture_a_static_frontal_mesh_for_a_turn(self):
        self.enroll()
        self.angles = self.centers['front'].copy()
        mesh = Mock()
        mesh.observe.return_value = self.mesh_pose(self.angles)
        auth = FacePipeline(self.engine, store=self.store, mesh=mesh, start_in_standby=False)
        self.addCleanup(auth.close)
        with patch('pipeline.GuidedCapture', side_effect=lambda purpose, settings, now, **kwargs:
                   GuidedCapture(purpose, settings, now, directions=('left', 'up'), **kwargs)):
            first = self.advance(auth)
            self.assertEqual(first.direction, 'left')
            for _ in range(30):
                result = self.advance(auth)
        self.assertEqual(result.direction, 'left')
        self.assertFalse(result.authentication_attempted)
        self.assertEqual(auth.challenge.samples, [])
        self.assertEqual(self.store.count_observations(), 0)

    def test_manual_grid_templates_rebuild_on_reopen_without_angles(self):
        self.enroll()
        with closing(self.store._connect()) as connection, connection:
            connection.execute('DELETE FROM face_grids')
            connection.execute('DELETE FROM face_models')
        reopened = initialize_database(self.store.path)
        self.assertTrue(reopened.has_grid_profile('owner'))
        self.assertTrue(reopened.has_fused_model('owner'))
        self.assertTrue(reopened.grid_basis_matches('owner', self.preset.basis))


if __name__ == '__main__':
    unittest.main()
