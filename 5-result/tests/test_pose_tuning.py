"""The standalone pose tester must use production checks and only save tuning values."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stages'))

import numpy as np

from guided_capture import DIRECTIONS, GuidedSettings, direction_matches, load_guided_settings
from liveness import Pose
import models
from pose_tuning import PoseTuner

BOX = (100, 100, 200, 250)


class PoseTuningTests(unittest.TestCase):
    def setUp(self):
        self.tuner = PoseTuner(GuidedSettings())

    def neutral(self):
        for i in range(5):
            self.tuner.observe(Pose(4, 18, 3, BOX), i*.1)
        self.tuner.calibrate(.4)

    def test_calibration_matches_runtime_yaw_pitch_baseline_and_preserves_roll(self):
        self.neutral()
        relative = self.tuner.observe(Pose(29, -2, 3, BOX), .5)
        self.assertEqual((relative.yaw, relative.pitch, relative.roll), (25, -20, 3))

    def test_reset_baseline_discards_previous_origin_and_pose_history(self):
        self.neutral()
        self.tuner.reset_baseline()
        self.assertIsNone(self.tuner.origin)
        self.assertEqual(len(self.tuner.recent), 0)
        self.assertIsNone(self.tuner.observe(Pose(10, 20, 0, BOX), .5))

    def test_unstable_or_stale_face_cannot_set_frontal_baseline(self):
        self.neutral()
        with self.assertRaises(ValueError):
            self.tuner.calibrate(2)
        self.tuner.observe(Pose(40, 0, 0, BOX), .5)
        with self.assertRaises(ValueError):
            self.tuner.calibrate(.5)
        self.tuner.observe(None, .6)
        with self.assertRaises(ValueError):
            self.tuner.calibrate(.6)

    def test_boundary_angles_signs_and_roll_match_production(self):
        for yaw_sign in (-1, 1):
            for pitch_sign in (-1, 1):
                self.tuner.configure(yaw_sign=yaw_sign, pitch_sign=pitch_sign)
                for target in DIRECTIONS:
                    for yaw, pitch, roll in ((0, 0, 0), (15, 0, 0), (-35, 0, 0),
                                             (0, -12, 0), (0, 30, 0), (0, -31, 0), (25, 0, 20)):
                        pose = Pose(yaw, pitch, roll, BOX)
                        angle, _ = self.tuner.check(target, pose, 0)
                        self.assertEqual(angle, direction_matches(target, pose, self.tuner.settings))

    def test_angle_pass_requires_hold_and_multiple_frames_to_pass_the_trial(self):
        pose = Pose(-25, 0, 0, BOX)
        self.assertEqual(self.tuner.check('left', pose, 0), (True, False))
        self.assertEqual(self.tuner.check('left', pose, .1), (True, False))
        self.assertEqual(self.tuner.check('left', pose, .3), (True, True))
        self.assertEqual(self.tuner.check('right', pose, .4), (False, False))
        self.assertEqual(self.tuner.check('left', None, .5), (False, False))
        self.assertEqual(self.tuner.check('left', pose, .6), (True, False))

    def test_adjusting_threshold_resets_the_hold(self):
        pose = Pose(-25, 0, 0, BOX)
        for now in (0, .1, .3):
            self.tuner.check('left', pose, now)
        self.tuner.configure(yaw_min=20.)
        self.assertEqual(self.tuner.check('left', pose, .4), (True, False))

    def test_invalid_thresholds_are_rejected_without_changing_current_settings(self):
        for values in ({'yaw_min': 10}, {'pitch_min': 40}, {'hold_seconds': 1},
                       {'front_degrees': float('nan')}, {'mesh_limit': .5}):
            with self.assertRaises(ValueError):
                self.tuner.configure(**values)
        self.assertEqual(self.tuner.settings, GuidedSettings())

    def test_explicit_save_preserves_other_fields_and_never_persists_personal_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'guided_capture.json'
            settings = GuidedSettings(mesh_limit=.2, depth_limit=.15, auth_timeout=40)
            path.write_text(json.dumps(asdict(settings)), encoding='utf-8')
            before = path.read_bytes()
            self.neutral()
            self.tuner.configure(yaw_min=20., yaw_max=40., pitch_sign=-1)
            self.assertEqual(path.read_bytes(), before)
            self.tuner.save(path)
            saved = load_guided_settings(path)
            self.assertEqual((saved.yaw_min, saved.yaw_max, saved.pitch_sign), (20, 40, -1))
            self.assertEqual((saved.mesh_limit, saved.depth_limit, saved.auth_timeout), (.2, .15, 40))
            self.assertNotIn('origin', json.loads(path.read_text(encoding='utf-8')))
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_pose_test_model_loading_only_checks_yunet(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / models.MODELS[0][1]
            path.write_bytes(b'model')
            with patch.object(models, 'MODEL_DIR', Path(folder)), \
                    patch.object(models, 'sha256', return_value=models.MODELS[0][2]), \
                    patch('models.urllib.request.urlopen') as download:
                self.assertEqual(models.ensure_models(detector_only=True), [path])
            download.assert_not_called()


class PoseTuningGuiTests(unittest.TestCase):
    def setUp(self):
        try:
            import tkinter as tk
            import pose_test
        except ImportError as exc:
            self.skipTest(str(exc))
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'guided_capture.json'
        self.path.write_text(json.dumps(asdict(GuidedSettings())), encoding='utf-8')
        self.app = pose_test.PoseTestApp(self.root, config=self.path)
        self.root.update_idletasks()

    def test_live_preview_uses_production_direction_verdict_and_shows_current_bounds(self):
        from test_landmark_filter import face
        self.app.tuner.origin = (4, 18)
        self.app.target.set('left')
        with patch('pose_test.yunet_pose', return_value=Pose(-21, 18, 0, BOX)):
            for now in (1., 1.05, 1.1):
                self.app.show((np.zeros((1080, 1920, 3), np.uint8), [face()],
                               Pose(-21, 18, 0, BOX), '', (1920, 1080), now))
        self.assertIn('각도 통과', self.app.verdict.get())
        self.assertIn('통과', self.app.table.item('left', 'values'))
        self.assertIn('−35', self.app.table.item('left', 'values')[2])
        self.assertIn('1920×1080', self.app.capture_info.get())
        self.assertEqual(len(self.app.canvas.find_withtag('raw_point')), 5)
        self.assertEqual(len(self.app.canvas.find_withtag('stable_point')), 10)

    def show_face(self, now, row=None):
        from test_landmark_filter import face
        if row is None:
            row = face()
        with patch('pose_test.yunet_pose', return_value=Pose(-25, 0, 0, BOX)):
            self.app.show((np.zeros((480, 640, 3), np.uint8), [row], Pose(-25, 0, 0, BOX), '', (640, 480), now))

    def test_spike_clears_hold_and_draws_no_cached_stable_points(self):
        from test_landmark_filter import face
        self.app.target.set('left')
        self.app.tuner.origin = (0, 0)
        for now in (1., 1.05, 1.1):
            self.show_face(now)
        self.assertIsNotNone(self.app.tuner.trial)
        row = face()
        row[8] += 40
        self.show_face(1.15, row)
        self.assertIn('제외', self.app.verdict.get())
        self.assertIsNone(self.app.tuner.trial)
        self.assertEqual(self.app.canvas.find_withtag('stable_point'), ())
        self.assertTrue(all(self.app.table.item(direction, 'values')[1] == '대기' for direction in DIRECTIONS))

    def test_losing_face_clears_baseline_and_reacquisition_needs_new_neutral(self):
        for now in (1., 1.05, 1.1):
            self.show_face(now)
        self.app.tuner.origin = (0, 0)
        self.app.show((np.zeros((480, 640, 3), np.uint8), [], None, 'face lost', (640, 480), 1.15))
        self.assertIsNone(self.app.tuner.origin)
        self.assertEqual(self.app.canvas.find_withtag('stable_point'), ())
        for now in (1.2, 1.25, 1.3):
            self.show_face(now)
        self.assertIn('정면', self.app.verdict.get())
        self.assertIsNone(self.app.tuner.trial)

    def test_comparison_mode_and_filter_adjustment_reset_baseline_without_saving_file(self):
        before = self.path.read_bytes()
        self.app.tuner.origin = (4, 18)
        self.app.use_stabilization.set(False)
        self.app.reset_stabilization()
        self.assertIsNone(self.app.tuner.origin)
        self.app.filter_vars['smoothing_seconds'].set('.3')
        self.app.apply_filter_values()
        self.assertEqual(self.app.landmarks.settings.smoothing_seconds, .3)
        self.assertIsNone(self.app.tuner.origin)
        self.assertEqual(self.path.read_bytes(), before)
        previous = self.app.landmarks
        self.app.filter_vars['smoothing_seconds'].set('bad')
        self.app.apply_filter_values()
        self.assertIs(self.app.landmarks, previous)
        self.assertIn('입력 오류', self.app.status.get())

    def test_raw_point_display_can_be_hidden_independently_of_filtering(self):
        self.app.show_raw_points.set(False)
        for now in (1., 1.05, 1.1):
            self.show_face(now)
        self.assertEqual(self.app.canvas.find_withtag('raw_point'), ())
        self.assertEqual(len(self.app.canvas.find_withtag('stable_point')), 10)

    def begin_manual(self):
        for now in (1., 1.05, 1.1):
            self.show_face(now)
        with patch('pose_test.time.monotonic', return_value=1.1):
            self.app.begin_manual()
        self.assertTrue(self.app.manual.editing)

    def test_manual_drag_and_apply_resume_with_corrected_points_and_new_baseline(self):
        self.begin_manual()
        self.assertEqual(len(self.app.canvas.find_withtag('manual_handle')), 5)
        self.assertIsNone(self.app.tuner.origin)
        cx, cy = self.app.manual_to_canvas(*self.app.manual.points[2])
        self.app.select_manual_point(SimpleNamespace(x=cx, y=cy))
        self.app.drag_manual_point(SimpleNamespace(x=cx+10, y=cy+5))
        self.assertEqual(self.app.point_table.item('2', 'values')[1:], ('210', '225'))
        with patch('pose_test.time.monotonic', return_value=1.1):
            self.app.apply_manual()
        self.assertTrue(self.app.manual.active)
        self.assertFalse(self.app.manual.editing)
        self.assertIsNone(self.app.tuner.origin)
        self.assertEqual(self.app.canvas.find_withtag('manual_handle'), ())
        for now in (1.15, 1.2, 1.25):
            self.show_face(now)
        self.assertEqual(len(self.app.canvas.find_withtag('manual_point')), 5)
        self.assertEqual(self.app.point_table.item('2', 'values')[1:], ('210', '225'))
        self.app.reset_manual()
        self.assertFalse(self.app.manual.active)

    def test_manual_edit_freezes_preview_and_never_uses_editing_frames_as_pose_samples(self):
        self.begin_manual()
        photo = self.app.photo
        before = self.app.manual.points.copy()
        self.show_face(1.15)
        self.assertIs(self.app.photo, photo)
        np.testing.assert_array_equal(self.app.manual.points, before)
        self.assertEqual(len(self.app.tuner.recent), 0)
        self.assertIsNone(self.app.tuner.trial)
        self.assertTrue(self.app.calibrate_button.instate(['disabled']))
        self.app.cancel_manual()
        self.assertFalse(self.app.manual.editing)
        self.assertFalse(self.app.manual.active)
        self.assertTrue(self.app.calibrate_button.instate(['!disabled']))

    def test_manual_edit_tracks_face_loss_and_blocks_apply_even_if_a_face_returns(self):
        self.begin_manual()
        self.app.show((np.zeros((480, 640, 3), np.uint8), [], None, 'lost', (640, 480), 1.15))
        self.show_face(1.2)
        with patch('pose_test.time.monotonic', return_value=1.2):
            self.app.apply_manual()
        self.assertFalse(self.app.edit_tracking_ok)
        self.assertTrue(self.app.manual_apply_button.instate(['disabled']))
        self.assertFalse(self.app.manual.active)
        self.assertTrue(self.app.manual.editing)
        self.app.cancel_manual()
        self.assertFalse(self.app.manual.active)

    def test_manual_edit_rejects_stale_capture_and_invalid_point_layout(self):
        self.begin_manual()
        with patch('pose_test.time.monotonic', return_value=2.):
            self.app.apply_manual()
        self.assertTrue(self.app.manual.editing)
        self.assertFalse(self.app.manual.active)
        self.app.manual.move(2, 200, 150)
        with patch('pose_test.time.monotonic', return_value=1.1):
            self.app.apply_manual()
        self.assertTrue(self.app.manual.editing)
        self.assertFalse(self.app.manual.active)
        self.assertIn('눈', self.app.status.get())

    def test_manual_drag_mapping_and_keyboard_nudge_use_native_fhd_pixels(self):
        from test_landmark_filter import face
        row = face()*np.array([3., 2.25]*7+[1])
        with patch('pose_test.yunet_pose', return_value=Pose(0, 0, 0, BOX)):
            for now in (1., 1.05, 1.1):
                self.app.show((np.zeros((1080, 1920, 3), np.uint8), [row], Pose(0, 0, 0, BOX), '', (1920, 1080), now))
        with patch('pose_test.time.monotonic', return_value=1.1):
            self.app.begin_manual()
        before = self.app.manual.points[2].copy()
        cx, cy = self.app.manual_to_canvas(*before)
        self.app.select_manual_point(SimpleNamespace(x=cx, y=cy))
        self.app.drag_manual_point(SimpleNamespace(x=cx+10, y=cy+5))
        np.testing.assert_allclose(self.app.manual.points[2]-before, [30, 15])
        self.assertEqual(self.app.nudge_manual_point(SimpleNamespace(keysym='Right')), 'break')
        np.testing.assert_allclose(self.app.manual.points[2]-before, [31, 15])

    def test_display_has_integer_korean_angles_while_pose_math_keeps_fractional_precision(self):
        from test_landmark_filter import face
        self.app.tuner.origin = (4.25, 18.5)
        estimate = Pose(-21.75, 19.25, 1.75, BOX)
        with patch('pose_test.yunet_pose', return_value=estimate):
            for now in (1., 1.05, 1.1):
                self.app.show((np.zeros((480, 640, 3), np.uint8), [face()], estimate, '', (640, 480), now))
        text = self.app.angles.get()
        self.assertNotRegex(text, r'\d\.\d')
        self.assertIn('좌우각도 -21°', text)
        self.assertIn('상하각도 +19°', text)
        self.assertIn('고개 기울임', text)
        self.assertIn('정면 기준각도', text)
        self.assertNotRegex(text, r'yaw|pitch|roll')
        self.assertEqual(self.app.tuner.recent[-1][1].yaw, -21.75)
        self.assertEqual(self.app.tuner.settings.hold_seconds, .25)
        self.app.vars['yaw_min'].set('15.5')
        with self.assertRaisesRegex(ValueError, '정수'):
            self.app.apply_values()

    def test_stalled_capture_discards_manual_offsets_and_yellow_points(self):
        self.begin_manual()
        self.app.manual.move(2, 210, 225)
        with patch('pose_test.time.monotonic', return_value=1.1):
            self.app.apply_manual()
        for now in (1.15, 1.2, 1.25):
            self.show_face(now)
        self.assertTrue(self.app.manual.active)
        self.assertEqual(len(self.app.canvas.find_withtag('manual_point')), 5)
        self.app.worker = Mock()
        self.app.worker.stop_event.is_set.return_value = False
        with patch('pose_test.time.monotonic', return_value=2.):
            self.app.poll()
        self.assertFalse(self.app.manual.active)
        self.assertEqual(self.app.canvas.find_withtag('manual_point'), ())
        self.assertEqual(self.app.point_table.get_children(), ())

    def test_stalled_camera_clears_last_good_pose_without_waiting_for_another_packet(self):
        for now in (1., 1.05, 1.1):
            self.show_face(now)
        self.app.tuner.origin = (0, 0)
        self.app.worker = Mock()
        self.app.worker.stop_event.is_set.return_value = False
        with patch('pose_test.time.monotonic', return_value=2.):
            self.app.poll()
        self.assertIsNone(self.app.tuner.origin)
        self.assertIsNone(self.app.tuner.trial)
        self.assertEqual(self.app.canvas.find_withtag('stable_point'), ())
        self.assertIn('입력 지연', self.app.verdict.get())
        self.assertTrue(all(self.app.table.item(direction, 'values')[1] == '대기' for direction in DIRECTIONS))

    def test_pose_camera_only_runs_yunet_and_releases_the_camera(self):
        import pose_test
        frames, events = Queue(maxsize=1), Queue()
        worker = pose_test.PoseCamera(0, pose_test.ENROLLMENT_PROFILE, frames, events)
        capture = Mock()
        capture.read.return_value = True, np.zeros((1080, 1920, 3), np.uint8)
        model = Path(self.temp.name) / 'detector.onnx'
        model.write_bytes(b'model')
        with patch('pose_test.ensure_models', return_value=[model]) as models_only, \
                patch('pose_test.cv2.FaceDetectorYN.create'), \
                patch('pose_test.open_camera', return_value=capture), \
                patch('pose_test.detect_faces', return_value=np.empty((0, 15))), \
                patch('liveness.FaceMeshEstimator') as mesh, \
                patch('database.initialize_database') as database, \
                patch.object(worker.stop_event, 'wait', side_effect=lambda delay: worker.stop_event.set()):
            worker.run()
        self.assertTrue(models_only.call_args.kwargs['detector_only'])
        self.assertEqual(frames.get_nowait()[0].shape, (1080, 1920, 3))
        mesh.assert_not_called()
        database.assert_not_called()
        capture.release.assert_called_once()

    def test_save_and_reload_controls_require_explicit_save(self):
        self.app.vars['yaw_min'].set('20')
        self.app.apply_values()
        self.assertEqual(load_guided_settings(self.path).yaw_min, 15)
        self.app.save()
        self.assertEqual(load_guided_settings(self.path).yaw_min, 20)
        self.app.vars['yaw_min'].set('25')
        self.app.reload()
        self.assertEqual(float(self.app.vars['yaw_min'].get()), 20)

    def test_invalid_controls_stop_verdict_and_cannot_overwrite_file(self):
        before = self.path.read_bytes()
        self.app.vars['yaw_min'].set('bad')
        self.app.tuner.origin = (0, 0)
        self.app.show((np.zeros((480, 640, 3), np.uint8), [], Pose(-25, 0, 0, BOX), '', (640, 480), 1.))
        self.assertIn('입력 오류', self.app.verdict.get())
        self.app.save()
        self.assertEqual(self.path.read_bytes(), before)

    def setup_guided_calibration(self):
        self.app.worker = Mock()
        self.app.worker.stop_event.is_set.return_value = False
        self.app.resolution.set('640×480 인증')
        self.clock = 1.
        self.app.start_calibration()

    def feed_guided(self, name, frames=8):
        from test_landmark_filter import face
        from test_pose_calibration import pose
        estimate = pose(name)
        for _ in range(frames):
            self.clock += .05
            with patch('pose_test.yunet_pose', return_value=estimate):
                self.app.show((np.zeros((480, 640, 3), np.uint8), [face()], estimate, '', (640, 480), self.clock))

    def press_p(self):
        with patch('pose_test.time.monotonic', return_value=self.clock):
            self.app.on_p_press()
        self.app.on_p_release()

    def complete_guided_calibration(self):
        from pose_calibration import ORDER
        self.setup_guided_calibration()
        for name in ORDER:
            self.assertEqual(self.app.guided.target, name)
            self.feed_guided(name)
            self.press_p()
        self.assertEqual(self.app.guided.mode, 'ready')

    def test_p_guided_calibration_records_all_directions_and_test_button_checks_them(self):
        from pose_calibration import ORDER
        before = self.path.read_bytes()
        self.complete_guided_calibration()
        self.assertTrue(self.app.calibration_file.exists())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(self.app.test_button.instate(['!disabled']))
        self.assertTrue(self.app.record_button.instate(['disabled']))
        self.assertEqual(self.app.table.get_children(), ORDER)
        self.assertIn('좌우 +65°', self.app.guided_table.item('right', 'values')[1])
        self.app.start_calibration_test()
        for target in ORDER:
            self.assertEqual(self.app.guided.target, target)
            self.feed_guided(target, 12)
        self.assertEqual(self.app.guided.mode, 'tested')
        self.assertEqual(self.app.guided.results, list(ORDER))
        self.assertIn('4/4', self.app.verdict.get())
        for target in ORDER:
            self.assertEqual(self.app.guided_table.item(target, 'values')[2], '테스트 통과')

    def test_held_p_cannot_capture_next_request_without_a_new_key_press(self):
        self.setup_guided_calibration()
        self.feed_guided('front')
        with patch('pose_test.time.monotonic', return_value=self.clock):
            self.app.on_p_press()
        self.assertEqual(self.app.guided.target, 'left')
        self.feed_guided('left')
        with patch('pose_test.time.monotonic', return_value=self.clock):
            self.app.on_p_press()
        self.assertEqual(self.app.guided.target, 'left')
        self.app.on_p_release()
        self.press_p()
        self.assertEqual(self.app.guided.target, 'right')

    def test_p_key_capture_works_with_korean_input_and_ignores_other_keys(self):
        self.setup_guided_calibration()
        self.feed_guided('front')
        with patch('pose_test.time.monotonic', return_value=self.clock):
            self.app.on_guided_key_press(SimpleNamespace(keysym='a', char='a', keycode=65))
            self.assertEqual(self.app.guided.target, 'front')
            korean_p = SimpleNamespace(keysym='unknown', char='ㅔ', keycode=80)
            self.assertEqual(self.app.on_guided_key_press(korean_p), 'break')
        self.app.on_guided_key_release(korean_p)
        self.assertEqual(self.app.guided.target, 'left')
        self.assertFalse(self.app.p_key_down)

    def test_early_p_and_face_loss_do_not_record_or_advance_and_no_partial_file_is_saved(self):
        self.setup_guided_calibration()
        self.press_p()
        self.assertEqual(self.app.guided.target, 'front')
        self.assertEqual(self.app.guided.records, {})
        self.feed_guided('front')
        self.press_p()
        self.clock += .05
        self.app.show((np.zeros((480, 640, 3), np.uint8), [], None, 'lost', (640, 480), self.clock))
        self.press_p()
        self.assertEqual(self.app.guided.target, 'left')
        self.assertEqual(list(self.app.guided.records), ['front'])
        self.assertFalse(self.app.calibration_file.exists())
        self.assertTrue(self.app.test_button.instate(['disabled']))
        self.assertTrue(self.app.manual_begin_button.instate(['disabled']))
        self.app.cancel_calibration()
        self.assertEqual(self.app.guided.records, {})

    def test_previous_request_can_be_retaken_without_typing_angle_values(self):
        self.setup_guided_calibration()
        self.feed_guided('front')
        self.press_p()
        self.feed_guided('left')
        self.press_p()
        self.app.previous_calibration_pose()
        self.assertEqual(self.app.guided.target, 'left')
        self.assertEqual(list(self.app.guided.records), ['front'])
        self.assertEqual(self.app.guided_table.item('left', 'values')[1], '미기록')
        self.feed_guided('left')
        self.press_p()
        self.assertEqual(self.app.guided.target, 'right')

    def test_saved_calibration_loads_for_same_pipeline_and_changed_filter_requires_recalibration(self):
        self.complete_guided_calibration()
        self.app.load_guided_profile()
        self.app.refresh_guided_view()
        self.assertEqual(self.app.guided.mode, 'ready')
        self.assertTrue(self.app.test_button.instate(['!disabled']))
        self.app.filter_vars['smoothing_seconds'].set('.3')
        self.app.apply_filter_values()
        self.app.load_guided_profile()
        self.app.refresh_guided_view()
        self.assertIsNone(self.app.guided.profile)
        self.assertTrue(self.app.test_button.instate(['disabled']))

    def test_named_jonghak_default_takes_priority_over_later_temporary_calibration(self):
        from pose_defaults import load_default_preset
        from pose_calibration import PoseProfile
        from test_pose_calibration import CENTERS, ORDER
        preset = load_default_preset()
        temporary = PoseProfile([CENTERS[name] for name in ORDER])
        temporary.save(self.app.calibration_file, self.app.calibration_context())
        runtime_before = self.app.calibration_file.read_bytes()
        (Path(self.temp.name) / 'preset.json').write_bytes(preset.path.read_bytes())
        (Path(self.temp.name) / 'pose_defaults.json').write_text(json.dumps(dict(
            version=1, enabled=True, name='종학', preset='preset.json')), encoding='utf-8')
        self.app.load_guided_profile()
        self.app.refresh_guided_view()
        np.testing.assert_array_equal(self.app.guided.profile.centers, preset.profile.centers)
        self.assertIn('종학', self.app.status.get())
        self.assertEqual(self.app.calibration_file.read_bytes(), runtime_before)

    def test_wrong_posture_cannot_complete_the_new_calibration_test(self):
        self.complete_guided_calibration()
        self.app.start_calibration_test()
        self.feed_guided('left', 20)
        self.assertEqual(self.app.guided.target, 'front')
        self.assertEqual(self.app.guided.results, [])
        self.app.cancel_calibration()
        self.assertEqual(self.app.guided.mode, 'ready')
        self.assertIsNotNone(self.app.guided.profile)


if __name__ == '__main__':
    unittest.main()
