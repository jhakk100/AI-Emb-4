"""Camera ownership, terminal packet delivery and GUI session regressions."""

from pathlib import Path
import json
from queue import Queue
import sys
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stages'))

import cv2
import numpy as np

import app
from auth_diagnostics import ATTEMPT_LIMIT, AuthDiagnostics
from capture_control import (ACTIVE_PROFILE, CALIBRATED_ACTIVE_PROFILE, ENROLLMENT_PROFILE, STANDBY_PROFILE,
                             configure_capture, EnrollmentResolutionGate)
from database import initialize_database
from pi_camera import PiCamera
from pipeline import FrameResult
from guided_capture import GuidedSettings
from standby import GateSettings
from stage5_cycle import Stage5Cycle, stage5


def result(**kwargs):
    return FrameResult([], False, 'test', 0, 0, 0, 1, 0, **kwargs)


class CameraRuntimeTests(unittest.TestCase):
    def test_diagnostic_worker_records_terminal_authentication_and_can_abort_a_stalled_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = AuthDiagnostics(directory)
            worker = app.CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1))
            worker.commands.put(('diagnostic_start', ('authenticate', 1, diagnostics)))
            capture = Mock()
            capture.read.return_value = True, np.zeros((1080, 1920, 3), np.uint8)
            capture.get.return_value = 30
            pipeline = Mock(profile=STANDBY_PROFILE, threshold=.45)
            pipeline.active = SimpleNamespace(settings=GuidedSettings(), gate_settings=GateSettings(), auth_trace=None)
            terminal = result(authentication_attempted=True)
            waiting = result(power_mode='standby')
            pipeline.process.side_effect = [terminal, waiting]
            second_started = False
            aborted = False

            def wait(delay):
                nonlocal second_started, aborted
                if diagnostics.completed == 1 and not second_started:
                    second_started = True
                    worker.commands.put(('diagnostic_start', ('authenticate', 2, diagnostics)))
                elif diagnostics.completed == 1 and pipeline.process.call_count == 2 and not aborted:
                    aborted = True
                    worker.commands.put(('diagnostic_abort', None))
                elif diagnostics.completed >= 2:
                    worker.stop_event.set()

            with patch('app.ensure_models', return_value=('d', 'r')), patch('app.FaceEngine'), \
                    patch('app.CameraSession', return_value=pipeline), \
                    patch('app.open_camera', return_value=capture), \
                    patch.object(worker.stop_event, 'wait', side_effect=wait):
                worker.run()
            entries = [json.loads(line) for line in diagnostics.path.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(terminal.diagnostic_only)
            self.assertEqual([(entry['attempt'], entry['outcome']) for entry in entries
                              if entry['event'] == 'attempt_end'], [(1, 'failure'), (2, 'incomplete')])
            self.assertEqual([entry['capture_size'] for entry in entries if entry['event'] == 'frame'],
                             [[1920, 1080], [1920, 1080]])
            self.assertIsNone(pipeline.active.auth_trace)

    def test_diagnostic_batch_closes_after_ten_and_keeps_attempt_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = AuthDiagnostics(directory)
            for _ in range(ATTEMPT_LIMIT):
                self.assertTrue(diagnostics.begin(camera=0))
                diagnostics.finish('incomplete', '대기 중')
            self.assertFalse(diagnostics.begin(camera=0))
            rows = [json.loads(line) for line in diagnostics.path.read_text(encoding='utf-8').splitlines()]
            endings = [row for row in rows if row['event'] == 'attempt_end']
            self.assertEqual([row['attempt'] for row in endings], list(range(1, 11)))
            self.assertEqual(rows[-1]['event'], 'batch_end')

    def test_manual_capture_command_reaches_camera_session_and_reports_acknowledgement(self):
        worker = app.CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1), session_id=7)
        worker.commands.put(('manual_capture', (7, 'left', 2, 1.)))
        capture = Mock()
        capture.read.return_value = True, np.zeros((1080, 1920, 3), np.uint8)
        capture.get.return_value = 20
        pipeline = Mock(profile=ENROLLMENT_PROFILE, manual_capture_status='수동 촬영 접수')
        pipeline.request_manual_capture.return_value = True
        pipeline.process.return_value = result(enrollment_completed='new')
        def wait(delay):
            if worker.pause_event.is_set():
                worker.stop_event.set()
        with patch('app.ensure_models', return_value=('d', 'r')), patch('app.FaceEngine'), \
                patch('app.CameraSession', return_value=pipeline), patch('app.open_camera', return_value=capture), \
                patch.object(worker.stop_event, 'wait', side_effect=wait):
            worker.run()
        pipeline.request_manual_capture.assert_called_once_with(7, 'left', 2, 1.)
        events = []
        while not worker.events.empty():
            events.append(worker.events.get_nowait())
        self.assertIn(('manual_capture', (7, True, '수동 촬영 접수')), events)
        capture.release.assert_called_once()
    def test_real_camera_flow_retries_registration_and_saves_all_twelve_without_authentication(self):
        from camera_session import CameraSession
        from liveness import Pose
        from test_secure_pipeline import detection, landmarks
        from face_core import Face

        with tempfile.TemporaryDirectory() as directory:
            store = initialize_database(Path(directory) / 'test.sqlite3')
            worker = app.CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1), purpose='preview')
            worker.commands.put(('purpose', ('enroll', 1, 'new')))
            worker.requested_session = 1
            clock, sessions, profiles, packets = [0.], [], [], []
            row = detection()*np.array([3., 2.25]*7+[1.], np.float32)
            box = tuple(row[:4])
            vector = np.zeros(128, np.float32)
            vector[0] = 1
            engine = Mock()
            engine.analyze.return_value = [Face(box, vector, 100)]
            mesh = Mock()
            mesh.observe.return_value = Pose(0, 0, 0, box, landmarks())
            mismatch_once, retried = [False], [False]

            def make_session(*args, **kwargs):
                kwargs.update(store=store)
                session = CameraSession(*args, **kwargs)
                sessions.append(session)
                return session

            def analyze(*args):
                challenge = sessions[0].active.challenge
                if challenge.samples and not mismatch_once[0]:
                    mismatch_once[0] = True
                    return [Face(box, np.roll(vector, 1), 100)]
                return [Face(box, vector, 100)]

            native = np.zeros((1080, 1920, 3), np.uint8)
            reads = [0]
            def read():
                reads[0] += 1
                clock[0] += .15
                profiles.append(sessions[0].profile)
                if reads[0] > 200:
                    raise RuntimeError('registration did not finish')
                return True, np.zeros((480, 640, 3), np.uint8) if reads[0] == 1 else native

            def wait(delay):
                if worker.pause_event.is_set():
                    packets.append(worker.frames.get_nowait()[-1])
                    if packets[-1].enrollment_error and not retried[0]:
                        retried[0] = True
                        worker.requested_session = 2
                        worker.commands.put(('retry_enrollment', 2))
                    else:
                        worker.stop_event.set()
                elif not worker.frames.empty():
                    packet = worker.frames.get_nowait()[-1]
                    if packet.direction:
                        worker.commands.put(('manual_capture', (packet.session_id, packet.direction,
                                                                 packet.enrollment_count, clock[0])))

            capture = Mock()
            capture.read.side_effect = read
            capture.get.return_value = 20
            engine.detect.return_value = np.array([row])
            engine.analyze.side_effect = analyze
            with patch('app.ensure_models', return_value=('d', 'r')), \
                    patch('app.FaceEngine', return_value=engine), \
                    patch('app.CameraSession', side_effect=make_session), \
                    patch('enrollment.FaceMeshEstimator', return_value=mesh), \
                    patch('app.open_camera', return_value=capture), \
                    patch('enrollment.monotonic', side_effect=lambda: clock[0]), \
                    patch('standby.time.monotonic', side_effect=lambda: clock[0]), \
                    patch.object(worker.stop_event, 'wait', side_effect=wait):
                worker.run()
            self.assertTrue(mismatch_once[0] and retried[0])
            self.assertEqual(len(packets), 2)
            self.assertTrue(packets[0].enrollment_error)
            self.assertEqual(packets[1].enrollment_completed, 'new')
            self.assertEqual(packets[1].enrollment_count, 12)
            self.assertTrue(all(profile == ENROLLMENT_PROFILE for profile in profiles))
            self.assertTrue(all(not packet.authentication_attempted for packet in packets))
            self.assertEqual(len(store.directional_samples('new')), 12)
            self.assertEqual(store.count_observations(), 0)
            capture.release.assert_called_once()
            mesh.close.assert_called_once()

    def test_worker_requests_fhd_and_never_processes_old_buffered_vga(self):
        worker = app.CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1))
        capture = Mock()
        capture.read.side_effect = [(True, np.zeros((480, 640, 3), np.uint8)),
                                    (True, np.zeros((1080, 1920, 3), np.uint8))]
        capture.get.return_value = 20
        pipeline = Mock(profile=ENROLLMENT_PROFILE)
        pipeline.process.return_value = result(enrollment_completed='new', power_mode='enrollment',
                                                processing_size=(1920, 1080))
        def wait(delay):
            if worker.pause_event.is_set():
                worker.stop_event.set()
        with patch('app.ensure_models', return_value=('d', 'r')), patch('app.FaceEngine'), \
                patch('app.CameraSession', return_value=pipeline), \
                patch('app.open_camera', return_value=capture) as opening, \
                patch.object(worker.stop_event, 'wait', side_effect=wait):
            worker.run()
        opening.assert_called_once_with(0, size=(1920, 1080), fps=30)
        pipeline.process.assert_called_once()
        self.assertEqual(pipeline.process.call_args.args[0].shape, (1080, 1920, 3))
        self.assertEqual(worker.frames.get_nowait()[-1].capture_size, (1920, 1080))
        capture.release.assert_called_once()

    def test_registration_discards_buffered_low_resolution_then_accepts_fhd(self):
        gate = EnrollmentResolutionGate()
        self.assertFalse(gate.accept(np.zeros((480, 640, 3), np.uint8), ENROLLMENT_PROFILE))
        self.assertTrue(gate.accept(np.zeros((1080, 1920, 3), np.uint8), ENROLLMENT_PROFILE))
        self.assertEqual(gate.misses, 0)

    def test_unsupported_fhd_camera_fails_instead_of_upscaling(self):
        gate = EnrollmentResolutionGate(max_misses=2)
        low = np.zeros((720, 1280, 3), np.uint8)
        self.assertFalse(gate.accept(low, ENROLLMENT_PROFILE))
        with self.assertRaisesRegex(RuntimeError, 'FHD'):
            gate.accept(low, ENROLLMENT_PROFILE)

    def test_terminal_packet_is_not_overwritten_and_resources_close(self):
        worker = app.CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1))
        capture = Mock()
        capture.read.return_value = True, np.zeros((1080, 1920, 3), dtype=np.uint8)
        capture.get.return_value = 30
        pipeline = Mock(profile=STANDBY_PROFILE)
        terminal = result(authentication_attempted=True)
        pipeline.process.return_value = terminal
        def wait(delay):
            if worker.pause_event.is_set():
                worker.stop_event.set()
        with patch('app.ensure_models', return_value=('d', 'r')), \
                patch('app.FaceEngine'), patch('app.CameraSession', return_value=pipeline), \
                patch('app.open_camera', return_value=capture), \
                patch.object(worker.stop_event, 'wait', side_effect=wait):
            worker.run()
        capture.read.assert_called_once()
        self.assertIs(worker.frames.get_nowait()[-1], terminal)
        capture.release.assert_called_once()
        pipeline.close.assert_called_once()

    def test_camera_worker_keeps_one_fhd_stream_across_standby_auth_and_enrollment(self):
        worker = app.CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1))
        capture = Mock()
        capture.read.return_value = True, np.zeros((1080, 1920, 3), np.uint8)
        pipeline = Mock(profile=STANDBY_PROFILE)
        seen = []
        next_profiles = iter((CALIBRATED_ACTIVE_PROFILE, ENROLLMENT_PROFILE, STANDBY_PROFILE))

        def process(frame):
            seen.append((pipeline.profile, frame.shape))
            pipeline.profile = next(next_profiles)
            return result(enrollment_completed='new' if len(seen) == 3 else None)

        pipeline.process.side_effect = process
        def wait(delay):
            if worker.pause_event.is_set():
                worker.stop_event.set()
        with patch('app.ensure_models', return_value=('d', 'r')), patch('app.FaceEngine'), \
                patch('app.CameraSession', return_value=pipeline), \
                patch('app.open_camera', return_value=capture) as opening, \
                patch.object(worker.stop_event, 'wait', side_effect=wait):
            worker.run()
        opening.assert_called_once_with(0, size=(1920, 1080), fps=30)
        self.assertEqual(seen, [(profile, (1080, 1920, 3)) for profile in
                                (STANDBY_PROFILE, CALIBRATED_ACTIVE_PROFILE, ENROLLMENT_PROFILE)])
        capture.set.assert_not_called()
        capture.configure_profile.assert_not_called()

    def test_console_camera_resizes_frames_in_software_without_reconfiguring_the_stream(self):
        import capture_control
        from integrate import Camera
        capture = Mock()
        capture.read.return_value = True, np.zeros((1080, 1920, 3), np.uint8)
        opening = Mock(return_value=capture)
        stages = SimpleNamespace(cv2=cv2, capture_control=capture_control,
                                 face_core=SimpleNamespace(open_camera=opening))
        camera = Camera(stages, 0, flip=False)
        self.addCleanup(camera.release)
        for profile in (STANDBY_PROFILE, CALIBRATED_ACTIVE_PROFILE, ACTIVE_PROFILE, ENROLLMENT_PROFILE):
            self.assertEqual(camera.read(profile).shape[:2], profile.size[::-1])
        opening.assert_called_once_with(0, size=(1920, 1080), fps=30)
        capture.set.assert_not_called()
        capture.configure_profile.assert_not_called()

    def test_camera_failure_closes_pipeline_and_reports_stopped(self):
        worker = app.CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1))
        pipeline = Mock(profile=STANDBY_PROFILE)
        with patch('app.ensure_models', return_value=('d', 'r')), patch('app.FaceEngine'), \
                patch('app.CameraSession', return_value=pipeline), \
                patch('app.open_camera', side_effect=RuntimeError('unavailable')):
            worker.run()
        pipeline.close.assert_called_once()
        events = []
        while not worker.events.empty():
            events.append(worker.events.get_nowait()[0])
        self.assertIn('error', events)
        self.assertEqual(events[-1], 'stopped')

    def test_pi_profile_changes_once_and_release_is_idempotent(self):
        device = Mock()
        device.capture_array.return_value = np.zeros((180, 320, 3), np.uint8)
        camera = PiCamera(device, (320, 180), 5)
        configure_capture(camera, (640, 480), 20)
        self.assertEqual(device.configure.call_count, 2)
        self.assertEqual(camera.get(cv2.CAP_PROP_FPS), 20)
        self.assertTrue(camera.read()[0])
        camera.release()
        camera.release()
        device.close.assert_called_once()

    def test_pi_start_failure_still_closes_camera(self):
        device = Mock()
        device.start.side_effect = RuntimeError('start failed')
        with self.assertRaises(RuntimeError):
            PiCamera(device, (320, 180), 5)
        device.close.assert_called_once()

    def test_result_processing_cannot_be_reset_while_worker_is_running(self):
        cycle = Stage5Cycle.__new__(Stage5Cycle)
        cycle.phase = 'processing'
        cycle._worker = Mock()
        cycle._worker.is_alive.return_value = True
        self.assertFalse(cycle.reset())
        self.assertEqual(cycle.phase, 'processing')

    def test_cancelling_manual_wait_does_not_wait_for_timeout_or_open(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(stage5.SET5, log_path=Path(directory) / 'test.csv',
                            enable_voice=False, manual_timeout=60)
            button = Mock()
            handler = stage5.ResultHandler(button=button, settings=settings)
            entered = threading.Event()
            outcomes = []
            handler._beep = lambda seconds: entered.set()
            thread = threading.Thread(target=lambda: outcomes.append(handler._manual()))
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                handler.cancel_event.set()
                thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertEqual(outcomes, [stage5.Action.LOCKED])
                self.assertIsNone(button.when_pressed)
                self.assertNotIn('MANUAL_OPEN', settings['log_path'].read_text(encoding='utf-8-sig'))
            finally:
                handler.cancel_event.set()
                thread.join(2)
                handler.close()

    def test_stage5_is_inert_while_suspended_for_registration(self):
        cycle = Stage5Cycle.__new__(Stage5Cycle)
        cycle._worker, cycle._suspended = None, False
        cycle.phase, cycle.completed = 'waiting', False
        cycle.handler, cycle.button = Mock(), Mock()
        self.assertTrue(cycle.suspend())
        self.assertFalse(cycle.start(0, 'owner', .9))
        self.assertFalse(cycle.reset())
        self.assertFalse(cycle.press_manual())
        self.assertIsNone(cycle.advance(0))
        cycle.handler.handle.assert_not_called()
        cycle.button.press.assert_not_called()
        cycle.resume()
        self.assertTrue(cycle.reset())


class GuiSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'GUI 세션이 없는 환경: {exc}')
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.door = Mock(phase='waiting', completed=False, processing=False)
        self.door.advance.return_value = None
        store = initialize_database(Path(self.temp.name) / 'test.sqlite3')
        with patch('app.Stage5Cycle', return_value=self.door):
            self.ui = app.FaceApp(self.root, store=store, use_gpio=False, enable_voice=False)
        self.root.update_idletasks()
        self.ui.worker = Mock()
        self.ui.worker.stop_event = threading.Event()
        self.ui.worker.commands = Queue()
        self.ui.worker.pause_event = threading.Event()

    def test_tab_switch_invalidates_old_packets_and_resets_challenge(self):
        old = self.ui.session_id
        self.ui.tabs.select(1)
        self.ui._tab_changed()
        self.assertGreater(self.ui.session_id, old)
        command, (purpose, session_id) = self.ui.worker.commands.get_nowait()
        self.assertEqual((command, purpose), ('purpose', 'preview'))
        self.assertEqual(session_id, self.ui.session_id)
        stale = result(session_id=old)
        self.ui.frames.put((None, [], 0, 0, '', False, 0, stale))
        with patch.object(self.ui, '_handle_frame') as handle:
            self.ui._poll_once()
        handle.assert_not_called()

    def test_failed_liveness_is_forwarded_even_without_a_face_embedding(self):
        self.ui.store.people = {'owner': {}}
        self.ui.tabs.select(0)
        telemetry = result(authentication_attempted=True, session_id=self.ui.session_id)
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        self.ui._handle_frame(frame, [], 1., 5., 'tracking lost', True, 0, telemetry)
        self.door.start.assert_called_once()
        self.assertIsNone(self.door.start.call_args.args[1])
        self.assertTrue(self.ui.worker.pause_event.is_set())

    def test_database_error_does_not_break_poll_scheduling(self):
        import sqlite3
        with patch.object(self.ui, '_poll_once', side_effect=sqlite3.OperationalError('locked')), \
                patch.object(self.root, 'after') as after:
            self.ui._poll()
        self.assertTrue(self.ui.worker.stop_event.is_set())
        after.assert_called_once()

    def test_threshold_change_invalidates_pending_authentication(self):
        old = self.ui.session_id
        self.ui.apply_threshold('.6')
        self.assertGreater(self.ui.session_id, old)
        self.assertEqual(self.ui.worker.commands.get_nowait(), ('threshold', .6))
        self.assertEqual(self.ui.worker.commands.get_nowait()[0], 'purpose')

    def test_guidance_draws_native_arrows_for_each_direction(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for direction in ('front', 'up', 'left', 'right'):
            telemetry = result()
            telemetry.direction = direction
            telemetry.arrow = direction
            self.ui._handle_frame(frame, [], 1., 5., 'direction prompt', False, 0, telemetry)
            types = [self.ui.canvas.type(item) for item in self.ui.canvas.find_all()]
            self.assertIn('oval' if direction == 'front' else 'line', types)

    def test_authentication_guide_stays_centered_and_diagnostic_result_skips_door(self):
        self.ui.tabs.select(0)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for telemetry in (result(power_mode='standby'), result(direction='up'),
                          result(authentication_attempted=True, diagnostic_only=True)):
            self.ui._handle_frame(frame, [], 1., 30., 'test', False, 0, telemetry)
            x1, y1, x2, y2 = self.ui.canvas.coords('authentication_face_guide')
            self.assertEqual(((x1+x2)/2, (y1+y2)/2), (320, 240))
        self.door.start.assert_not_called()

    def test_review_window_labels_an_attempt_by_the_selected_actual_person(self):
        from test_secure_pipeline import profile, sample
        self.ui.store.enroll_directional('owner', profile())
        calibration = self.ui.store.calibration
        attempt = calibration.begin('test', .45)
        calibration.record_scores(attempt, sample().embedding)
        calibration.finish(attempt, 'rejected', '방향 촬영 시간 초과')
        self.ui.show_auth_review()
        review = self.ui.auth_review
        review.window.withdraw()
        review.table.selection_set(str(attempt.id))
        review._select()
        review.person.set('owner')
        review._save('registered')
        current, = calibration.recent_attempts()
        self.assertEqual(current['outcome'], 'rejected')
        self.assertEqual(current['verified_name'], 'owner')
        self.assertEqual(calibration.status()['genuine'], 1)
        self.assertIn('본인 1/10', self.ui.calibration_text.get())

    def test_auth_hint_is_inside_camera_frame_and_clears_without_moving_oval(self):
        self.ui.tabs.select(0)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        hint = result(direction='left', pose_hint='raise_chin')
        self.ui._handle_frame(frame, [], 1., 30., 'test', False, 0, hint)
        canvas = self.ui.canvas
        item, = canvas.find_withtag('authentication_pose_hint')
        self.assertEqual(canvas.itemcget(item, 'text'), '턱을 살짝 들어주세요')
        self.assertTrue(60 < canvas.coords(item)[1] < 420)
        guide = canvas.coords('authentication_face_guide')
        self.ui._handle_frame(frame, [], 1.1, 30., 'test', False, 0, result(direction='up'))
        self.assertEqual(canvas.find_withtag('authentication_pose_hint'), ())
        self.assertEqual(canvas.coords('authentication_face_guide'), guide)

    def test_registration_sends_name_to_worker_instead_of_collecting_on_ui(self):
        import time
        self.ui.gate_ready = False
        self.ui.last_frame_at = time.monotonic()
        self.ui.name.set('new')
        self.ui.register()
        command, payload = self.ui.worker.commands.get_nowait()
        self.assertEqual(command, 'purpose')
        self.assertEqual(payload[0], 'enroll')
        self.assertEqual(payload[2], 'new')

    def _start_registration(self):
        import time
        self.ui.gate_ready = True
        self.ui.last_frame_at = time.monotonic()
        self.ui.name.set('new')
        self.ui.register()

    def test_registration_locks_tabs_and_restores_the_registration_tab(self):
        self._start_registration()
        epoch = self.ui.session_id
        self.assertEqual(self.ui.tabs.tab(self.ui.auth_tab, 'state'), 'disabled')
        self.assertEqual(self.ui.tabs.tab(self.ui.history_tab, 'state'), 'disabled')
        self.ui._tab_changed()
        self.assertIsNotNone(self.ui.enrollment)
        self.assertEqual(self.ui.tabs.select(), str(self.ui.enroll_tab))
        self.assertEqual(self.ui.session_id, epoch)
        self.assertTrue(self.ui.grid_button.instate(['disabled']))
        self.assertFalse(self.ui._set_purpose('authenticate'))
        self.assertEqual(self.ui.session_id, epoch)

    def test_registration_displays_twelve_photos_and_a_centered_face_guide(self):
        self._start_registration()
        for count in (0, 1):
            telemetry = result(purpose='enroll', session_id=self.ui.session_id, direction='front',
                               enrollment_count=count)
            self.ui._handle_enrollment_frame(np.zeros((1080, 1920, 3), np.uint8), 1., 20., telemetry)
            texts = [self.ui.canvas.itemcget(item, 'text') for item in self.ui.canvas.find_all()
                     if self.ui.canvas.type(item) == 'text']
            self.assertTrue(any(f'{count}/12장' in text for text in texts))
            x1, y1, x2, y2 = self.ui.canvas.coords('enrollment_face_guide')
            self.assertEqual(((x1+x2)/2, (y1+y2)/2), (320, 240))
        self.door.start.assert_not_called()
        self.assertEqual(self.ui.progress['maximum'], 12)

    def test_explicit_registration_cancel_unlocks_ui_and_notifies_worker(self):
        self._start_registration()
        self.ui.worker.commands.get_nowait()
        self.ui.cancel_enrollment()
        self.assertIsNone(self.ui.enrollment)
        self.assertEqual(self.ui.tabs.tab(self.ui.auth_tab, 'state'), 'normal')
        self.assertEqual(self.ui.worker.commands.get_nowait()[0], 'cancel_enrollment')

    def test_threshold_changes_are_ignored_during_registration(self):
        self._start_registration()
        epoch = self.ui.session_id
        self.ui.apply_threshold('.7')
        self.assertEqual(self.ui.session_id, epoch)
        self.assertEqual(self.ui.threshold.get(), .45)
        self.assertTrue(self.ui.threshold_slider.instate(['disabled']))

    def enrollment_packet(self, **kwargs):
        telemetry = result(purpose='enroll', power_mode='enrollment',
                           processing_size=(1920, 1080), session_id=self.ui.session_id, **kwargs)
        self.ui.frames.put((np.zeros((1080, 1920, 3), np.uint8), [],
                            time.monotonic(), 20., telemetry.status, False, 0, telemetry))
        self.ui._poll_once()
        return telemetry

    def test_enrollment_failure_stays_in_registration_and_retry_keeps_ui_locked(self):
        self._start_registration()
        self.ui.worker.commands.get_nowait()
        self.enrollment_packet(enrollment_error='얼굴 소실')
        self.assertEqual(self.ui.enrollment.error, '얼굴 소실')
        self.assertEqual(self.ui.tabs.tab(self.ui.auth_tab, 'state'), 'disabled')
        self.assertTrue(self.ui.worker.commands.empty())
        self.assertIn('얼굴등록', self.ui.indicators.phase.get())
        self.assertIn('FHD', self.ui.power_text.get())
        self.assertTrue(self.ui.retry_button.instate(['!disabled']))
        self.door.start.assert_not_called()
        self.door.advance.assert_not_called()
        old = self.ui.session_id
        self.ui.restart_enrollment()
        self.assertEqual(self.ui.worker.commands.get_nowait(), ('retry_enrollment', old+1))
        self.assertEqual(self.ui.tabs.tab(self.ui.auth_tab, 'state'), 'disabled')
        self.assertIsNone(self.ui.enrollment.error)

    def test_enrollment_completion_waits_for_exit_and_does_not_enter_stage5(self):
        self._start_registration()
        self.ui.worker.commands.get_nowait()
        self.enrollment_packet(enrollment_completed='new', enrollment_count=12)
        self.assertTrue(self.ui.enrollment.completed)
        self.assertTrue(self.ui.worker.commands.empty())
        self.assertIn('나가기', self.ui.cancel_button.cget('text'))
        self.assertEqual(self.ui.tabs.tab(self.ui.auth_tab, 'state'), 'disabled')
        self.assertEqual(self.ui.progress['value'], 12)
        self.door.start.assert_not_called()
        self.door.advance.assert_not_called()
        self.ui.cancel_enrollment()
        self.assertIsNone(self.ui.enrollment)
        self.door.resume.assert_called_once()

    def test_delayed_authentication_result_cannot_escape_registration_even_in_same_epoch(self):
        self._start_registration()
        self.ui.store.people = {'owner': {}}
        telemetry = result(session_id=self.ui.session_id, authentication_attempted=True,
                           authenticated_name='owner', liveness_passed=True)
        self.ui._handle_frame(np.zeros((480, 640, 3), np.uint8), [], 1., 20., telemetry=telemetry)
        self.assertIsNotNone(self.ui.enrollment)
        self.door.start.assert_not_called()
        self.assertIn('얼굴등록', self.ui.indicators.phase.get())

    def test_late_stage5_event_and_stage_labels_do_not_reset_enrollment(self):
        self._start_registration()
        epoch = self.ui.session_id
        self.door.advance.return_value = 'finished'
        self.ui.stages.put_nowait('1단계 절전 대기')
        self.ui._poll_once()
        self.door.advance.assert_not_called()
        self.assertFalse(self.ui.worker.stop_event.is_set())
        self.assertEqual(self.ui.session_id, epoch)
        self.ui.indicators.set_stage(4, '인증 실패', 'red')
        self.ui.indicators.idle('1단계 절전')
        self.assertIn('실행 안 함', self.ui.indicators.details[3].get())
        self.assertIn('얼굴등록', self.ui.indicators.phase.get())

    def test_camera_error_and_stopped_event_keep_enrollment_pending(self):
        self._start_registration()
        self.ui.enrollment.error = 'FHD 입력 불가'
        self.ui.events.put(('stopped', ''))
        self.ui._poll_once()
        self.assertIsNone(self.ui.worker)
        self.assertIsNotNone(self.ui.enrollment)
        self.assertEqual(self.ui.tabs.tab(self.ui.auth_tab, 'state'), 'disabled')
        self.assertIn('얼굴등록', self.ui.indicators.phase.get())
        self.assertTrue(self.ui.retry_button.instate(['!disabled']))
        self.door.resume.assert_not_called()

    def test_duplicate_enrollment_request_does_not_invalidate_active_epoch(self):
        self._start_registration()
        epoch = self.ui.session_id
        self.assertFalse(self.ui._set_purpose('enroll'))
        self.assertEqual(self.ui.session_id, epoch)

    def test_capturing_longer_than_two_seconds_keeps_registration_display(self):
        self._start_registration()
        self.ui.last_frame_at = time.monotonic()-5
        self.ui.result.set('얼굴등록 · 사진 분석 중')
        self.ui._poll_once()
        self.assertIn('얼굴등록', self.ui.result.get())
        self.assertEqual(self.ui.indicators.color, 'purple')

    def test_named_default_is_visible_and_grid_viewer_shows_both_model_types(self):
        from test_secure_pipeline import profile
        self.assertIn('종학', self.ui.preset_text.get())
        self.ui.store.enroll_directional('owner', profile())
        self.ui._refresh_people()
        self.ui.people.selection_set(0)
        self.ui.show_face_grid()
        windows = [child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel)]
        self.assertEqual(len(windows), 1)
        self.addCleanup(windows[0].destroy)
        self.assertIn('통합 3D', windows[0].title())
        self.assertTrue(hasattr(windows[0], 'grid_photo'))
        self.assertTrue(hasattr(windows[0], 'fused_photo'))

    def test_manual_p_and_button_capture_one_current_enrollment_frame_without_changing_stage(self):
        self._start_registration()
        self.ui.worker.commands.get_nowait()
        self.enrollment_packet(direction='front')
        self.assertTrue(self.ui.manual_capture_button.instate(['!disabled']))
        epoch = self.ui.session_id
        event = SimpleNamespace(keysym='P', char='P', keycode=80)
        with patch('app.time.monotonic', return_value=self.ui.last_frame_at+.01):
            self.assertEqual(self.ui._enrollment_key_press(event), 'break')
            self.ui._enrollment_key_press(event)
        command, payload = self.ui.worker.commands.get_nowait()
        self.assertEqual((command, payload[:3]), ('manual_capture', (epoch, 'front', 0)))
        self.assertTrue(self.ui.worker.commands.empty())
        self.assertEqual(self.ui.session_id, epoch)
        self.door.start.assert_not_called()
        self.assertEqual(self.ui.tabs.select(), str(self.ui.enroll_tab))
        self.ui._enrollment_key_release(event)
        self.enrollment_packet(direction='front', enrollment_count=1)
        self.assertTrue(self.ui.request_enrollment_photo())
        self.assertEqual(self.ui.worker.commands.get_nowait()[1][:3], (epoch, 'front', 1))

    def test_manual_capture_is_disabled_for_preparation_failure_completion_and_other_stages(self):
        self.assertFalse(self.ui.request_enrollment_photo())
        self.assertIsNone(self.ui._enrollment_key_press(SimpleNamespace(keysym='p')))
        self._start_registration()
        self.ui.worker.commands.get_nowait()
        self.assertFalse(self.ui.request_enrollment_photo())
        self.assertTrue(self.ui.manual_capture_button.instate(['disabled']))
        self.enrollment_packet(enrollment_error='lost')
        self.assertFalse(self.ui.request_enrollment_photo())
        self.assertTrue(self.ui.manual_capture_button.instate(['disabled']))
        self.ui.enrollment.error = None
        self.enrollment_packet(enrollment_completed='new', enrollment_count=12)
        self.assertFalse(self.ui.request_enrollment_photo())
        self.assertTrue(self.ui.manual_capture_button.instate(['disabled']))
        self.assertTrue(self.ui.worker.commands.empty())

    def test_manual_capture_rejects_stale_preview_and_ignores_old_session_ack(self):
        self._start_registration()
        self.ui.worker.commands.get_nowait()
        self.enrollment_packet(direction='front')
        with patch('app.time.monotonic', return_value=self.ui.last_frame_at+1.):
            self.assertFalse(self.ui.request_enrollment_photo())
        self.assertTrue(self.ui.worker.commands.empty())
        before = self.ui.manual_feedback.get()
        self.ui.events.put(('manual_capture', (self.ui.session_id-1, True, 'stale')))
        self.ui._poll_once()
        self.assertEqual(self.ui.manual_feedback.get(), before)

    def test_manual_capture_result_and_rejection_are_shown_without_discarding_registration(self):
        self._start_registration()
        self.ui.worker.commands.get_nowait()
        self.enrollment_packet(direction='front', enrollment_count=1, manual_captured=True,
                               manual_capture_status='수동 촬영 완료 · 정면 · 전체 1/12장')
        self.assertIn('수동 촬영 완료', self.ui.manual_feedback.get())
        self.ui.events.put(('manual_capture', (self.ui.session_id, False, '촬영 간격을 기다리세요')))
        self.ui._poll_once()
        self.assertIn('간격', self.ui.manual_feedback.get())
        self.assertIsNotNone(self.ui.enrollment)
        self.assertEqual(len(self.ui.enrollment.samples), 1)
        self.door.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
