"""Camera ownership, terminal packet delivery and GUI session regressions."""

from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
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
from capture_control import (STANDBY_PROFILE, ENROLLMENT_PROFILE, configure_capture,
                             EnrollmentResolutionGate)
from database import initialize_database
from pi_camera import PiCamera
from pipeline import FrameResult
from stage5_cycle import Stage5Cycle, stage5


def result(**kwargs):
    return FrameResult([], False, 'test', 0, 0, 0, 1, 0, **kwargs)


class CameraRuntimeTests(unittest.TestCase):
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
                patch('app.FacePipeline', return_value=pipeline), \
                patch('app.open_camera', return_value=capture) as opening, \
                patch.object(worker.stop_event, 'wait', side_effect=wait):
            worker.run()
        opening.assert_called_once_with(0, size=(1920, 1080), fps=20)
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
        capture.read.return_value = True, np.zeros((180, 320, 3), dtype=np.uint8)
        capture.get.return_value = 5
        pipeline = Mock(profile=STANDBY_PROFILE)
        terminal = result(authentication_attempted=True)
        pipeline.process.return_value = terminal
        def wait(delay):
            if worker.pause_event.is_set():
                worker.stop_event.set()
        with patch('app.ensure_models', return_value=('d', 'r')), \
                patch('app.FaceEngine'), patch('app.FacePipeline', return_value=pipeline), \
                patch('app.open_camera', return_value=capture), \
                patch.object(worker.stop_event, 'wait', side_effect=wait):
            worker.run()
        capture.read.assert_called_once()
        self.assertIs(worker.frames.get_nowait()[-1], terminal)
        capture.release.assert_called_once()
        pipeline.close.assert_called_once()

    def test_camera_failure_closes_pipeline_and_reports_stopped(self):
        worker = app.CameraWorker(0, Queue(maxsize=1), Queue(), Queue(maxsize=1))
        pipeline = Mock(profile=STANDBY_PROFILE)
        with patch('app.ensure_models', return_value=('d', 'r')), patch('app.FaceEngine'), \
                patch('app.FacePipeline', return_value=pipeline), \
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
        for direction in ('front', 'up', 'down', 'left', 'right'):
            telemetry = result()
            telemetry.direction = direction
            telemetry.arrow = direction
            self.ui._handle_frame(frame, [], 1., 5., 'direction prompt', False, 0, telemetry)
            types = [self.ui.canvas.type(item) for item in self.ui.canvas.find_all()]
            self.assertIn('oval' if direction == 'front' else 'line', types)

    def test_registration_sends_name_to_worker_instead_of_collecting_on_ui(self):
        import time
        self.ui.gate_ready = True
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
        self.assertFalse(self.ui._set_purpose('authenticate'))
        self.assertEqual(self.ui.session_id, epoch)

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


if __name__ == '__main__':
    unittest.main()
