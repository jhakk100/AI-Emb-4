"""Shared-camera UI: four directions x three registration photos, two-photo authentication.

YuNet guides movement; captured photos run SFace and Face Mesh in the camera worker.
Enrollment and authentication have separate processors owned by CameraSession.
Stage5Cycle accepts authentication results only and is suspended during enrollment.
"""

import argparse
from io import BytesIO
from contextlib import ExitStack
from dataclasses import asdict
import math
import sqlite3
from types import SimpleNamespace
from queue import Empty, Full, Queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "stages"))  # 1~4단계 모듈(임시통합에서 복사)

from face_core import DEFAULT_THRESHOLD, FaceEngine, REGISTRATION_SAMPLES, open_camera
from models import ensure_models
from authenticate import authenticate_face
from database import delete_user, initialize_database
from stage5_cycle import Stage5Cycle
from capture_control import (CAMERA_FPS, CAMERA_SIZE, STANDBY_PROFILE, frame_delay, prepare_frame,
                             EnrollmentResolutionGate)
from camera_session import CameraSession
from enrollment import FACE_GUIDE_RADII
from guided_capture import LABELS
from status_panel import StatusPanel
from face_grid import draw_mesh_preview, render_grids, render_fused_model, grid_triangles
from pose_defaults import load_default_preset
from auth_diagnostics import ATTEMPT_LIMIT, AuthDiagnostics
from auth_review import AuthenticationReview, policy_text

POSE_HINT_TEXT = {
    'raise_chin': '턱을 살짝 들어주세요',
    'lower_chin': '턱을 살짝 내려주세요',
    'turn_left': '고개를 화면 왼쪽으로 조금 돌려주세요',
    'turn_right': '고개를 화면 오른쪽으로 조금 돌려주세요',
    'level_head': '옆으로 기울인 고개를 바로 세워주세요',
}


class CameraWorker(threading.Thread):
    def __init__(self, index, frames, events, stages, save_interval=1.0, threshold=DEFAULT_THRESHOLD,
                 purpose="authenticate", session_id=0, enrollment_name=None):
        super().__init__(daemon=True)
        self.index, self.frames, self.events = index, frames, events
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.stages, self.commands = stages, Queue()
        self.save_interval = save_interval
        self.threshold = threshold
        self.purpose, self.session_id = purpose, session_id
        self.requested_session = session_id
        self.enrollment_name = enrollment_name

    def _stage(self, text):
        try:
            self.stages.put_nowait(text)
        except Full:
            try:
                self.stages.get_nowait()
            except Empty:
                pass
            self.stages.put_nowait(text)

    def run(self):
        capture = pipeline = None
        diagnostics = None
        last_diagnostic_status = ''

        def finish_diagnostic(outcome, reason):
            nonlocal diagnostics
            if diagnostics is not None:
                recorder, diagnostics = diagnostics, None
                try:
                    if outcome in ('incomplete', 'error'):
                        pipeline.finish_auth_attempt('interrupted' if outcome == 'incomplete' else 'error', reason)
                finally:
                    recorder.finish(outcome, reason)
                    pipeline.active.auth_trace = None
                    self.events.put(('diagnostic_finished', (recorder, outcome)))

        try:
            self._stage("모델 준비 / 카메라 연결")
            paths = ensure_models(lambda text: self.events.put(("status", text)))
            if self.stop_event.is_set():
                return
            engine = FaceEngine(*paths)
            pipeline = CameraSession(engine, source=f"camera:{self.index}",
                                    save_interval=self.save_interval, progress=self._stage,
                                    threshold=self.threshold, purpose=self.purpose,
                                    session_id=self.session_id, enrollment_name=self.enrollment_name,
                                    session_valid=lambda: self.requested_session == pipeline.session_id
                                        and not self.stop_event.is_set())
            self.events.put(("status", f"카메라 {self.index} 연결 중…"))
            capture = open_camera(self.index, size=CAMERA_SIZE, fps=CAMERA_FPS)
            failures = 0
            previous_frame_at = None
            previous_processing_at = None
            resolution_gate = EnrollmentResolutionGate()
            while not self.stop_event.is_set():
                while True:
                    try:
                        setting, value = self.commands.get_nowait()
                    except Empty:
                        break
                    if setting == "save_interval":
                        pipeline.save_interval = value
                    elif setting == "threshold":
                        finish_diagnostic('incomplete', '유사도 기준 변경')
                        pipeline.threshold = value
                    elif setting == "reset_gate":
                        pipeline.reset_standby()
                    elif setting == "retry_enrollment":
                        if pipeline.restart_enrollment(value):
                            self.pause_event.clear()
                    elif setting == 'manual_capture':
                        session_id, direction, count, requested_at = value
                        accepted = pipeline.request_manual_capture(session_id, direction, count, requested_at)
                        self.events.put(('manual_capture', (session_id, accepted, pipeline.manual_capture_status)))
                    elif setting == 'diagnostic_abort':
                        finish_diagnostic('incomplete', last_diagnostic_status or '사용자가 시도를 중단함')
                        self.pause_event.set()
                    elif setting in ("purpose", "cancel_enrollment", "diagnostic_start"):
                        purpose, session_id, *name = value
                        finish_diagnostic('incomplete', '인증 화면 또는 실행 상태 변경')
                        accepted = pipeline.set_purpose(purpose, session_id,
                                                        enrollment_name=name[0] if purpose == 'enroll' and name else None,
                                                        cancel_enrollment=setting == 'cancel_enrollment')
                        if accepted:
                            self.pause_event.clear()
                        if setting == 'diagnostic_start':
                            diagnostics = name[0]
                            last_diagnostic_status = ''
                            threshold = pipeline.begin_auth_attempt(diagnostic=True) if accepted else None
                            if accepted and diagnostics.begin(session_id=session_id, camera=self.index,
                                                              camera_size=CAMERA_SIZE, target_fps=CAMERA_FPS,
                                                              embedding_threshold=pipeline.threshold,
                                                              front_threshold=threshold if isinstance(threshold, (int, float)) else None,
                                                              settings=asdict(pipeline.active.settings),
                                                              standby_settings=asdict(pipeline.active.gate_settings)):
                                pipeline.active.auth_trace = diagnostics.write
                            else:
                                finish_diagnostic('incomplete', '진단을 시작할 수 없습니다')
                                self.pause_event.set()
                if self.pause_event.is_set():
                    self.stop_event.wait(.05)
                    continue
                started = time.monotonic()
                profile = pipeline.profile
                read_started = time.monotonic() if diagnostics else None
                ok, frame = capture.read()
                read_ms = round((time.monotonic()-read_started)*1000, 2) if diagnostics else None
                if not ok or frame is None:
                    if diagnostics:
                        diagnostics.write('camera_read', passed=False, read_ms=read_ms)
                    pipeline.reset_standby()
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("카메라 영상이 끊겼습니다. 다른 앱의 카메라 사용 여부를 확인하세요.")
                    self.stop_event.wait(0.1)
                    continue
                failures = 0
                if not resolution_gate.accept(frame, STANDBY_PROFILE):
                    if diagnostics:
                        diagnostics.write('camera_read', passed=False, read_ms=read_ms,
                                          size=(frame.shape[1], frame.shape[0]), reason='FHD 미만')
                    self._stage('FHD 카메라 입력 대기')
                    self.stop_event.wait(.05)
                    continue
                # Driver buffers can return bursts after a blocking read. Limit
                # actual inference starts as well as camera-loop start times.
                if previous_processing_at is not None:
                    delay = frame_delay(previous_processing_at, time.monotonic(), profile.fps)
                    if delay > 0 and self.stop_event.wait(delay):
                        break
                previous_processing_at = time.monotonic()
                native_size = (frame.shape[1], frame.shape[0])
                # Recognition and display use the same mirrored frame.
                frame = prepare_frame(cv2.flip(frame, 1), profile.size)
                processing_started = time.monotonic()
                result = pipeline.process(frame)
                result.process_ms = (time.monotonic() - processing_started) * 1000  # 5단계 로그 auth_ms
                result.capture_size = native_size
                result.capture_fps = capture.get(cv2.CAP_PROP_FPS)
                captured_at = time.monotonic()
                fps = 1 / max(captured_at - previous_frame_at, .001) if previous_frame_at is not None else 0
                previous_frame_at = captured_at
                if diagnostics is not None:
                    last_diagnostic_status = result.status
                    diagnostics.write('frame', mode=result.power_mode, detected_count=result.detected_count,
                                      requested=result.direction, status=result.status, pose_hint=result.pose_hint,
                                      auth_attempt_id=result.auth_attempt_id, front_threshold=result.front_threshold,
                                      processing_ms=round(result.process_ms, 2), read_ms=read_ms,
                                      capture_size=native_size, processing_size=result.processing_size,
                                      observed_fps=round(fps, 2),
                                      reported_fps=(result.capture_fps
                                                    if result.capture_fps is not None and math.isfinite(result.capture_fps)
                                                    else None),
                                      embedding_errors=result.embedding_errors)
                    if result.authentication_attempted:
                        result.diagnostic_only = True
                        outcome = 'success' if result.authenticated_name is not None else 'failure'
                        finish_diagnostic(outcome, result.status)
                    elif diagnostics.error:
                        finish_diagnostic('error', '진단 로그 저장 오류')
                        self.pause_event.set()
                packet = (frame, result.faces, captured_at, fps, result.status,
                          result.ready, result.saved_count, result)
                try:
                    self.frames.put_nowait(packet)
                except Full:
                    try:
                        self.frames.get_nowait()
                    except Empty:
                        pass
                    self.frames.put_nowait(packet)
                if result.authentication_attempted or result.enrollment_completed or result.enrollment_error:
                    # Exactly one terminal packet; never overwrite it with a later frame.
                    self.pause_event.set()
                self.stop_event.wait(frame_delay(started, time.monotonic(), profile.fps))
        except Exception as exc:
            self.events.put(("error", str(exc)))
            try:
                finish_diagnostic('error', str(exc))
                if pipeline is not None:
                    pipeline.finish_auth_attempt('error', str(exc))
            except (OSError, sqlite3.Error) as record_error:
                self.events.put(('error', f'인증 시도 기록 오류: {record_error}'))
        finally:
            try:
                finish_diagnostic('incomplete', '카메라 중지')
                with ExitStack() as cleanup:
                    if pipeline is not None:
                        cleanup.callback(pipeline.close)
                    if capture is not None:
                        cleanup.callback(capture.release)
            except Exception as exc:
                self.events.put(("error", f"자원 정리 오류: {exc}"))
            finally:
                self.events.put(("stopped", ""))


class FaceApp:
    def __init__(self, root, camera=0, store=None, use_gpio=True, enable_voice=None):
        self.root = root
        self.store = store if store is not None else initialize_database()
        self.worker = None
        self.frames, self.events = Queue(maxsize=1), Queue()
        self.stages = Queue(maxsize=1)
        self.enrollment = None
        self.manual_p_down = False
        self.auth_diagnostics = None
        self.diagnostic_active = False
        self.diagnostic_count = 0
        self.diagnostic_caption = tk.StringVar(value='인증 10회 진단 시작')
        self.diagnostic_status = tk.StringVar(value='')
        self.session_id = 0
        self.closing = False
        self.last_frame_at = 0
        self.gate_ready = False
        self.last_history_refresh = 0
        self.crop_photo = None
        self.applied_save_interval = 1.0
        self.save_interval = tk.StringVar(value="1.0")
        self.save_policy_text = tk.StringVar(value="두 방향이 모두 통과하면 사진 2장을 함께 저장")
        self.save_activity = tk.StringVar(value="DB 저장 대기")
        self.last_saved_at = 0
        self.last_matched = False
        self.last_challenge_prompt = None
        self.last_detected_count = 0
        self.last_embedding_errors = 0
        self.has_error = False
        self.door = Stage5Cycle(use_gpio=use_gpio, enable_voice=enable_voice)  # 5단계 결과 처리
        self.mode = tk.StringVar(value="continuous")
        self.door_message = tk.StringVar(value="문이 닫혀 있습니다. 인증 대기 중입니다.")
        self.manual_text = tk.StringVar(value="수동 개방 버튼 (3회 실패 시 활성)")
        self.standby_text = f"대기 · 카메라 FHD {CAMERA_SIZE[0]}×{CAMERA_SIZE[1]} / 목표 {CAMERA_FPS} FPS"
        self.power_text = tk.StringVar(value=self.standby_text)
        self.saved_text = tk.StringVar(value="누적 Crop 0개")
        self.photo = None
        self.camera = tk.StringVar(value=str(camera))
        self.name = tk.StringVar()
        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.applied_threshold = DEFAULT_THRESHOLD
        self.status = tk.StringVar(value="카메라 시작을 누르면 미리보기가 켜집니다.")
        self.result = tk.StringVar(value="인식 대기 중")
        self.threshold_text = tk.StringVar(value=f"{DEFAULT_THRESHOLD:.2f}")
        self.calibration_text = tk.StringVar()
        self.auth_review = None
        self.manual_feedback = tk.StringVar(value='가이드에 얼굴을 맞추고 P 키를 눌러주세요.')
        preset = load_default_preset()
        self.preset_text = tk.StringVar(value=f'기본 자세 보정: {preset.name}' if preset else '기본 자세 보정: 일반 설정')
        self._build()
        self._refresh_people()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind('<KeyPress>', self._enrollment_key_press, add='+')
        root.bind('<KeyRelease>', self._enrollment_key_release, add='+')
        root.after(33, self._poll)

    def _build(self):
        self.root.title("얼굴 출입")
        self.root.geometry("1060x940")
        self.root.minsize(1020, 900)
        self.root.configure(bg="#f4f6fa")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6fa")
        style.configure("TLabel", background="#f4f6fa", font=("맑은 고딕", 10))
        style.configure("TButton", font=("맑은 고딕", 10), padding=4)
        style.configure("Result.TLabel", font=("맑은 고딕", 13, "bold"))

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 8))
        self.start_button = ttk.Button(toolbar, text="카메라 시작", command=self.start_camera)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(toolbar, text="카메라 중지", command=self.stop_camera, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 0))
        self.options_button = ttk.Button(toolbar, text="옵션", command=self.toggle_options)
        self.options_button.pack(side="right")

        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)
        left, right = ttk.Frame(content), ttk.Frame(content, padding=(20, 0, 0, 0))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="ns")
        self.options_panel = right
        self.options_panel.grid_remove()

        self.indicators = StatusPanel(left)
        self.indicators.pack(fill="x", pady=(0, 6))
        ttk.Label(left, textvariable=self.power_text, foreground="#0369a1", wraplength=640).pack(anchor="w", pady=(0, 6))
        self.canvas = tk.Canvas(left, width=640, height=480, bg="#141d2d", highlightthickness=0)
        self.canvas.pack(anchor="n")
        self._placeholder()
        ttk.Label(left, textvariable=self.door_message, foreground="#166534",
                  font=("맑은 고딕", 16, "bold"), wraplength=640).pack(anchor="w", pady=(6, 0))
        ttk.Label(left, textvariable=self.result, style="Result.TLabel", wraplength=640).pack(anchor="w", pady=(6, 3))
        ttk.Label(left, textvariable=self.status, wraplength=640).pack(anchor="w")

        ttk.Label(right, text="작동 모드", style="Result.TLabel").pack(anchor="w")
        ttk.Label(right, textvariable=self.preset_text, foreground='#7e22ce').pack(anchor='w')
        mode_row = ttk.Frame(right)
        mode_row.pack(fill="x", pady=4)
        ttk.Radiobutton(mode_row, text="연속 작동", variable=self.mode, value="continuous",
                        command=self.apply_mode).pack(side="left")
        ttk.Radiobutton(mode_row, text="1회 작동", variable=self.mode, value="once",
                        command=self.apply_mode).pack(side="right")
        self.mode_buttons = mode_row.winfo_children()
        ttk.Label(right, text="1회: 인증 → 열림 → 닫힘 후 중지").pack(anchor="w", pady=(0, 4))
        ttk.Label(right, text="카메라 연결", style="Result.TLabel").pack(anchor="w")
        row = ttk.Frame(right)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="카메라 번호").pack(side="left")
        self.camera_input = ttk.Spinbox(row, from_=0, to=9, width=5, textvariable=self.camera)
        self.camera_input.pack(side="right")
        self.manual_button = ttk.Button(right, textvariable=self.manual_text, command=self.press_manual, state="disabled")
        self.manual_button.pack(fill="x", pady=(0, 8))

        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True)
        auth_tab, enroll_tab, history_tab = (ttk.Frame(tabs, padding=8) for _ in range(3))
        self.tabs, self.auth_tab = tabs, auth_tab   # 5단계(출입 처리)는 '인식·기록' 탭에서만
        self.enroll_tab, self.history_tab = enroll_tab, history_tab
        tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        tabs.add(auth_tab, text="인식·기록")
        tabs.add(enroll_tab, text="사용자 등록")
        tabs.add(history_tab, text="수집 기록")
        ttk.Label(auth_tab, text="정면 본인 확인 후 화살표가 가리키는\n무작위 두 방향을 바라보세요.\n두 사진의 임베딩·Mesh가 등록 정보와\n모두 일치해야 출입·저장합니다.").pack(anchor="w", pady=8)
        ttk.Label(auth_tab, textvariable=self.saved_text, style="Result.TLabel").pack(anchor="w", pady=8)
        ttk.Label(auth_tab, textvariable=self.save_policy_text, wraplength=240).pack(anchor="w", pady=4)
        ttk.Label(auth_tab, textvariable=self.save_activity, wraplength=240).pack(anchor="w", pady=(0, 8))
        self.diagnostic_button = ttk.Button(auth_tab, textvariable=self.diagnostic_caption,
                                             command=self.toggle_diagnostic_attempt)
        self.diagnostic_button.pack(fill='x')
        ttk.Label(auth_tab, textvariable=self.diagnostic_status, wraplength=250).pack(anchor='w', pady=(2, 0))
        ttk.Label(auth_tab, text="LED: 흰색 대기 / 노랑 조건 확인\n파랑 저장 / 초록 인증 / 빨강 미등록\n보라 사용자 등록 / 회색 중지\n빨강 깜빡임 수동 개방 버튼 대기", wraplength=240).pack(anchor="w", pady=8)
        ttk.Label(enroll_tab, text="이름을 지정한 경우에만 등록합니다.").pack(anchor="w")
        self.name_input = ttk.Entry(enroll_tab, textvariable=self.name, width=25, font=("맑은 고딕", 11))
        self.name_input.pack(fill="x", pady=5)
        self.register_button = ttk.Button(enroll_tab, text="이름 입력 후 얼굴 등록", command=self.register, state="disabled")
        self.register_button.pack(fill="x")
        capture_row = ttk.Frame(enroll_tab)
        capture_row.pack(fill='x', pady=4)
        self.manual_capture_button = ttk.Button(capture_row, text='수동 촬영 (P)', command=self.request_enrollment_photo,
                                                state='disabled')
        self.manual_capture_button.pack(side='left', fill='x', expand=True)
        self.cancel_button = ttk.Button(capture_row, text="등록 취소", command=self.cancel_enrollment, state="disabled")
        self.cancel_button.pack(side='left', fill='x', expand=True)
        self.retry_button = ttk.Button(enroll_tab, text='처음부터 다시 촬영',
                                       command=self.restart_enrollment, state='disabled')
        self.retry_button.pack(fill='x', pady=(0, 4))
        self.progress = ttk.Progressbar(enroll_tab, maximum=REGISTRATION_SAMPLES)
        self.progress.pack(fill="x", pady=(0, 6))
        ttk.Label(enroll_tab, textvariable=self.manual_feedback, wraplength=260).pack(anchor='w', pady=(0, 4))
        ttk.Label(enroll_tab, text="정면·위·좌·우 각각 3장, 총 12장\n안내 자세에서 얼굴을 중앙 가이드에 맞추고\nP 키나 촬영 버튼을 한 장씩 눌러주세요.").pack(anchor="w")

        ttk.Label(enroll_tab, text="등록된 얼굴", style="Result.TLabel").pack(anchor="w", pady=(8, 4))
        self.people = tk.Listbox(enroll_tab, height=3, font=("맑은 고딕", 11), exportselection=False)
        self.people.pack(fill="x")
        self.delete_button = ttk.Button(enroll_tab, text="선택한 등록 삭제", command=self.delete_person)
        self.delete_button.pack(fill="x", pady=6)
        self.grid_button = ttk.Button(enroll_tab, text='등록한 grid / 통합 3D 모델 보기', command=self.show_face_grid)
        self.grid_button.pack(fill='x', pady=3)

        ttk.Label(auth_tab, text="기본 유사도 기준 · 높을수록 엄격").pack(anchor="w", pady=(6, 0))
        self.threshold_slider = ttk.Scale(auth_tab, from_=0.30, to=0.80, variable=self.threshold,
                                          command=self.apply_threshold)
        self.threshold_slider.pack(fill="x")
        ttk.Label(auth_tab, textvariable=self.threshold_text).pack(anchor="e")
        ttk.Button(auth_tab, text='인증 시도 확인 · 유사도 교차점', command=self.show_auth_review).pack(fill='x')
        ttk.Label(auth_tab, textvariable=self.calibration_text, wraplength=250).pack(anchor='w')
        ttk.Label(history_tab, text="최근 Crop · 선택하면 미리보기").pack(anchor="w")
        self.history = ttk.Treeview(history_tab, columns=("id", "name", "embedding"), show="headings", height=6)
        self.history.heading("id", text="기록 ID")
        self.history.heading("embedding", text="임베딩")
        self.history.heading("name", text="인증 사용자")
        self.history.column("id", width=55, stretch=False)
        self.history.column("name", width=95)
        self.history.column("embedding", width=70)
        self.history.pack(fill="x", pady=8)
        self.history.bind("<<TreeviewSelect>>", self._show_crop)
        self.crop_preview = ttk.Label(history_tab, text="기록을 선택하세요.")
        self.crop_preview.pack(pady=8)

    def toggle_options(self):
        if self.options_panel.winfo_manager():
            self.options_panel.grid_remove()
            self.options_button.configure(text="옵션")
            if self.enrollment is None and self.tabs.select() != str(self.auth_tab):
                self.tabs.select(self.auth_tab)
        else:
            self.options_panel.grid()
            self.options_button.configure(text="옵션 닫기")

    def apply_save_interval(self):
        try:
            interval = float(self.save_interval.get())
            if not math.isfinite(interval) or not .2 <= interval <= 60:
                raise ValueError
        except ValueError:
            messagebox.showerror("저장 간격", "0.2~60초 사이의 숫자를 입력하세요.")
            return
        self.applied_save_interval = interval
        self.save_policy_text.set(f"인증 성공 얼굴만 {interval:g}초마다 저장")
        if self.worker and not self.worker.stop_event.is_set():
            self.worker.commands.put(("save_interval", interval))
        self.status.set(f"DB 저장 간격을 {interval:g}초로 적용했습니다.")

    def apply_threshold(self, value):
        if self.enrollment is not None:
            self.threshold.set(self.applied_threshold)
            return
        threshold = float(value)
        self.applied_threshold = threshold
        self.threshold_text.set(f"{threshold:.2f}")
        if self.worker and not self.worker.stop_event.is_set():
            self.worker.commands.put(("threshold", threshold))
            if self.door.phase == "waiting":
                self._set_purpose(self._purpose())

    def apply_mode(self):
        if self.enrollment is not None:
            self.mode.set(self.door.mode)
            return
        self.door.mode = self.mode.get()
        self.status.set("연속 모드: 문이 닫히면 1단계로 돌아갑니다." if self.door.mode == "continuous"
                        else "1회 모드: 인증 후 문이 닫히면 카메라를 중지합니다.")

    def _refresh_history(self):
        if not hasattr(self.store, "recent_observations"):
            return
        rows = self.store.recent_observations(8)
        selected = self.history.selection()
        self.history.delete(*self.history.get_children())
        for row in rows:
            self.history.insert("", "end", iid=str(row["id"]), values=(
                row["id"], row["matched_name"] or "", "128D" if row["has_embedding"] else "없음"))
        if selected and self.history.exists(selected[0]):
            self.history.selection_set(selected[0])
        self.saved_text.set(f"누적 Crop {self.store.count_observations():,}개")

    def _show_crop(self, event=None):
        selected = self.history.selection()
        if not selected:
            return
        try:
            with Image.open(BytesIO(self.store.observation_crop(int(selected[0])))) as image:
                image.thumbnail((210, 160))
                self.crop_photo = ImageTk.PhotoImage(image.copy())
            self.crop_preview.configure(image=self.crop_photo, text="")
        except (KeyError, OSError) as exc:
            self.status.set(f"Crop 미리보기 오류: {exc}")

    def _placeholder(self, text="카메라 미리보기\n오른쪽의 ‘카메라 시작’을 눌러주세요."):
        self.canvas.delete("all")
        self.canvas.create_text(320, 240, text=text, fill="#cbd5e1", font=("맑은 고딕", 15), justify="center", width=580)
        self.photo = None

    def _refresh_people(self):
        self.people.delete(0, "end")
        for name in sorted(self.store.people):
            self.people.insert("end", name)
        self._refresh_calibration()

    def _refresh_calibration(self):
        self.calibration_text.set(policy_text(self.store.calibration.status()))

    def show_auth_review(self):
        if self.auth_review is None or not self.auth_review.window.winfo_exists():
            self.auth_review = AuthenticationReview(self.root, self.store, self._refresh_calibration)
        else:
            self.auth_review.refresh()
            self.auth_review.window.lift()

    def show_face_grid(self):
        if self.enrollment is not None and not self.enrollment.completed:
            return
        selected = self.people.curselection()
        name = self.enrollment.name if self.enrollment is not None else self.people.get(selected[0]) if selected else None
        if name is None:
            messagebox.showinfo('얼굴 grid', '등록한 이름을 선택하세요.')
            return
        grids = self.store.face_grids(name)
        if not self.store.has_grid_profile(name) or not self.store.has_fused_model(name):
            messagebox.showinfo('얼굴 grid', '4방향 grid 또는 통합 3D 모델이 없습니다. 얼굴을 다시 등록하세요.')
            return
        preview = render_grids(grids)
        model = self.store.face_model(name)
        fused = render_fused_model(model)
        window = tk.Toplevel(self.root)
        window.title(f'{name} · 4방향 grid 및 통합 3D 얼굴 모델')
        ttk.Label(window, text=f'{name} · 4방향 표면과 통합 모델', font=('맑은 고딕', 13, 'bold')).pack(padx=12, pady=10)
        tabs = ttk.Notebook(window)
        tabs.pack(padx=12, fill='both', expand=True)
        directional_tab, fused_tab = ttk.Frame(tabs), ttk.Frame(tabs)
        tabs.add(directional_tab, text='4방향 grid')
        tabs.add(fused_tab, text='통합 3D 모델')
        window.grid_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        window.fused_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)))
        ttk.Label(directional_tab, text='위 / 좌·정면·우').pack(pady=6)
        ttk.Label(directional_tab, image=window.grid_photo).pack()
        ttk.Label(fused_tab, image=window.fused_photo).pack(pady=6)
        ttk.Label(fused_tab, text=f'468개 정점 / {len(grid_triangles())}개 삼각형 · 같은 통합 모델을 3각도로 표시').pack(pady=5)
        ttk.Label(window, text='RGB 영상에서 추정한 상대 형상입니다. 인증은 방향별 grid와 통합 모델을 모두 비교합니다.').pack(padx=12, pady=10)

    def start_camera(self):
        if self.worker is not None:
            return
        if self.door.processing:
            self.status.set("진행 중인 결과 처리가 끝난 후 다시 시작하세요.")
            return
        try:
            index = int(self.camera.get())
            if not 0 <= index <= 9:
                raise ValueError
        except ValueError:
            messagebox.showerror("카메라 번호", "0~9 사이의 번호를 입력하세요.")
            return
        self.frames, self.events = Queue(maxsize=1), Queue()
        self.stages = Queue(maxsize=1)
        self.last_frame_at = 0
        self.gate_ready = False
        self.has_error = False
        self.last_detected_count = 0
        self.last_embedding_errors = 0
        self.last_saved_at = 0
        if self.enrollment is None:
            self.door.reset()
        self.last_challenge_prompt = None
        self.door.mode = self.mode.get()
        self.door_message.set("문이 닫혀 있습니다. 인증 대기 중입니다.")
        self.power_text.set(self.standby_text + " · 카메라 준비 중")
        self.start_button.configure(state="disabled")
        self.camera_input.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("모델 준비 중… 첫 실행에는 약 37 MB 다운로드가 필요합니다.")
        self.indicators.phase.set("현재 단계: 모델 준비 / 카메라 연결")
        self.indicators.set_led("white", "카메라 준비 중", blink=True)
        self.worker = CameraWorker(index, self.frames, self.events, self.stages,
                                   self.applied_save_interval, self.threshold.get(),
                                   self._purpose(), self.session_id,
                                   self.enrollment.name if self.enrollment else None)
        if self.enrollment is not None:
            self.enrollment.error = None
            self._show_enrollment_state()
        self.worker.start()

    def stop_camera(self):
        if self.worker is not None:
            if self.door.phase != "waiting":
                self.status.set("진행 중인 결과 처리를 마친 후 카메라를 중지합니다.")
                self.worker.stop_event.set()
                return
            self.worker.stop_event.set()
            self.stop_button.configure(state="disabled")
            self.register_button.configure(state="disabled")
            self.cancel_enrollment()
            self.status.set("카메라를 종료하고 있습니다…")
            self.result.set("인식 중지")
            self._placeholder("카메라 중지 중…")
            self.indicators.idle("카메라 중지 중")

    def register(self):
        if self.enrollment is not None or self.door.phase != 'waiting':
            return
        if self.worker is None or self.worker.stop_event.is_set() or time.monotonic() - self.last_frame_at > 2:
            messagebox.showinfo("얼굴 등록", "먼저 카메라 영상을 확인해주세요.")
            return
        name = self.name.get().strip()
        if not name or len(name) > 40:
            messagebox.showinfo("얼굴 등록", "이름을 1~40자로 입력하세요.")
            return
        if name in self.store.people and not messagebox.askyesno("다시 등록", f"‘{name}’의 기존 임베딩을 새로 등록할까요?"):
            return
        if not self.door.suspend():
            self.status.set('출입 처리가 끝난 뒤 등록을 시작하세요.')
            return
        self.enrollment = SimpleNamespace(name=name, samples=[], error=None, completed=False, direction=None)
        self.manual_p_down = False
        self.manual_feedback.set('가이드에 얼굴을 맞추고 P 키를 눌러주세요.')
        self._lock_enrollment_ui(True)
        self._set_purpose("enroll", begin_enrollment=True)
        self.progress["value"] = 0
        self.register_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.retry_button.configure(state='disabled')
        self.name_input.configure(state="disabled")
        self.delete_button.configure(state="disabled")
        self.status.set("얼굴등록단계 · 정면 촬영 안내를 준비 중입니다.")
        self._show_enrollment_state()

    def _lock_enrollment_ui(self, locked):
        for tab in (self.auth_tab, self.history_tab):
            self.tabs.tab(tab, state='disabled' if locked else 'normal')
        for button in self.mode_buttons:
            button.configure(state='disabled' if locked else 'normal')
        self.threshold_slider.configure(state='disabled' if locked else 'normal')
        self.grid_button.configure(state='disabled' if locked else 'normal')
        self.manual_capture_button.configure(state='disabled')
        if locked:
            self.tabs.select(self.enroll_tab)
        self.indicators.set_enrollment(locked)

    def cancel_enrollment(self):
        was_enrolling = self.enrollment is not None
        self.enrollment = None
        self.manual_p_down = False
        self._lock_enrollment_ui(False)
        if was_enrolling:
            self.door.resume()
            self.indicators.idle('얼굴등록 종료 · 미리보기 준비')
            self._set_purpose(self._purpose(), cancel_enrollment=True)
        self.progress["value"] = 0
        self.cancel_button.configure(state="disabled", text='등록 취소')
        self.retry_button.configure(state='disabled')
        self.manual_capture_button.configure(state='disabled')
        self.manual_feedback.set('가이드에 얼굴을 맞추고 P 키를 눌러주세요.')
        self.name_input.configure(state="normal")
        self.delete_button.configure(state="normal")
        if self.worker and not self.worker.stop_event.is_set() and time.monotonic()-self.last_frame_at <= 2:
            self.register_button.configure(state="normal")

    def restart_enrollment(self):
        if self.enrollment is None or not self.enrollment.error or self.enrollment.completed:
            return
        self.enrollment.error = None
        self.enrollment.samples = []
        self.enrollment.direction = None
        self.manual_p_down = False
        self.manual_capture_button.configure(state='disabled')
        self.manual_feedback.set('재촬영 준비 · 정면 안내 후 P 키 사용 가능')
        self.session_id += 1
        self.last_challenge_prompt = None
        self.has_error = False
        self.retry_button.configure(state='disabled')
        self.progress['value'] = 0
        if self.worker is None:
            self.start_camera()
        elif not self.worker.stop_event.is_set():
            self.worker.requested_session = self.session_id
            self.worker.commands.put(('retry_enrollment', self.session_id))
        self.status.set('얼굴등록단계 · FHD 유지 · 정면부터 다시 촬영합니다.')
        self._show_enrollment_state()

    def _show_enrollment_state(self):
        self.indicators.set_enrollment(True)
        self.power_text.set('얼굴등록단계 · FHD 1920×1080 고정 · 인증·출입 처리 중지')
        self.door_message.set('얼굴등록 중 · 출입 처리는 실행하지 않습니다.')

    @staticmethod
    def _is_p_key(event):
        return (getattr(event, 'keysym', '').lower() == 'p'
                or getattr(event, 'char', '') in ('ㅔ', 'ㅖ')
                or sys.platform == 'win32' and getattr(event, 'keycode', None) == 80)

    def _enrollment_key_press(self, event):
        if self.enrollment is None or not self._is_p_key(event):
            return
        if not self.manual_p_down:
            self.manual_p_down = True
            self.request_enrollment_photo()
        return 'break'

    def _enrollment_key_release(self, event):
        if self._is_p_key(event):
            self.manual_p_down = False

    def request_enrollment_photo(self):
        ticket = self.enrollment
        if (ticket is None or ticket.error or ticket.completed or self.worker is None
                or self.worker.stop_event.is_set()):
            return False
        if not ticket.direction:
            self.manual_feedback.set('FHD 영상과 방향 안내가 나오면 P 키를 눌러주세요.')
            return False
        now = time.monotonic()
        if now-self.last_frame_at > .5:
            self.manual_feedback.set('새 카메라 영상이 필요합니다. 영상이 갱신된 후 다시 눌러주세요.')
            return False
        self.worker.commands.put(('manual_capture', (self.session_id, ticket.direction, len(ticket.samples), now)))
        self.manual_feedback.set('촬영 요청 · 다음 영상을 확인합니다.')
        return True

    def _purpose(self):
        if self.enrollment is not None:
            return "enroll"
        if self.auth_diagnostics is not None and not self.diagnostic_active:
            return "preview"
        return "authenticate" if self.tabs.select() == str(self.auth_tab) else "preview"

    def toggle_diagnostic_attempt(self):
        if self.diagnostic_active:
            if self.worker and not self.worker.stop_event.is_set():
                self.worker.commands.put(('diagnostic_abort', None))
                self.diagnostic_button.configure(state='disabled')
            return
        if self.worker is None or self.worker.stop_event.is_set() or self.door.phase != 'waiting':
            self.diagnostic_status.set('카메라를 시작하고 인증 대기 상태에서 눌러주세요.')
            return
        if self.diagnostic_count >= ATTEMPT_LIMIT:
            self.auth_diagnostics = None
            self.diagnostic_count = 0
            self.diagnostic_caption.set('인증 10회 진단 시작')
            self._set_purpose('authenticate')
            return
        if self.auth_diagnostics is None:
            try:
                self.auth_diagnostics = AuthDiagnostics()
            except OSError as exc:
                self.diagnostic_status.set(f'진단 로그를 열 수 없습니다: {exc}')
                return
        self.diagnostic_active = True
        self.diagnostic_caption.set('이번 시도 중단으로 기록')
        self.diagnostic_status.set(f'{self.diagnostic_count+1}/{ATTEMPT_LIMIT} 시도 중 · 출입 동작 없이 인증 검사')
        self._set_purpose('authenticate', diagnostics=self.auth_diagnostics)

    def _set_purpose(self, purpose, cancel_enrollment=False, begin_enrollment=False, diagnostics=None):
        if self.enrollment is not None and not cancel_enrollment:
            if purpose != 'enroll' or not begin_enrollment:
                return False
        self.session_id += 1
        self.gate_ready = False
        self.last_challenge_prompt = None
        if self.worker and not self.worker.stop_event.is_set():
            self.worker.requested_session = self.session_id
            value = (purpose, self.session_id)
            if purpose == 'enroll':
                value += (self.enrollment.name,)
            if diagnostics is not None:
                self.worker.commands.put(('diagnostic_start', (*value, diagnostics)))
            else:
                self.worker.commands.put(("cancel_enrollment" if cancel_enrollment else "purpose", value))
        return True

    def _tab_changed(self, event=None):
        if self.enrollment is not None:
            if self.tabs.select() != str(self.enroll_tab):
                self.tabs.select(self.enroll_tab)
            return
        if self.door.phase == "waiting":
            self._set_purpose(self._purpose())

    def delete_person(self):
        if self.enrollment is not None:
            return
        selected = self.people.curselection()
        if not selected:
            return
        name = self.people.get(selected[0])
        if messagebox.askyesno("등록 삭제", f"‘{name}’의 저장된 임베딩을 삭제할까요?"):
            try:
                delete_user(self.store, name)
            except OSError as exc:
                messagebox.showerror("삭제 실패", str(exc))
                return
            self._refresh_people()
            self.status.set(f"‘{name}’의 등록을 삭제했습니다.")

    def _handle_frame(self, frame, faces, captured_at, fps, gate_status=None, gate_ready=True,
                      saved_count=None, telemetry=None):
        if telemetry is not None and telemetry.session_id != self.session_id:
            return
        if self.enrollment is not None:
            if telemetry is not None and telemetry.purpose == 'enroll':
                self._handle_enrollment_frame(frame, captured_at, fps, telemetry)
            return
        if telemetry is not None and telemetry.purpose == 'enroll':
            return
        self.last_frame_at = captured_at
        self.gate_ready = gate_ready and bool(faces)
        self.last_matched = False
        self.last_detected_count = telemetry.detected_count if telemetry else len(faces)
        self.last_embedding_errors = telemetry.embedding_errors if telemetry else 0
        if telemetry is not None:
            if telemetry.liveness_step and telemetry.liveness_step != self.last_challenge_prompt:
                self.door.handler.voice.say_text(telemetry.liveness_step)
                self.last_challenge_prompt = telemetry.liveness_step
            if telemetry.saved_this_frame:
                self.last_saved_at = time.monotonic()
            self.save_activity.set("인증 성공 시에만 저장" if telemetry.authenticated_name is None else
                                   (f"두 방향 통과 · {telemetry.saved_this_frame}장 함께 저장" if telemetry.saved_this_frame
                                    else "두 방향 비교 완료 대기"))
            self.indicators.set_stage(1, "얼굴 대기" if not telemetry.detected_count else
                                      ("조건 통과" if self.gate_ready else "조건 확인 중"),
                                      "green" if self.gate_ready else "yellow")
            self.indicators.set_stage(2, f"얼굴 {telemetry.detected_count}개 검출", "blue" if telemetry.detected_count else "off")
            self.indicators.set_stage(3, f"{len(faces)}개 생성 / {telemetry.embedding_errors}개 실패",
                                      "red" if telemetry.embedding_errors else ("green" if faces else "off"))
            if telemetry.power_mode == "standby":
                self.indicators.set_stage(3, "절전 · 임베딩 중지")
                self.indicators.set_stage(1, "절전 · 조건 확인", "yellow")
            mode_text = ("얼굴등록단계 · FHD 고정" if telemetry.power_mode == 'enrollment' else
                         "1단계 절전" if telemetry.power_mode == "standby" else "2~4단계 활성")
            width, height = telemetry.processing_size
            details = f"{mode_text}: {width}×{height} / 목표 {telemetry.target_fps} FPS · 실제 처리 {fps:.1f} FPS"
            if telemetry.capture_size is not None:
                native_w, native_h = telemetry.capture_size
                details += f" | 카메라 {native_w}×{native_h} / 보고 {telemetry.capture_fps or 0:g} FPS"
            self.power_text.set(details)
            self.indicators.set_stage(4, self.save_activity.get(), "blue" if telemetry.saved_this_frame else "off")
        self.register_button.configure(state="normal")
        if gate_status:
            self.status.set(gate_status)

        self.canvas.delete("all")
        height, width = frame.shape[:2]
        scale = min(640 / width, 480 / height)
        preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        if telemetry is not None and telemetry.mesh_landmarks is not None:
            draw_mesh_preview(preview, telemetry.mesh_landmarks)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        offset_x, offset_y = (640 - preview.shape[1]) // 2, (480 - preview.shape[0]) // 2
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.photo)
        labels = []
        display_faces = faces
        if not faces and telemetry is not None:
            display_faces = [SimpleNamespace(box=box) for box in (telemetry.detection_boxes or [])]
        for face in display_faces:
            if telemetry is not None:
                name, score = telemetry.authenticated_name, telemetry.match_score
            else:
                name, score = (authenticate_face(self.store, face.embedding, self.threshold.get())
                               if self.gate_ready else (None, None))
            self.last_matched = self.last_matched or name is not None
            color = "#4ade80" if name else "#fbbf24"
            x, y, w, h = face.box
            x, y, w, h = x * scale + offset_x, y * scale + offset_y, w * scale, h * scale
            self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=3)
            if not name:
                continue
            label = f"{name} · {score:.3f}" if score is not None else name
            labels.append(label)
            text = self.canvas.create_text(max(4, x), max(4, y - 26), text=label, anchor="nw",
                                           fill=color, font=("맑은 고딕", 11, "bold"))
            background = self.canvas.create_rectangle(self.canvas.bbox(text), fill="#141d2d", outline="")
            self.canvas.tag_lower(background, text)
        self.result.set(" / ".join(labels))
        self._draw_guidance(telemetry)
        if telemetry is not None and telemetry.purpose == 'authenticate':
            self._draw_face_guide(preview, offset_x, offset_y, 'authentication_face_guide')
            if telemetry.pose_hint and not telemetry.authentication_attempted:
                self._draw_pose_hint(telemetry.pose_hint, preview, offset_x, offset_y)
        front_cut = telemetry.front_threshold if telemetry is not None and telemetry.front_threshold is not None else self.threshold.get()
        self.canvas.create_text(10, 468, anchor="sw", text=f"{fps:.0f} FPS · 정면 기준 {front_cut:.3f}",
                                fill="white", font=("맑은 고딕", 10))
        if saved_count is not None:
            self.saved_text.set(f"누적 Crop {saved_count:,}개")
        if self.gate_ready:
            if self.last_matched:
                detail = "인증 성공 / 저장 완료" if telemetry and telemetry.saved_this_frame else "인증 성공"
                self.indicators.set_stage(4, detail, "green")
            elif telemetry is not None and telemetry.authentication_attempted:
                self.indicators.set_stage(4, "인증 실패", "red")
        # 4단계 인증이 실제로 수행된 프레임(조건 통과 + 얼굴 1개)이면 결과를 5단계로 넘김(성공·실패 모두)
        attempted = telemetry is not None and telemetry.authentication_attempted
        on_auth_tab = self.tabs.select() == str(self.auth_tab)
        if attempted and telemetry.diagnostic_only:
            self.door_message.set('진단 인증 성공' if telemetry.authenticated_name else '진단 인증 실패')
            self.status.set(telemetry.status)
        elif attempted and self.enrollment is None and self.worker is not None and not on_auth_tab:
            self.door_message.set("‘사용자 등록’·‘수집 기록’ 탭에서는 출입 처리(5단계)를 하지 않습니다.")
        elif attempted and self.enrollment is None and self.worker is not None and not self.store.people:
            self.door_message.set("등록된 사용자가 없습니다. ‘사용자 등록’ 탭에서 먼저 등록하세요.")
            self._set_purpose(self._purpose())
        elif attempted and self.enrollment is None and self.worker is not None:
            name, score = telemetry.authenticated_name, telemetry.match_score
            if name is not None and not telemetry.liveness_passed:
                raise RuntimeError("동작 검사를 통과하지 않은 성공 결과입니다.")
            if self.door.start(time.monotonic(), name, score, frame=frame,
                               auth_ms=getattr(telemetry, "process_ms", None), message=telemetry.status):
                self.worker.pause_event.set()
                if name:   # 성공 문구는 임시통합 그대로
                    self.door_message.set("문이 열렸습니다.")
                    self.status.set("3초 후 문이 닫힙니다. 열림 중에는 인증·DB 저장을 일시 정지합니다.")
                    self.indicators.phase.set("현재 단계: 인증 완료 → 문 열림")
                else:      # 실패 처리는 5단계에서 추가
                    self.door_message.set("인증 실패")
                    self.status.set(telemetry.status)
                    self.indicators.phase.set("현재 단계: 인증 실패 → 5단계 결과 처리")
                self.indicators.set_stage(5, "처리 중", "yellow")
                self.register_button.configure(state="disabled")
        self._update_led()

    def _draw_guidance(self, telemetry):
        if telemetry is None or not telemetry.direction:
            return
        target = telemetry.direction
        vectors = {'up': (320, 105, 320, 45),
                   'left': (350, 70, 290, 70), 'right': (290, 70, 350, 70)}
        self.canvas.create_rectangle(260, 30, 380, 140, fill='#141d2d', outline='')
        if target in vectors:
            self.canvas.create_line(*vectors[target], fill='#fbbf24', width=6,
                                    arrow=tk.LAST, arrowshape=(16, 20, 10))
        else:
            self.canvas.create_oval(305, 55, 335, 85, outline='#fbbf24', width=4)
        self.canvas.create_text(320, 120, text=LABELS[target], fill='white', font=('맑은 고딕', 13, 'bold'))
        self.result.set(telemetry.status)

    def _draw_face_guide(self, preview, ox, oy, tag):
        height, width = preview.shape[:2]
        rx, ry = FACE_GUIDE_RADII
        cx, cy = ox+width/2, oy+height/2
        self.canvas.create_oval(cx-rx*width, cy-ry*height, cx+rx*width, cy+ry*height,
                                outline='#54e1d5', width=2, tags=tag)

    def _draw_pose_hint(self, hint, preview, ox, oy):
        text = POSE_HINT_TEXT.get(hint)
        if text is None:
            return
        height, width = preview.shape[:2]
        item = self.canvas.create_text(ox+width/2, oy+height-30, text=text, width=width-40,
                                       justify='center', fill='#fef3c7', font=('맑은 고딕', 15, 'bold'),
                                       tags='authentication_pose_hint')
        x1, y1, x2, y2 = self.canvas.bbox(item)
        background = self.canvas.create_rectangle(x1-13, y1-7, x2+13, y2+7, fill='#141d2d', outline='#fbbf24')
        self.canvas.tag_lower(background, item)

    def _handle_enrollment_frame(self, frame, captured_at, fps, telemetry):
        self.last_frame_at = captured_at
        self.gate_ready = False
        self.enrollment.samples = [None] * telemetry.enrollment_count
        self.enrollment.direction = telemetry.direction
        if telemetry.manual_capture_status:
            self.manual_feedback.set(telemetry.manual_capture_status)
        self.progress['value'] = telemetry.enrollment_count
        self._show_enrollment_state()
        self.status.set(telemetry.status)
        if telemetry.liveness_step and telemetry.liveness_step != self.last_challenge_prompt:
            self.door.handler.voice.say_text(telemetry.liveness_step)
            self.last_challenge_prompt = telemetry.liveness_step
        if telemetry.enrollment_error:
            self.enrollment.error = telemetry.enrollment_error
            self.status.set(f'{telemetry.enrollment_error} · 등록 단계에서 다시 촬영하거나 취소하세요.')
            self.retry_button.configure(state='normal')
        elif telemetry.enrollment_completed:
            self.enrollment.completed = True
            self.store.refresh()
            self._refresh_people()
            self.retry_button.configure(state='disabled')
            self.cancel_button.configure(text='등록 완료 · 나가기')
            self.grid_button.configure(state='normal')
            self.status.set(f'‘{telemetry.enrollment_completed}’ 등록 완료 · 4방향 grid·통합 3D 모델·임베딩 저장')
        self.manual_capture_button.configure(state='normal' if telemetry.direction and not self.enrollment.error
                                             and not self.enrollment.completed else 'disabled')
        # Registration never renders authentication names, scores or stage 1-5 results.
        self.canvas.delete('all')
        height, width = frame.shape[:2]
        scale = min(640 / width, 480 / height)
        preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        if telemetry.mesh_landmarks is not None:
            draw_mesh_preview(preview, telemetry.mesh_landmarks)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        ox, oy = (640-preview.shape[1])//2, (480-preview.shape[0])//2
        self.canvas.create_image(ox, oy, anchor='nw', image=self.photo)
        for x, y, w, h in telemetry.detection_boxes or []:
            self.canvas.create_rectangle(x*scale+ox, y*scale+oy, (x+w)*scale+ox,
                                         (y+h)*scale+oy, outline='#c084fc', width=3)
        self.result.set('등록 완료' if self.enrollment.completed else
                        '등록 촬영 중단 · 다시 촬영 대기' if self.enrollment.error else telemetry.status)
        if not self.enrollment.error and not self.enrollment.completed:
            self._draw_guidance(telemetry)
            self._draw_face_guide(preview, ox, oy, 'enrollment_face_guide')
        self.canvas.create_text(10, 468, anchor='sw',
            text=f'얼굴등록 전용 · FHD 1920×1080 · {telemetry.enrollment_count}/{REGISTRATION_SAMPLES}장 · {fps:.0f} FPS',
            fill='white', font=('맑은 고딕', 10))
        self._update_led()

    def press_manual(self):
        if self.enrollment is not None:
            return
        if self.door.press_manual():
            self.status.set("수동 개방 버튼을 눌렀습니다.")

    def _advance_door(self):
        if self.enrollment is not None:
            self.manual_button.configure(state='disabled')
            return
        try:
            event = self.door.advance(time.monotonic())
        except RuntimeError as exc:
            self.has_error = True
            self.status.set(str(exc))
            event = "reset"
        if self.door.phase == "processing":
            _, blink, text = self.door.led_state()
            self.door_message.set(text)
            self.manual_button.configure(state="normal" if blink else "disabled")
            self.manual_text.set("수동 개방 버튼 · 지금 누르면 개방" if blink else "수동 개방 버튼 (3회 실패 시 활성)")
            return
        self.manual_button.configure(state="disabled")
        self.manual_text.set("수동 개방 버튼 (3회 실패 시 활성)")
        if event in ("retry", "reset", "finished"):
            action = self.door.last_action
            handler = self.door.handler
            if action is not None:
                messages = {"OPENED": ("문이 닫혔습니다.", "green"),
                            "RETRY": (f"인증 실패 · 다시 시도합니다. ({handler.fail_count}/{handler.settings['max_retries']})", "red"),
                            "MANUAL_OPEN": ("수동 개방 후 문이 닫혔습니다.", "green"),
                            "LOCKED": ("3회 실패 · 잠금 유지", "red")}
                text, color = messages[action.value]
                self.door_message.set(text)
                self.indicators.set_stage(5, f"{action.value} · 연속 실패 {handler.fail_count}/"
                                             f"{handler.settings['max_retries']}", color)
        if event in ("retry", "reset", "finished"):
            while True:
                try:
                    self.frames.get_nowait()
                except Empty:
                    break
            self.gate_ready, self.last_matched = False, False
            self.last_detected_count = self.last_embedding_errors = 0
            self.last_frame_at = self.last_saved_at = 0
            self.power_text.set(self.standby_text + (" · 카메라 중지" if event == "finished" else " · 전환 중"))
            finished = event == "finished"
            stage5_text, stage5_color = self.indicators.details[4].get(), self.indicators.stage5_color
            self.indicators.idle("1단계 · 1회 작동 완료 / 재시작 대기" if finished else "1단계 · 얼굴 대기")
            self.indicators.set_stage(5, stage5_text, stage5_color)   # 마지막 5단계 결과는 남겨 둠
            self.result.set("1회 작동 완료 · 카메라 시작을 눌러 재시작" if finished else "1단계 · 얼굴 대기")
            if self.worker:
                if finished:
                    self.worker.stop_event.set()
                    self.stop_button.configure(state="disabled")
                else:
                    self._set_purpose(self._purpose())
                    self.worker.pause_event.clear()
            self.register_button.configure(state="disabled")

    def _update_led(self):
        if self.enrollment is not None:
            text = ('얼굴등록 완료 · 나가기 대기' if self.enrollment.completed else
                    '얼굴등록 중단 · 다시 촬영 대기' if self.enrollment.error else
                    f'얼굴등록 {len(self.enrollment.samples)}/{REGISTRATION_SAMPLES}')
            self.indicators.set_led('purple', text, blink=not self.enrollment.completed)
        elif self.has_error:
            self.indicators.set_led("red", "오류 · 상태 메시지 확인", blink=True)
        elif self.door.phase == "processing":
            color, blink, text = self.door.led_state()
            self.indicators.set_led(color, text, blink=blink)
        elif self.worker is None or self.worker.stop_event.is_set():
            self.indicators.set_led("off", "카메라 꺼짐")
        elif self.last_frame_at and time.monotonic() - self.last_frame_at > 2:
            self.indicators.set_led("yellow", "카메라 프레임 대기", blink=True)
        elif self.last_embedding_errors:
            self.indicators.set_led("red", "3단계 · 임베딩 생성 오류", blink=True)
        elif time.monotonic() - self.last_saved_at < .2:
            self.indicators.set_led("blue", "4단계 · DB 저장 완료")
        elif self.gate_ready:
            self.indicators.set_led("green" if self.last_matched else "red",
                                    "4단계 · 인증 성공" if self.last_matched else "4단계 · 미등록 얼굴")
        elif self.last_detected_count:
            self.indicators.set_led("yellow", "1단계 · 대기 조건 확인", blink=True)
        else:
            self.indicators.set_led("white", "1단계 · 카메라 준비 / 얼굴 대기", blink=True)

    def _poll(self):
        if self.closing:
            return
        try:
            self._poll_once()
        except (sqlite3.Error, OSError, ValueError, RuntimeError) as exc:
            self.has_error = True
            self.status.set(f"처리 오류: {exc}")
            if self.worker:
                self.worker.stop_event.set()
        finally:
            if not self.closing:
                self.root.after(33, self._poll)

    def _poll_once(self):
        self._advance_door()
        try:
            phase = self.stages.get_nowait()
        except Empty:
            pass
        else:
            if self.worker and not self.worker.stop_event.is_set() and self.door.phase == "waiting" and not self.door.completed:
                self.indicators.phase.set('현재 단계: 얼굴등록 · FHD 고정' if self.enrollment is not None
                                          else f"현재 단계: {phase}")
        if (self.tabs.index(self.tabs.select()) == 2
                and time.monotonic() - self.last_history_refresh >= 1):
            self._refresh_history()
            self.last_history_refresh = time.monotonic()
        while True:
            try:
                kind, value = self.events.get_nowait()
            except Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == 'manual_capture':
                session_id, accepted, message = value
                if self.enrollment is not None and session_id == self.session_id:
                    self.manual_feedback.set(message)
            elif kind == 'diagnostic_finished':
                diagnostic, outcome = value
                if diagnostic is self.auth_diagnostics:
                    self.diagnostic_active = False
                    self.diagnostic_button.configure(state='normal')
                    self.diagnostic_count = diagnostic.completed
                    if diagnostic.error:
                        self.diagnostic_status.set(f'로그 저장 오류: {diagnostic.error}')
                        self.auth_diagnostics = None
                        self.diagnostic_count = 0
                        self.diagnostic_caption.set('인증 10회 진단 시작')
                    else:
                        label = {'success': '성공', 'failure': '실패', 'incomplete': '중단', 'error': '오류'}[outcome]
                        self.diagnostic_status.set(f'{self.diagnostic_count}/{ATTEMPT_LIMIT} 기록 · {label}\n{diagnostic.path.name}')
                        self.diagnostic_caption.set('일반 인증 재개' if self.diagnostic_count >= ATTEMPT_LIMIT
                                                    else '다음 진단 시도 시작')
            elif kind == "error":
                self.has_error = True
                self.status.set(value)
                if self.enrollment is not None:
                    self.enrollment.error = value
                self._placeholder("카메라를 연결하지 못했습니다.\n상태 메시지를 확인한 후 다시 시작하세요.")
                messagebox.showerror("카메라 연결 오류", value)
            elif kind == "stopped":
                self.worker = None
                self.gate_ready = False
                if self.enrollment is not None:
                    self.enrollment.error = self.enrollment.error or '등록용 카메라가 중지되었습니다.'
                    self.start_button.configure(state='normal')
                    self.stop_button.configure(state='disabled')
                    self.retry_button.configure(state='normal')
                    self.manual_capture_button.configure(state='disabled')
                    self._show_enrollment_state()
                    self.result.set('얼굴등록 중단 · 카메라 재시작 또는 취소 대기')
                    continue
                self.cancel_enrollment()
                self.start_button.configure(state="normal")
                self.camera_input.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.register_button.configure(state="disabled")
                self.power_text.set(self.standby_text + " · 카메라 중지")
                if self.door.completed:
                    self.result.set("1회 작동 완료 · 카메라 시작을 눌러 재시작")
                    self.indicators.idle("1단계 · 1회 작동 완료 / 재시작 대기")
                else:
                    self.result.set("인식 대기 중")
                    self._placeholder()
                    self.indicators.idle("카메라 꺼짐")
                if self.has_error:
                    self.indicators.phase.set("현재 단계: 오류로 중지 · 상태 메시지 확인")
        if self.worker and not self.worker.stop_event.is_set():
            try:
                packet = self.frames.get_nowait()
            except Empty:
                pass
            else:
                if self.enrollment is not None or (self.door.phase == "waiting" and not self.door.completed):
                    if packet[-1].session_id == self.session_id:
                        self._handle_frame(*packet)
            if (self.enrollment is None and self.door.phase == "waiting" and self.last_frame_at
                    and time.monotonic() - self.last_frame_at > 2):
                self.register_button.configure(state="disabled")
                self.result.set("카메라 영상을 기다리고 있습니다…")
                self._placeholder("카메라 영상을 기다리고 있습니다…")
        self._update_led()

    def close(self):
        self.closing = True
        self.root.withdraw()
        self.door.cancel()
        if self.worker:
            self.worker.stop_event.set()
            # Keep the process alive until the capture is released.
            self._wait_for_close()
        else:
            self._wait_for_close()

    def _wait_for_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(50, self._wait_for_close)
        else:
            if self.door.processing:
                self.root.after(50, self._wait_for_close)
                return
            self.door.close()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="얼굴인식 출입 1~5단계 통합 화면")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--no-gpio", action="store_true", help="GPIO 장치(LED·부저·버튼)를 쓰지 않음")
    parser.add_argument("--no-voice", action="store_true", help="음성 안내 끔")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        FaceApp(root, args.camera, use_gpio=not args.no_gpio, enable_voice=False if args.no_voice else None)
    except (ValueError, OSError) as exc:
        root.withdraw()
        messagebox.showerror("등록 데이터 오류", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
