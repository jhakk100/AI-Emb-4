"""1~5단계 통합 화면 (임시통합 app.py 기반). run.bat 또는 python app.py로 실행.

임시통합의 1~4단계 화면·등록·기록 기능은 그대로 두고, 인증 결과 처리(문 열림 흉내 door_cycle)를
5단계(5-result.py)로 바꿨다: 성공 시 문 열림, 실패 시 재시도, 3회 실패 시 경보·수동 개방 버튼·잠금,
보안 로그(logs/security_log.csv)와 결과 사진(logs/snapshots/) 저장. 1~4단계 모듈은 stages/에 있다.
"""

import argparse
from io import BytesIO
import math
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

from face_core import DEFAULT_THRESHOLD, Enrollment, FaceEngine, REGISTRATION_SAMPLES, open_camera
from models import ensure_models
from authenticate import authenticate_face
from database import add_user, delete_user, initialize_database
from stage5_cycle import Stage5Cycle
from capture_control import STANDBY_PROFILE, configure_capture, frame_delay, prepare_frame
from pipeline import FacePipeline
from status_panel import StatusPanel


class CameraWorker(threading.Thread):
    def __init__(self, index, frames, events, stages, save_interval=1.0, threshold=DEFAULT_THRESHOLD):
        super().__init__(daemon=True)
        self.index, self.frames, self.events = index, frames, events
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.stages, self.commands = stages, Queue()
        self.save_interval = save_interval
        self.threshold = threshold

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
        capture = None
        try:
            self._stage("모델 준비 / 카메라 연결")
            paths = ensure_models(lambda text: self.events.put(("status", text)))
            if self.stop_event.is_set():
                return
            engine = FaceEngine(*paths)
            pipeline = FacePipeline(engine, source=f"camera:{self.index}",
                                    save_interval=self.save_interval, progress=self._stage,
                                    threshold=self.threshold)
            self.events.put(("status", f"카메라 {self.index} 연결 중…"))
            configured_profile = pipeline.profile
            capture = open_camera(self.index, size=configured_profile.size, fps=configured_profile.fps)
            failures = 0
            previous_frame_at = None
            previous_processing_at = None
            while not self.stop_event.is_set():
                while True:
                    try:
                        setting, value = self.commands.get_nowait()
                    except Empty:
                        break
                    if setting == "save_interval":
                        pipeline.save_interval = value
                    elif setting == "threshold":
                        pipeline.threshold = value
                    elif setting == "reset_gate":
                        pipeline.reset_standby()
                if self.pause_event.is_set():
                    self.stop_event.wait(.05)
                    continue
                started = time.monotonic()
                profile = pipeline.profile
                if profile != configured_profile:
                    configure_capture(capture, profile.size, profile.fps)
                    configured_profile = profile
                    previous_frame_at = None
                    previous_processing_at = None
                ok, frame = capture.read()
                if not ok or frame is None:
                    pipeline.reset_standby()
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("카메라 영상이 끊겼습니다. 다른 앱의 카메라 사용 여부를 확인하세요.")
                    self.stop_event.wait(0.1)
                    continue
                failures = 0
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
                self.stop_event.wait(frame_delay(started, time.monotonic(), profile.fps))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            if capture is not None:
                capture.release()
            self.events.put(("stopped", ""))


class FaceApp:
    def __init__(self, root, camera=0, store=None, use_gpio=True, enable_voice=None):
        self.root = root
        self.store = store if store is not None else initialize_database()
        self.worker = None
        self.frames, self.events = Queue(maxsize=1), Queue()
        self.stages = Queue(maxsize=1)
        self.enrollment = None
        self.last_frame_at = 0
        self.gate_ready = False
        self.last_history_refresh = 0
        self.crop_photo = None
        self.applied_save_interval = 1.0
        self.save_interval = tk.StringVar(value="1.0")
        self.save_policy_text = tk.StringVar(value="인증 성공 얼굴만 1초마다 저장")
        self.save_activity = tk.StringVar(value="DB 저장 대기")
        self.last_saved_at = 0
        self.last_matched = False
        self.last_detected_count = 0
        self.last_embedding_errors = 0
        self.has_error = False
        self.door = Stage5Cycle(use_gpio=use_gpio, enable_voice=enable_voice)  # 5단계 결과 처리
        self.mode = tk.StringVar(value="continuous")
        self.door_message = tk.StringVar(value="문이 닫혀 있습니다. 인증 대기 중입니다.")
        self.manual_text = tk.StringVar(value="수동 개방 버튼 (3회 실패 시 활성)")
        low_w, low_h = STANDBY_PROFILE.size
        self.standby_text = f"1단계 절전: {low_w}×{low_h} / {STANDBY_PROFILE.fps} FPS · 임베딩 중지"
        self.power_text = tk.StringVar(value=self.standby_text)
        self.saved_text = tk.StringVar(value="누적 Crop 0개")
        self.photo = None
        self.camera = tk.StringVar(value=str(camera))
        self.name = tk.StringVar()
        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.status = tk.StringVar(value="카메라 시작을 누르면 미리보기가 켜집니다.")
        self.result = tk.StringVar(value="인식 대기 중")
        self.threshold_text = tk.StringVar(value=f"{DEFAULT_THRESHOLD:.2f}")
        self._build()
        self._refresh_people()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(40, self._poll)

    def _build(self):
        self.root.title("1~5단계 통합 · 얼굴 등록 및 인증")
        self.root.geometry("1060x940")
        self.root.minsize(1020, 900)
        self.root.configure(bg="#f4f6fa")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6fa")
        style.configure("TLabel", background="#f4f6fa", font=("맑은 고딕", 10))
        style.configure("TButton", font=("맑은 고딕", 10), padding=5)
        style.configure("Title.TLabel", font=("맑은 고딕", 22, "bold"))
        style.configure("Result.TLabel", font=("맑은 고딕", 13, "bold"))

        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="내 얼굴 인식", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="실시간 검출·인식 / 인증 성공 얼굴만 DB 저장 / 사용자 등록은 별도 탭").pack(anchor="w", pady=(4, 12))
        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)
        left, right = ttk.Frame(content), ttk.Frame(content, padding=(20, 0, 0, 0))
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="y")

        self.indicators = StatusPanel(left)
        self.indicators.pack(fill="x", pady=(0, 10))
        ttk.Label(left, textvariable=self.power_text, foreground="#0369a1", wraplength=640).pack(anchor="w", pady=(0, 6))
        self.canvas = tk.Canvas(left, width=640, height=480, bg="#141d2d", highlightthickness=0)
        self.canvas.pack(anchor="n")
        self._placeholder()
        ttk.Label(left, textvariable=self.door_message, foreground="#166534",
                  font=("맑은 고딕", 16, "bold"), wraplength=640).pack(anchor="w", pady=(10, 0))
        ttk.Label(left, textvariable=self.result, style="Result.TLabel", wraplength=640).pack(anchor="w", pady=(12, 4))
        ttk.Label(left, textvariable=self.status, wraplength=640).pack(anchor="w")

        ttk.Label(right, text="작동 모드", style="Result.TLabel").pack(anchor="w")
        mode_row = ttk.Frame(right)
        mode_row.pack(fill="x", pady=4)
        ttk.Radiobutton(mode_row, text="연속 작동", variable=self.mode, value="continuous",
                        command=self.apply_mode).pack(side="left")
        ttk.Radiobutton(mode_row, text="1회 작동", variable=self.mode, value="once",
                        command=self.apply_mode).pack(side="right")
        ttk.Label(right, text="1회: 인증 → 열림 → 닫힘 후 중지").pack(anchor="w", pady=(0, 8))
        ttk.Label(right, text="카메라 연결", style="Result.TLabel").pack(anchor="w")
        row = ttk.Frame(right)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="카메라 번호").pack(side="left")
        self.camera_input = ttk.Spinbox(row, from_=0, to=9, width=5, textvariable=self.camera)
        self.camera_input.pack(side="right")
        self.start_button = ttk.Button(right, text="카메라 시작", command=self.start_camera)
        self.start_button.pack(fill="x")
        self.stop_button = ttk.Button(right, text="카메라 중지", command=self.stop_camera, state="disabled")
        self.stop_button.pack(fill="x", pady=(6, 6))
        self.manual_button = ttk.Button(right, textvariable=self.manual_text, command=self.press_manual, state="disabled")
        self.manual_button.pack(fill="x", pady=(0, 12))

        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True)
        auth_tab, enroll_tab, history_tab = (ttk.Frame(tabs, padding=8) for _ in range(3))
        self.tabs, self.auth_tab = tabs, auth_tab   # 5단계(출입 처리)는 '인식·기록' 탭에서만
        tabs.add(auth_tab, text="인식·기록")
        tabs.add(enroll_tab, text="사용자 등록")
        tabs.add(history_tab, text="수집 기록")
        ttk.Label(auth_tab, text="대기 조건과 사용자 인증을 모두\n통과한 얼굴만 DB에 저장합니다.\n\n미등록·인증 실패·여러 얼굴은\n저장하지 않습니다. 새 사용자 등록은\n‘사용자 등록’ 탭에서 진행하세요.").pack(anchor="w", pady=8)
        ttk.Label(auth_tab, textvariable=self.saved_text, style="Result.TLabel").pack(anchor="w", pady=8)
        ttk.Label(auth_tab, text="DB 저장 간격 (초)").pack(anchor="w", pady=(8, 4))
        storage_row = ttk.Frame(auth_tab)
        storage_row.pack(fill="x")
        ttk.Spinbox(storage_row, from_=0.2, to=60, increment=0.5, width=7,
                    textvariable=self.save_interval).pack(side="left")
        ttk.Button(storage_row, text="적용", command=self.apply_save_interval).pack(side="right")
        ttk.Label(auth_tab, textvariable=self.save_policy_text, wraplength=240).pack(anchor="w", pady=4)
        ttk.Label(auth_tab, textvariable=self.save_activity, wraplength=240).pack(anchor="w", pady=(0, 8))
        ttk.Label(auth_tab, text="LED: 흰색 대기 / 노랑 조건 확인\n파랑 저장 / 초록 인증 / 빨강 미등록\n보라 사용자 등록 / 회색 중지\n빨강 깜빡임 수동 개방 버튼 대기", wraplength=240).pack(anchor="w", pady=8)
        ttk.Label(enroll_tab, text="이름을 지정한 경우에만 등록합니다.").pack(anchor="w")
        self.name_input = ttk.Entry(enroll_tab, textvariable=self.name, width=25, font=("맑은 고딕", 11))
        self.name_input.pack(fill="x", pady=8)
        self.register_button = ttk.Button(enroll_tab, text="이름 입력 후 얼굴 등록", command=self.register, state="disabled")
        self.register_button.pack(fill="x")
        self.cancel_button = ttk.Button(enroll_tab, text="등록 취소", command=self.cancel_enrollment, state="disabled")
        self.cancel_button.pack(fill="x", pady=4)
        self.progress = ttk.Progressbar(enroll_tab, maximum=REGISTRATION_SAMPLES)
        self.progress.pack(fill="x", pady=(0, 6))
        ttk.Label(enroll_tab, text="얼굴 한 명 · 선명한 샘플 15개\n정면을 유지하며 조금씩 움직여주세요.").pack(anchor="w")

        ttk.Label(enroll_tab, text="등록된 얼굴", style="Result.TLabel").pack(anchor="w", pady=(12, 6))
        self.people = tk.Listbox(enroll_tab, height=3, font=("맑은 고딕", 11), exportselection=False)
        self.people.pack(fill="x")
        self.delete_button = ttk.Button(enroll_tab, text="선택한 등록 삭제", command=self.delete_person)
        self.delete_button.pack(fill="x", pady=6)

        ttk.Label(auth_tab, text="인식 기준 · 높을수록 엄격").pack(anchor="w", pady=(6, 0))
        ttk.Scale(auth_tab, from_=0.30, to=0.80, variable=self.threshold,
                  command=self.apply_threshold).pack(fill="x")
        ttk.Label(auth_tab, textvariable=self.threshold_text).pack(anchor="e")
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
        ttk.Label(outer, text="얼굴 Crop·임베딩은 stages/data/faces.sqlite3, 5단계 보안 로그·결과 사진은 logs/에 저장됩니다.").pack(anchor="w", pady=(16, 0))

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
        threshold = float(value)
        self.threshold_text.set(f"{threshold:.2f}")
        if self.worker and not self.worker.stop_event.is_set():
            self.worker.commands.put(("threshold", threshold))

    def apply_mode(self):
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

    def start_camera(self):
        if self.worker is not None:
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
        self.door.reset()
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
                                   self.applied_save_interval, self.threshold.get())
        self.worker.start()

    def stop_camera(self):
        if self.worker is not None:
            if self.door.phase != "waiting":
                self.door_message.set("문이 닫혔습니다.")
            self.door.reset()
            self.worker.stop_event.set()
            self.stop_button.configure(state="disabled")
            self.register_button.configure(state="disabled")
            self.cancel_enrollment()
            self.status.set("카메라를 종료하고 있습니다…")
            self.result.set("인식 중지")
            self._placeholder("카메라 중지 중…")
            self.indicators.idle("카메라 중지 중")

    def register(self):
        if self.worker is None or self.worker.stop_event.is_set() or time.monotonic() - self.last_frame_at > 2:
            messagebox.showinfo("얼굴 등록", "먼저 카메라 영상을 확인해주세요.")
            return
        if not self.gate_ready:
            messagebox.showinfo("얼굴 등록", "얼굴을 중앙에 두고 정면을 바라보며 대기 조건을 통과해주세요.")
            return
        name = self.name.get().strip()
        if not name or len(name) > 40:
            messagebox.showinfo("얼굴 등록", "이름을 1~40자로 입력하세요.")
            return
        if name in self.store.people and not messagebox.askyesno("다시 등록", f"‘{name}’의 기존 임베딩을 새로 등록할까요?"):
            return
        self.enrollment = Enrollment(name)
        self.progress["value"] = 0
        self.register_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.name_input.configure(state="disabled")
        self.delete_button.configure(state="disabled")
        self.status.set("정면을 바라보며 얼굴을 화면에 맞춰주세요.")

    def cancel_enrollment(self):
        self.enrollment = None
        self.progress["value"] = 0
        self.cancel_button.configure(state="disabled")
        self.name_input.configure(state="normal")
        self.delete_button.configure(state="normal")
        if self.worker and not self.worker.stop_event.is_set() and self.gate_ready:
            self.register_button.configure(state="normal")

    def delete_person(self):
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
        self.last_frame_at = captured_at
        self.gate_ready = gate_ready and bool(faces)
        self.last_matched = False
        self.last_detected_count = telemetry.detected_count if telemetry else len(faces)
        self.last_embedding_errors = telemetry.embedding_errors if telemetry else 0
        if telemetry is not None:
            if telemetry.saved_this_frame:
                self.last_saved_at = time.monotonic()
            self.save_activity.set("인증 성공 시에만 저장" if telemetry.authenticated_name is None else
                                   (f"이번에 {telemetry.saved_this_frame}장 저장" if telemetry.saved_this_frame
                                    else f"다음 저장까지 {telemetry.next_save_in:.1f}초"))
            self.indicators.set_stage(1, "얼굴 대기" if not telemetry.detected_count else
                                      ("조건 통과" if self.gate_ready else "조건 확인 중"),
                                      "green" if self.gate_ready else "yellow")
            self.indicators.set_stage(2, f"얼굴 {telemetry.detected_count}개 검출", "blue" if telemetry.detected_count else "off")
            self.indicators.set_stage(3, f"{len(faces)}개 생성 / {telemetry.embedding_errors}개 실패",
                                      "red" if telemetry.embedding_errors else ("green" if faces else "off"))
            if telemetry.power_mode == "standby":
                self.indicators.set_stage(3, "절전 · 임베딩 중지")
                self.indicators.set_stage(1, "절전 · 조건 확인", "yellow")
            mode_text = "1단계 절전" if telemetry.power_mode == "standby" else "2~4단계 활성"
            width, height = telemetry.processing_size
            details = f"{mode_text}: {width}×{height} / 목표 {telemetry.target_fps} FPS · 실제 처리 {fps:.1f} FPS"
            if telemetry.capture_size is not None:
                native_w, native_h = telemetry.capture_size
                details += f" | 카메라 {native_w}×{native_h} / 보고 {telemetry.capture_fps or 0:g} FPS"
            self.power_text.set(details)
            self.indicators.set_stage(4, self.save_activity.get(), "blue" if telemetry.saved_this_frame else "off")
        if self.enrollment:
            status = self.enrollment.accept(faces if self.gate_ready else [], captured_at)
            self.status.set(gate_status if not self.gate_ready and gate_status else status)
            self.progress["value"] = len(self.enrollment.samples)
            if captured_at - self.enrollment.started > 60:
                self.cancel_enrollment()
                self.status.set("등록 시간이 초과되었습니다. 조명과 위치를 조절한 후 다시 등록하세요.")
            elif len(self.enrollment.samples) == REGISTRATION_SAMPLES:
                name = self.enrollment.name
                try:
                    add_user(self.store, name, self.enrollment.samples)
                except (ValueError, OSError) as exc:
                    self.cancel_enrollment()
                    messagebox.showerror("등록 실패", str(exc))
                else:
                    self.cancel_enrollment()
                    self._refresh_people()
                    self.status.set(f"‘{name}’ 등록 완료 · 128차원 임베딩을 저장했습니다.")
        else:
            self.register_button.configure(state="normal" if self.gate_ready else "disabled")
            if gate_status:
                self.status.set(gate_status)

        self.canvas.delete("all")
        height, width = frame.shape[:2]
        scale = min(640 / width, 480 / height)
        preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
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
            label = (name or "미등록 · 저장 안 함") if self.gate_ready else "인증 대기 · 저장 안 함"
            if score is not None:
                label += f" · {score:.3f}"
            labels.append(label)
            x, y, w, h = face.box
            x, y, w, h = x * scale + offset_x, y * scale + offset_y, w * scale, h * scale
            self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=3)
            text = self.canvas.create_text(max(4, x), max(4, y - 26), text=label, anchor="nw",
                                           fill=color, font=("맑은 고딕", 11, "bold"))
            background = self.canvas.create_rectangle(self.canvas.bbox(text), fill="#141d2d", outline="")
            self.canvas.tag_lower(background, text)
        self.result.set(" / ".join(labels) if labels else "대기 조건 확인 중")
        self.canvas.create_text(10, 468, anchor="sw", text=f"{fps:.0f} FPS · 유사도 기준 {self.threshold.get():.2f}",
                                fill="white", font=("맑은 고딕", 10))
        if saved_count is not None:
            self.saved_text.set(f"누적 Crop {saved_count:,}개")
        if self.gate_ready:
            self.indicators.set_stage(4, ("인증 성공" if self.last_matched else "미등록 얼굴")
                                      + " / " + ("저장 안 함" if not self.last_matched else
                                                 ("저장 완료" if telemetry and telemetry.saved_this_frame else "저장 대기")),
                                      "green" if self.last_matched else "red")
        # 4단계 인증이 실제로 수행된 프레임(조건 통과 + 얼굴 1개)이면 결과를 5단계로 넘김(성공·실패 모두)
        attempted = (telemetry is not None and self.gate_ready and telemetry.detected_count == 1
                     and len(faces) == 1)
        on_auth_tab = self.tabs.select() == str(self.auth_tab)
        if attempted and self.enrollment is None and self.worker is not None and not on_auth_tab:
            self.door_message.set("‘사용자 등록’·‘수집 기록’ 탭에서는 출입 처리(5단계)를 하지 않습니다.")
        elif attempted and self.enrollment is None and self.worker is not None and not self.store.people:
            self.door_message.set("등록된 사용자가 없습니다. ‘사용자 등록’ 탭에서 먼저 등록하세요.")
        elif attempted and self.enrollment is None and self.worker is not None:
            name, score = telemetry.authenticated_name, telemetry.match_score
            if self.door.start(time.monotonic(), name, score, frame=frame,
                               auth_ms=getattr(telemetry, "process_ms", None)):
                self.worker.pause_event.set()
                if name:   # 성공 문구는 임시통합 그대로
                    self.door_message.set("문이 열렸습니다.")
                    self.status.set("3초 후 문이 닫힙니다. 열림 중에는 인증·DB 저장을 일시 정지합니다.")
                    self.indicators.phase.set("현재 단계: 인증 완료 → 문 열림")
                else:      # 실패 처리는 5단계에서 추가
                    self.door_message.set("인증 실패")
                    self.status.set("인증 실패 처리 중에는 인증·DB 저장을 일시 정지합니다.")
                    self.indicators.phase.set("현재 단계: 인증 실패 → 5단계 결과 처리")
                self.indicators.set_stage(5, "처리 중", "yellow")
                self.register_button.configure(state="disabled")
        self._update_led()

    def press_manual(self):
        if self.door.press_manual():
            self.status.set("수동 개방 버튼을 눌렀습니다.")

    def _advance_door(self):
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
                    self.worker.commands.put(("reset_gate", None))
                    self.worker.pause_event.clear()
            self.register_button.configure(state="disabled")

    def _update_led(self):
        if self.has_error:
            self.indicators.set_led("red", "오류 · 상태 메시지 확인", blink=True)
        elif self.door.phase == "processing":
            color, blink, text = self.door.led_state()
            self.indicators.set_led(color, text, blink=blink)
        elif self.worker is None or self.worker.stop_event.is_set():
            self.indicators.set_led("off", "카메라 꺼짐")
        elif self.last_frame_at and time.monotonic() - self.last_frame_at > 2:
            self.indicators.set_led("yellow", "카메라 프레임 대기", blink=True)
        elif self.enrollment:
            self.indicators.set_led("purple", f"사용자 등록 {len(self.enrollment.samples)}/{REGISTRATION_SAMPLES}", blink=True)
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
        self._advance_door()
        try:
            phase = self.stages.get_nowait()
        except Empty:
            pass
        else:
            if self.worker and not self.worker.stop_event.is_set() and self.door.phase == "waiting" and not self.door.completed:
                self.indicators.phase.set(f"현재 단계: {phase}")
        if time.monotonic() - self.last_history_refresh >= 1:
            self._refresh_history()
            self.last_history_refresh = time.monotonic()
        while True:
            try:
                kind, value = self.events.get_nowait()
            except Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == "error":
                self.has_error = True
                self.status.set(value)
                self._placeholder("카메라를 연결하지 못했습니다.\n상태 메시지를 확인한 후 다시 시작하세요.")
                messagebox.showerror("카메라 연결 오류", value)
            elif kind == "stopped":
                self.worker = None
                self.gate_ready = False
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
                if self.door.phase == "waiting" and not self.door.completed:
                    self._handle_frame(*packet)
            if self.door.phase == "waiting" and self.last_frame_at and time.monotonic() - self.last_frame_at > 2:
                self.register_button.configure(state="disabled")
                self.result.set("카메라 영상을 기다리고 있습니다…")
                self._placeholder("카메라 영상을 기다리고 있습니다…")
        self._update_led()
        self.root.after(40, self._poll)

    def close(self):
        self.door.close()   # 5단계 장치(LED·부저·버튼)·음성 정리
        if self.worker:
            self.worker.stop_event.set()
            # Keep the process alive until the capture is released.
            self.root.withdraw()
            self._wait_for_close()
        else:
            self.root.destroy()

    def _wait_for_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(50, self._wait_for_close)
        else:
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
