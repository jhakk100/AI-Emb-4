"""Standalone YuNet direction threshold tester; never enrolls or operates the door."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from queue import Empty, Full, Queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent / 'stages'))

from capture_control import ACTIVE_PROFILE, ENROLLMENT_PROFILE, EnrollmentResolutionGate, frame_delay, prepare_frame
from detection import detect_faces
from face_core import open_camera
from guided_capture import DIRECTIONS, LABELS, direction_matches, load_guided_settings, yunet_pose
from models import ROOT, ensure_models
from pose_tuning import PoseTuner
from landmark_filter import LandmarkFilter, LandmarkFilterSettings
from manual_landmarks import ManualLandmarks
from standby import overlap_ratio
from pose_calibration import ORDER, PoseCalibration, PoseProfile
from pose_defaults import load_default_preset

CONFIG = ROOT / 'guided_capture.json'
CONTROLS = (
    ('front_degrees', '정면 허용각 (°)', 1, 40, 1),
    ('yaw_min', '좌우 최소각 (°)', 2, 55, 1),
    ('yaw_max', '좌우 최대각 (°)', 3, 59, 1),
    ('pitch_min', '상하 최소각 (°)', 1, 55, 1),
    ('pitch_max', '상하 최대각 (°)', 2, 59, 1),
    ('cross_degrees', '다른 축 허용각 (°)', 1, 29, 1),
    ('roll_max', '고개 기울임 허용각 (°)', 1, 44, 1),
    ('hold_seconds', '자세 유지 시간 (초)', .05, 3, .05),
    ('min_frames', '최소 연속 프레임', 2, 30, 1),
    ('max_gap', '추적 간격 제한 (초)', .1, 5, .1),
)
ANGLE_FIELDS = {key for key, *_ in CONTROLS if key.endswith('degrees') or key.startswith(('yaw_', 'pitch_', 'roll_'))}


def angle_text(value):
    return f'{int(value):+d}°'


class PoseCamera(threading.Thread):
    def __init__(self, index, profile, frames, events):
        super().__init__(daemon=True)
        self.index, self.profile, self.frames, self.events = index, profile, frames, events
        self.stop_event = threading.Event()

    def run(self):
        capture = None
        try:
            path = ensure_models(lambda text: self.events.put(('status', text)), detector_only=True)[0]
            detector = cv2.FaceDetectorYN.create('onnx', np.frombuffer(path.read_bytes(), np.uint8),
                np.empty(0, np.uint8), (640, 480), .5, .3, 5000)
            if self.stop_event.is_set():
                return
            capture = open_camera(self.index, size=self.profile.size, fps=self.profile.fps)
            gate = EnrollmentResolutionGate()
            failures = 0
            while not self.stop_event.is_set():
                started = time.monotonic()
                ok, native = capture.read()
                if not ok or native is None:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError('카메라 영상을 받을 수 없습니다.')
                    self.stop_event.wait(.1)
                    continue
                failures = 0
                if not gate.accept(native, self.profile):
                    self.stop_event.wait(.05)
                    continue
                native_size = native.shape[1], native.shape[0]
                frame = prepare_frame(cv2.flip(native, 1), self.profile.size)
                rows = detect_faces(detector, frame, size=ACTIVE_PROFILE.size)
                raw, reason = None, ''
                if len(rows) == 1:
                    try:
                        raw = yunet_pose(rows[0], self.profile.size)
                    except (ValueError, cv2.error) as exc:
                        reason = str(exc)
                else:
                    reason = '얼굴 한 명만 보여주세요.'
                packet = (frame, rows, raw, reason, native_size, time.monotonic())
                try:
                    self.frames.put_nowait(packet)
                except Full:
                    try:
                        self.frames.get_nowait()
                    except Empty:
                        pass
                    self.frames.put_nowait(packet)
                self.stop_event.wait(frame_delay(started, time.monotonic(), self.profile.fps))
        except Exception as exc:
            self.events.put(('error', str(exc)))
        finally:
            try:
                if capture is not None:
                    capture.release()
            except Exception as exc:
                self.events.put(('error', f'카메라 정리 오류: {exc}'))
            finally:
                self.events.put(('stopped', ''))


class PoseTestApp:
    def __init__(self, root, camera=0, config=CONFIG):
        self.root, self.config = root, Path(config)
        self.tuner = PoseTuner(load_guided_settings(self.config))
        self.landmarks = LandmarkFilter()
        self.manual = ManualLandmarks()
        self.guided = PoseCalibration()
        self.guided_context = None
        self.calibration_file = self.config.with_name('pose_calibration.json')
        self.p_key_down = False
        self.latest_packet = self.last_base_row = None
        self.edit_frame = self.edit_selected = None
        self.edit_tracking_ok = True
        self.worker = None
        self.frames, self.events = Queue(maxsize=1), Queue()
        self.closing = False
        self.last_packet_at = None
        self.photo = None
        self.vars = {}
        self.camera = tk.StringVar(value=str(camera))
        self.resolution = tk.StringVar(value='1920×1080 등록')
        self.target = tk.StringVar(value='front')
        self.yaw_reverse, self.pitch_reverse = tk.BooleanVar(), tk.BooleanVar()
        self.use_stabilization = tk.BooleanVar(value=True)
        self.show_raw_points = tk.BooleanVar(value=True)
        self.filter_vars = {key: tk.StringVar(value=value) for key, value in
                            (('smoothing_seconds', '.18'), ('jump_percent', '12'), ('min_confidence', '80'))}
        self.status = tk.StringVar(value='보정 시작 → 정면·좌·우·위에서 각각 P 키 → 보정 테스트')
        self.angles = tk.StringVar(value='정면 기준 미설정')
        self.capture_info = tk.StringVar(value='카메라 꺼짐 · 사진/DB 저장 없음 · 출입 처리 없음')
        self.verdict = tk.StringVar(value='보정 시작을 누르고 요청 자세에서 P 키를 누르세요.')
        self.filter_info = tk.StringVar(value='하늘색 원: 원본 점 / 초록 십자: 안정화 점 · 카메라 시작 후 새 영점 설정')
        self.manual_info = tk.StringVar(value='수동 보정 없음 · 정면에서 수동 위치 조정을 시작하세요.')
        self.guided_info = tk.StringVar(value='보정 시작 → 정면·좌·우·위에서 P 키 → 테스트')
        self._build()
        self.load_values()
        self.load_guided_profile()
        self.refresh_guided_view()
        root.bind('<KeyPress>', self.on_guided_key_press, add='+')
        root.bind('<KeyRelease>', self.on_guided_key_release, add='+')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(40, self.poll)

    def _build(self):
        root = self.root
        root.title('얼굴 방향 · P 키 보정 및 테스트')
        root.geometry('1140x940')
        outer = ttk.Frame(root, padding=15)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='얼굴 방향 판정 · 수동 임계값 테스트', font=('맑은 고딕', 18, 'bold')).pack(anchor='w')
        ttk.Label(outer, text='요청 자세에서 P 키로 기준을 기록합니다 · 등록/인증 앱과 분리된 테스트 도구입니다.').pack(anchor='w')
        body = ttk.Frame(outer)
        body.pack(fill='both', expand=True, pady=10)
        left, right = ttk.Frame(body), ttk.Frame(body, padding=(18, 0, 0, 0))
        left.pack(side='left', fill='both', expand=True)
        right.pack(side='right', fill='y')
        self.canvas = tk.Canvas(left, width=640, height=480, bg='#141d2d', highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.select_manual_point)
        self.canvas.bind('<B1-Motion>', self.drag_manual_point)
        self.canvas.bind('<ButtonRelease-1>', lambda event: self.canvas.configure(cursor=''))
        for key in ('Left', 'Right', 'Up', 'Down'):
            self.canvas.bind(f'<{key}>', self.nudge_manual_point)
        ttk.Label(left, textvariable=self.angles, font=('맑은 고딕', 12), wraplength=640).pack(anchor='w', pady=5)
        ttk.Label(left, textvariable=self.capture_info, wraplength=640).pack(anchor='w')
        ttk.Label(left, textvariable=self.filter_info, wraplength=640, foreground='#15803d').pack(anchor='w', pady=4)
        self.table = ttk.Treeview(left, columns=('direction', 'angle', 'condition'), show='headings', height=5)
        for column, title, width in (('direction', '방향', 95), ('angle', '현재 각도 판정', 110), ('condition', '통과 조건', 400)):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, stretch=column == 'condition')
        self.table.pack(fill='x', pady=8)
        for direction in ORDER:
            self.table.insert('', 'end', iid=direction, values=(LABELS[direction], '대기', ''))
        row = ttk.Frame(right)
        row.pack(fill='x')
        ttk.Label(row, text='카메라 번호').pack(side='left')
        self.camera_input = ttk.Spinbox(row, from_=0, to=9, textvariable=self.camera, width=4)
        self.camera_input.pack(side='right')
        self.size_input = ttk.Combobox(right, textvariable=self.resolution, state='readonly',
                                      values=('1920×1080 등록', '640×480 인증'), width=27)
        self.size_input.pack(fill='x', pady=5)
        row = ttk.Frame(right)
        row.pack(fill='x')
        self.start_button = ttk.Button(row, text='카메라 시작', command=self.start)
        self.start_button.pack(side='left', expand=True, fill='x')
        self.stop_button = ttk.Button(row, text='중지', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', expand=True, fill='x')
        self.calibrate_button = ttk.Button(right, text='보정 시작 · 정면 → 좌 → 우 → 위', command=self.start_calibration)
        self.calibrate_button.pack(fill='x', pady=8)
        row = ttk.Frame(right)
        row.pack(fill='x')
        self.record_button = ttk.Button(row, text='P · 현재 자세 기록', command=self.record_pose, state='disabled')
        self.record_button.pack(side='left', fill='x', expand=True)
        self.test_button = ttk.Button(row, text='보정 테스트', command=self.start_calibration_test, state='disabled')
        self.test_button.pack(side='left', fill='x', expand=True)
        ttk.Label(right, textvariable=self.guided_info, wraplength=380).pack(anchor='w', pady=5)
        ttk.Label(right, text='현재 시험할 방향').pack(anchor='w')
        targets = ttk.Frame(right)
        targets.pack(fill='x', pady=4)
        self.target_buttons = []
        for direction in ORDER:
            button = ttk.Radiobutton(targets, text=LABELS[direction].replace('화면 ', ''),
                                     value=direction, variable=self.target)
            button.pack(side='left')
            self.target_buttons.append(button)
        ttk.Label(right, textvariable=self.verdict, wraplength=380,
                  font=('맑은 고딕', 12, 'bold')).pack(anchor='w', pady=6)
        notebook = ttk.Notebook(right)
        notebook.pack(fill='x', pady=5)
        guided_tab, angle_tab, filter_tab, manual_tab = (ttk.Frame(notebook, padding=5) for _ in range(4))
        notebook.add(guided_tab, text='보정·테스트')
        notebook.add(angle_tab, text='고급 각도')
        notebook.add(filter_tab, text='좌표 안정화')
        notebook.add(manual_tab, text='고급 점 조정')
        notebook.select(guided_tab)
        self.guided_table = ttk.Treeview(guided_tab, columns=('direction', 'angles', 'state'),
                                         show='headings', height=5)
        for key, title, width in (('direction', '순서', 65), ('angles', 'P로 기록한 기준각도', 175), ('state', '상태', 90)):
            self.guided_table.heading(key, text=title)
            self.guided_table.column(key, width=width)
        self.guided_table.pack(fill='x', pady=6)
        for direction in ORDER:
            self.guided_table.insert('', 'end', iid=direction, values=(LABELS[direction].replace('화면 ', ''), '미기록', '대기'))
        row = ttk.Frame(guided_tab)
        row.pack(fill='x', pady=5)
        self.back_button = ttk.Button(row, text='이전 자세 다시', command=self.previous_calibration_pose, state='disabled')
        self.back_button.pack(side='left', expand=True, fill='x')
        self.guided_cancel_button = ttk.Button(row, text='보정/테스트 취소', command=self.cancel_calibration, state='disabled')
        self.guided_cancel_button.pack(side='left', expand=True, fill='x')
        ttk.Label(guided_tab, text='숫자를 입력할 필요가 없습니다.\n'
                  '1. 보정 시작을 누릅니다.\n2. 요청한 방향을 보고 잠시 멈춘 뒤 P 키를 누릅니다.\n'
                  '3. 정면 → 왼쪽 → 오른쪽 → 위를 모두 기록합니다.\n'
                  '4. 보정 테스트를 누르고 같은 순서의 요청을 수행합니다.\n'
                  '테스트는 자세를 유지하면 자동으로 확인합니다. P 키는 필요 없습니다.\n'
                  '기준 4개는 자동 저장합니다. 사람이나 카메라 위치가 바뀌면 다시 보정하세요.',
                  wraplength=370).pack(anchor='w', pady=7)
        tuning = ttk.Frame(angle_tab)
        tuning.pack(fill='x')
        self.angle_inputs = []
        for index, (key, label, low, high, step) in enumerate(CONTROLS):
            ttk.Label(tuning, text=label).grid(row=index, column=0, sticky='w', pady=3)
            var = tk.StringVar()
            self.vars[key] = var
            control = ttk.Spinbox(tuning, from_=low, to=high, increment=step, textvariable=var, width=8)
            control.grid(row=index, column=1, sticky='e', padx=6)
            self.angle_inputs.append(control)
        ttk.Button(angle_tab, text='정면 기준만 설정 · 기존 방식', command=self.calibrate).pack(fill='x', pady=4)
        ttk.Checkbutton(angle_tab, text='좌우 판정 반전', variable=self.yaw_reverse).pack(anchor='w', pady=(6, 0))
        ttk.Checkbutton(angle_tab, text='상하 판정 반전', variable=self.pitch_reverse).pack(anchor='w')
        ttk.Label(angle_tab, text='입력 조건: 정면 < 좌우 최소 < 최대 < 60°\n상하 최소 < 최대 < 60° · 유지 시간 < 추적 간격\n고개 기울임은 정면 기준으로 빼지 않고 원래 값을 검사합니다.',
                  wraplength=380).pack(anchor='w', pady=7)
        self.stabilize_check = ttk.Checkbutton(filter_tab, text='안정화된 점으로 각도/방향 판정',
                                               variable=self.use_stabilization, command=self.reset_stabilization)
        self.stabilize_check.pack(anchor='w', pady=4)
        ttk.Checkbutton(filter_tab, text='원본 점도 함께 표시', variable=self.show_raw_points).pack(anchor='w')
        for key, label, low, high, step in (
                ('smoothing_seconds', '반응 시간 (초)', .02, 1, .02),
                ('jump_percent', '한 점 튐 제한 (얼굴 폭 %)', 2, 50, 1),
                ('min_confidence', '최소 얼굴 신뢰도 (%)', 50, 99, 1)):
            row = ttk.Frame(filter_tab)
            row.pack(fill='x', pady=6)
            ttk.Label(row, text=label).pack(side='left')
            ttk.Spinbox(row, from_=low, to=high, increment=step, width=7,
                        textvariable=self.filter_vars[key]).pack(side='right')
        self.filter_button = ttk.Button(filter_tab, text='좌표 안정화 적용 / 영점 초기화', command=self.apply_filter_values)
        self.filter_button.pack(fill='x', pady=8)
        ttk.Label(filter_tab, text='반응 시간을 늘리면 더 부드럽지만 움직임에 늦게 반응합니다.\n'
                  '하늘색 원: 원본 / 초록 십자: 안정화 / 연결선: 보정량\n'
                  '체크를 해제하면 원본 각도로 판정하며 유효성 검사는 유지합니다.\n'
                  '급변 값은 제외하고 새 연속 3프레임을 기다립니다.\n'
                  '얼굴 소실·다중 얼굴·추적 단절 시 좌표와 영점을 초기화합니다.\n'
                  '이 탭의 보정값은 이번 시험에만 사용하며 파일에 저장하지 않습니다.',
                  wraplength=365).pack(anchor='w', pady=7)
        self.manual_begin_button = ttk.Button(manual_tab, text='수동 위치 조정 · 영상 정지', command=self.begin_manual)
        self.manual_begin_button.pack(fill='x', pady=4)
        self.manual_apply_button = ttk.Button(manual_tab, text='조정 적용 · 실시간 복귀', command=self.apply_manual, state='disabled')
        self.manual_apply_button.pack(fill='x', pady=4)
        self.manual_cancel_button = ttk.Button(manual_tab, text='조정 취소 · 실시간 복귀', command=self.cancel_manual, state='disabled')
        self.manual_cancel_button.pack(fill='x', pady=4)
        ttk.Button(manual_tab, text='수동 보정 지우기 · 자동 검출로 초기화', command=self.reset_manual).pack(fill='x', pady=4)
        self.point_table = ttk.Treeview(manual_tab, columns=('name', 'x', 'y'), show='headings', height=5)
        for key, title, width in (('name', '화면 기준 위치', 140), ('x', '가로 좌표', 70), ('y', '세로 좌표', 70)):
            self.point_table.heading(key, text=title)
            self.point_table.column(key, width=width)
        self.point_table.pack(fill='x', pady=5)
        ttk.Label(manual_tab, textvariable=self.manual_info, wraplength=365).pack(anchor='w', pady=5)
        ttk.Label(manual_tab, text='정면을 보고 영상 정지 → 노란 점을 드래그 → 적용\n'
                  '점을 선택한 뒤 방향키로 원본 영상의 1픽셀씩 움직일 수 있습니다.\n'
                  '조정하는 동안 고개를 움직이지 마세요. 위치 보정은 얼굴 크기에 맞춰 적용합니다.\n'
                  '적용 후 정면 기준을 다시 설정하세요. 보정값은 파일에 저장하지 않습니다.',
                  wraplength=365).pack(anchor='w', pady=5)
        self.reload_button = ttk.Button(right, text='저장값 다시 불러오기', command=self.reload)
        self.reload_button.pack(fill='x', pady=3)
        self.save_button = ttk.Button(right, text='임계값 저장 · 등록/인증에 공통 적용', command=self.save)
        self.save_button.pack(fill='x', pady=3)
        ttk.Label(outer, textvariable=self.status, wraplength=1080).pack(anchor='w', pady=5)

    def load_values(self):
        for key, var in self.vars.items():
            value = getattr(self.tuner.settings, key)
            var.set(str(int(value)) if key in ANGLE_FIELDS else f'{value:g}')
        self.yaw_reverse.set(self.tuner.settings.yaw_sign == -1)
        self.pitch_reverse.set(self.tuner.settings.pitch_sign == -1)

    def apply_values(self):
        values = {key: float(var.get()) for key, var in self.vars.items()}
        if not values['min_frames'].is_integer():
            raise ValueError('최소 프레임 수는 정수여야 합니다.')
        values['min_frames'] = int(values['min_frames'])
        if any(not values[key].is_integer() for key in ANGLE_FIELDS):
            raise ValueError('각도는 소수점 없이 정수로 입력하세요.')
        values.update(yaw_sign=-1 if self.yaw_reverse.get() else 1,
                      pitch_sign=-1 if self.pitch_reverse.get() else 1)
        self.tuner.configure(**values)
        if self.landmarks.settings.max_gap != self.tuner.settings.max_gap:
            self.landmarks.settings = replace(self.landmarks.settings, max_gap=self.tuner.settings.max_gap)

    def calibration_context(self):
        profile = ENROLLMENT_PROFILE if self.resolution.get().startswith('1920') else ACTIVE_PROFILE
        return dict(camera=int(self.camera.get()), size=list(profile.size), mirrored=True,
                    solver='yunet-5pt-sqpnp-v1', stabilized=self.use_stabilization.get(),
                    filtering=asdict(replace(self.landmarks.settings, max_gap=self.tuner.settings.max_gap)),
                    manual_offsets=self.manual.offsets.tolist() if self.manual.active else None)

    def profile_compatible(self):
        try:
            return self.guided.profile is not None and self.guided_context == self.calibration_context()
        except (ValueError, tk.TclError):
            return False

    def load_guided_profile(self):
        self.guided.reset()
        self.guided_context = None
        try:
            default = load_default_preset(self.config.parent)
            context = self.calibration_context()
            if default is not None:
                if default.context != context:
                    raise ValueError(f'기본값 {default.name}은 FHD/저장된 좌표 안정화 설정을 사용합니다.')
                profile = default.profile
            elif self.calibration_file.exists():
                profile = PoseProfile.load(self.calibration_file, context)
            else:
                return
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.status.set(f'이전 보정값을 사용할 수 없습니다. 다시 보정하세요. {exc}')
            return
        self.guided.profile, self.guided.mode, self.guided.index = profile, 'ready', len(ORDER)
        self.guided.records = {name: profile.centers[index].copy() for index, name in enumerate(ORDER)}
        self.guided_context = context
        self.status.set(f'기본값 {default.name}을 불러왔습니다. 보정 테스트로 확인하세요.' if default else
                        '저장한 4방향 보정값을 불러왔습니다. 보정 테스트로 확인하세요.')

    def refresh_guided_view(self):
        mode, target = self.guided.mode, self.guided.target
        compatible = self.profile_compatible()
        running = mode in ('collect', 'test')
        editing = self.manual.editing
        self.calibrate_button.configure(state='disabled' if editing else 'normal',
            text='보정 다시 시작 · 정면부터 재기록' if running else '보정 시작 · 정면 → 좌 → 우 → 위')
        self.record_button.configure(state='normal' if mode == 'collect' and not editing else 'disabled')
        self.test_button.configure(state='normal' if compatible and not running and not editing else 'disabled')
        self.back_button.configure(state='normal' if mode == 'collect' and self.guided.index else 'disabled')
        self.guided_cancel_button.configure(state='normal' if running else 'disabled')
        for widget in (*self.target_buttons, *self.angle_inputs, self.stabilize_check, self.filter_button,
                       self.manual_begin_button, self.reload_button, self.save_button):
            widget.configure(state='disabled' if running or editing else 'normal')
        for name in ORDER:
            center = self.guided.records.get(name)
            angles = f'좌우 {angle_text(center[0])} / 상하 {angle_text(center[1])}' if center is not None else '미기록'
            state = '기록 완료' if center is not None else '대기'
            if name in self.guided.results:
                state = '테스트 통과'
            elif name == target:
                state = 'P 키 대기' if mode == 'collect' else '테스트 중'
            self.guided_table.item(name, values=(LABELS[name].replace('화면 ', ''), angles, state))
        if running:
            self.target.set(target)
            self.guided_info.set(f'{"보정" if mode == "collect" else "테스트"} {self.guided.index+1}/{len(ORDER)} · '
                                 f'{LABELS[target]}을 바라보고 ' + ('P 키를 누르세요.' if mode == 'collect' else '잠시 유지하세요.'))
        elif mode == 'tested':
            self.guided_info.set('보정 테스트 완료 · 정면·좌·우·위 4개 모두 통과')
        elif compatible:
            self.guided_info.set('4방향 보정 완료 · 보정 테스트 버튼을 눌러 확인하세요.')
        elif self.guided.profile is not None:
            self.guided_info.set('카메라/좌표 설정이 바뀌었습니다. 보정을 다시 시작하세요.')
        else:
            self.guided_info.set('보정 시작 → 정면·좌·우·위에서 P 키 → 테스트')

    def start_calibration(self):
        if self.manual.editing:
            return
        if self.worker is None:
            self.start()
        if self.worker is None or self.worker.stop_event.is_set():
            self.status.set('카메라를 먼저 시작하세요.')
            return
        self.manual.clear()
        self.manual_info.set('P 키 보정 사용 · 점 위치를 일일이 입력할 필요가 없습니다.')
        self.landmarks.reset()
        self.tuner.reset_baseline()
        self.load_values()
        self.guided.start()
        self.guided_context = self.calibration_context()
        self.p_key_down = False
        self.canvas.focus_set()
        self.status.set('정면 → 좌 → 우 → 위 순서로 요청 자세에서 잠시 멈추고 P 키를 누르세요.')
        self.refresh_guided_view()

    def on_p_press(self, event=None):
        if self.guided.mode != 'collect' or self.manual.editing:
            return
        if not self.p_key_down:
            self.p_key_down = True
            self.record_pose()
        return 'break'

    @staticmethod
    def is_p_key(event):
        return (getattr(event, 'keysym', '').lower() == 'p'
                or getattr(event, 'char', '') in ('ㅔ', 'ㅖ')
                or sys.platform == 'win32' and getattr(event, 'keycode', None) == 80)

    def on_guided_key_press(self, event):
        if self.is_p_key(event):
            return self.on_p_press(event)

    def on_guided_key_release(self, event):
        if self.is_p_key(event):
            self.on_p_release(event)

    def on_p_release(self, event=None):
        self.p_key_down = False

    def record_pose(self):
        if self.manual.editing:
            return
        try:
            if self.guided_context != self.calibration_context():
                raise ValueError('좌표 설정이 달라졌습니다. 보정을 다시 시작하세요.')
            captured = self.guided.capture(time.monotonic())
        except ValueError as exc:
            self.status.set(str(exc))
            return
        front = self.guided.records['front']
        self.tuner.origin = tuple(front[:2])
        self.status.set(f'{LABELS[captured]} 기준 기록 완료 · 다음 요청을 수행하세요.')
        if self.guided.profile is not None:
            try:
                self.guided.profile.save(self.calibration_file, self.guided_context)
            except OSError as exc:
                self.status.set(f'4방향 보정은 완료했습니다. 파일 저장 실패: {exc} · 현재 창에서는 테스트할 수 있습니다.')
            else:
                self.status.set('정면·좌·우·위 보정값 저장 완료 · 보정 테스트 버튼으로 확인하세요.')
        self.refresh_guided_view()

    def previous_calibration_pose(self):
        if self.guided.back():
            self.tuner.reset_baseline()
            self.status.set('이전 자세로 돌아갔습니다. 자세를 잡고 P 키를 다시 누르세요.')
            self.refresh_guided_view()

    def cancel_calibration(self):
        self.guided.cancel()
        self.tuner.reset_baseline()
        self.p_key_down = False
        self.status.set('보정/테스트를 중지했습니다.')
        self.refresh_guided_view()

    def start_calibration_test(self):
        if self.manual.editing:
            return
        if self.worker is None:
            self.start()
        if self.worker is None or self.worker.stop_event.is_set() or not self.profile_compatible():
            self.status.set('현재 카메라 설정으로 4방향 보정을 먼저 완료하세요.')
            return
        self.guided.start_test()
        self.p_key_down = False
        self.canvas.focus_set()
        self.status.set('보정 테스트 · 요청 방향에서 잠시 유지하면 자동 확인합니다. P 키는 누르지 않아도 됩니다.')
        self.refresh_guided_view()

    def reset_stabilization(self):
        self.invalidate_tracking('좌표/영점 초기화 · 정면 기준을 다시 설정하세요.')
        self.status.set('보정 방식이 바뀌었습니다. 정면에서 영점을 다시 설정하세요.')

    def invalidate_tracking(self, reason, clear_manual=False):
        self.guided.reset()
        self.guided_context = None
        self.p_key_down = False
        if clear_manual:
            self.manual.clear()
            self.finish_manual_editor()
            self.manual_info.set('추적 종료 · 수동 보정 초기화')
        self.landmarks.reset()
        self.tuner.reset_baseline()
        self.last_packet_at = None
        self.last_base_row = None
        self.canvas.delete('stable_point')
        self.canvas.delete('correction')
        self.canvas.delete('manual_point')
        self.point_table.delete(*self.point_table.get_children())
        self.angles.set('정면 기준각도 미설정 · 정면 기준을 새로 설정하세요.')
        self.verdict.set(reason)
        self.filter_info.set(reason)
        for direction in DIRECTIONS:
            values = self.table.item(direction, 'values')
            self.table.item(direction, values=(values[0], '대기', values[2]))
        self.refresh_guided_view()

    def apply_filter_values(self):
        try:
            settings = LandmarkFilterSettings(
                smoothing_seconds=float(self.filter_vars['smoothing_seconds'].get()),
                jump_ratio=float(self.filter_vars['jump_percent'].get())/100,
                min_confidence=float(self.filter_vars['min_confidence'].get())/100,
                max_gap=self.tuner.settings.max_gap)
        except (ValueError, tk.TclError) as exc:
            self.status.set(f'좌표 안정화 입력 오류: {exc}')
            return
        self.landmarks = LandmarkFilter(settings)
        self.reset_stabilization()

    def set_manual_editor_controls(self, editing):
        for widget in (self.calibrate_button, self.stabilize_check, self.filter_button,
                       self.reload_button, self.save_button, self.manual_begin_button):
            widget.configure(state='disabled' if editing else 'normal')
        self.manual_apply_button.configure(state='normal' if editing else 'disabled')
        self.manual_cancel_button.configure(state='normal' if editing else 'disabled')
        self.refresh_guided_view()

    def begin_manual(self):
        if self.manual.editing or self.guided.mode in ('collect', 'test'):
            return
        if (self.latest_packet is None or self.last_base_row is None
                or time.monotonic()-self.latest_packet[-1] > self.tuner.settings.max_gap):
            self.status.set('카메라에서 정면 얼굴의 안정된 5개 점을 먼저 확인하세요.')
            return
        frame = self.latest_packet[0]
        try:
            self.manual.begin(self.last_base_row, (frame.shape[1], frame.shape[0]))
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.edit_frame = frame.copy()
        self.edit_selected = None
        self.edit_tracking_ok = True
        self.guided.reset()
        self.guided_context = None
        self.tuner.reset_baseline()
        self.set_manual_editor_controls(True)
        self.angles.set('수동 위치 조정 중 · 정면 기준각도와 방향 판정은 초기화했습니다.')
        self.verdict.set('수동 위치 조정 중 · 방향 판정 중지')
        self.status.set('노란 점을 눈·코끝·입꼬리에 드래그하세요. 점 선택 후 방향키는 1픽셀씩 이동합니다.')
        height, width = frame.shape[:2]
        scale = min(640/width, 480/height)
        preview = cv2.resize(frame, (round(width*scale), round(height*scale)))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        self.canvas.delete('all')
        self.canvas.create_image((640-preview.shape[1])//2, (480-preview.shape[0])//2,
                                 anchor='nw', image=self.photo)
        for x, y in self.manual.source[4:14].reshape(5, 2):
            x, y = self.manual_to_canvas(x, y)
            self.canvas.create_oval(x-4, y-4, x+4, y+4, outline='#38bdf8', width=2)
        for direction in DIRECTIONS:
            values = self.table.item(direction, 'values')
            self.table.item(direction, values=(values[0], '조정 중', values[2]))
        self.draw_manual_handles()

    def manual_transform(self):
        width, height = self.manual.size
        scale = min(640/width, 480/height)
        return scale, (640-round(width*scale))//2, (480-round(height*scale))//2

    def manual_to_canvas(self, x, y):
        scale, ox, oy = self.manual_transform()
        return x*scale+ox, y*scale+oy

    def update_point_table(self, row, labels=None):
        labels = labels if labels is not None else ManualLandmarks.labels_for(row)
        for index, (x, y) in enumerate(np.asarray(row)[4:14].reshape(5, 2)):
            values = (labels[index], int(x), int(y))
            if self.point_table.exists(str(index)):
                self.point_table.item(str(index), values=values)
            else:
                self.point_table.insert('', 'end', iid=str(index), values=values)

    def draw_manual_handles(self):
        if not self.manual.editing:
            return
        self.canvas.delete('manual_handle')
        self.canvas.delete('manual_text')
        for index, (x, y) in enumerate(self.manual.points):
            cx, cy = self.manual_to_canvas(x, y)
            color = '#fb7185' if self.edit_selected == index else '#facc15'
            self.canvas.create_oval(cx-7, cy-7, cx+7, cy+7, fill=color,
                                    outline='#141d2d', width=2, tags='manual_handle')
            self.canvas.create_text(cx+10, cy-12, text=self.manual.labels()[index],
                                    fill=color, anchor='sw', tags='manual_text')
        row = self.manual.source.copy()
        row[4:14] = self.manual.points.reshape(-1)
        self.update_point_table(row, self.manual.labels())
        if self.edit_tracking_ok:
            self.manual_info.set('영상 정지 중 · 노란 점을 드래그하거나 선택 후 방향키로 조정하세요.')

    def select_manual_point(self, event):
        if not self.manual.editing:
            return
        positions = np.array([self.manual_to_canvas(x, y) for x, y in self.manual.points])
        distances = np.linalg.norm(positions-[event.x, event.y], axis=1)
        self.edit_selected = int(np.argmin(distances)) if distances.min() <= 18 else None
        if self.edit_selected is not None:
            self.canvas.focus_set()
            self.canvas.configure(cursor='fleur')
        self.draw_manual_handles()

    def drag_manual_point(self, event):
        if not self.manual.editing or self.edit_selected is None:
            return
        scale, ox, oy = self.manual_transform()
        self.manual.move(self.edit_selected, (event.x-ox)/scale, (event.y-oy)/scale)
        self.draw_manual_handles()

    def nudge_manual_point(self, event):
        if not self.manual.editing or self.edit_selected is None:
            return
        dx, dy = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}[event.keysym]
        x, y = self.manual.points[self.edit_selected]
        self.manual.move(self.edit_selected, x+dx, y+dy)
        self.draw_manual_handles()
        return 'break'

    def finish_manual_editor(self):
        self.edit_frame = self.edit_selected = None
        self.set_manual_editor_controls(False)
        self.canvas.configure(cursor='')
        self.canvas.delete('manual_handle')
        self.canvas.delete('manual_text')

    def apply_manual(self):
        if not self.manual.editing:
            return
        if (not self.edit_tracking_ok or self.last_packet_at is None
                or time.monotonic()-self.last_packet_at > self.tuner.settings.max_gap):
            self.status.set('조정 중 얼굴 추적이 끊겼습니다. 취소 후 정면에서 다시 조정하세요.')
            return
        try:
            self.manual.commit()
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.finish_manual_editor()
        self.invalidate_tracking('수동 위치 보정 적용 · 정면 기준을 다시 설정하세요.')
        self.manual_info.set('수동 보정 적용 중 · 얼굴 크기에 맞춘 위치 차이만 반영합니다.')
        self.status.set('실시간 영상으로 복귀합니다. 정면 기준을 새로 설정하세요.')

    def cancel_manual(self):
        if not self.manual.editing:
            return
        if self.edit_tracking_ok:
            self.manual.cancel()
        else:
            self.manual.clear()
        self.finish_manual_editor()
        self.invalidate_tracking('수동 조정 취소 · 정면 기준을 다시 설정하세요.')
        self.manual_info.set('기존 수동 보정 유지' if self.manual.active else '수동 보정 없음')

    def reset_manual(self):
        self.manual.clear()
        self.finish_manual_editor()
        self.invalidate_tracking('자동 검출로 초기화 · 정면 기준을 다시 설정하세요.')
        self.manual_info.set('수동 보정 없음 · 자동 검출 위치 사용')

    def track_manual_source(self, packet):
        frame, rows, _, _, _, captured = packet
        size = frame.shape[1], frame.shape[0]
        valid = (len(rows) == 1 and np.asarray(rows).shape == (1, 15)
                 and np.isfinite(rows).all() and rows[0][14] >= self.landmarks.settings.min_confidence
                 and size == self.manual.size and overlap_ratio(self.manual.source[:4], rows[0][:4]) >= .25
                 and (self.last_packet_at is None or 0 < captured-self.last_packet_at <= self.tuner.settings.max_gap))
        self.last_packet_at = captured
        if not valid:
            self.edit_tracking_ok = False
            self.manual_apply_button.configure(state='disabled')
            self.manual_info.set('얼굴 추적 변경 · 적용 불가 · 취소하고 다시 시작하세요.')

    def start(self):
        if self.worker is not None:
            return
        try:
            index = int(self.camera.get())
            if not 0 <= index <= 9:
                raise ValueError
        except ValueError:
            messagebox.showerror('카메라 번호', '0~9 사이의 번호를 입력하세요.', parent=self.root)
            return
        profile = ENROLLMENT_PROFILE if self.resolution.get().startswith('1920') else ACTIVE_PROFILE
        self.tuner = PoseTuner(self.tuner.settings)  # camera/profile changes require a new frontal baseline
        self.landmarks.reset()
        self.manual.clear()
        self.latest_packet = self.last_base_row = None
        self.last_packet_at = None
        self.frames, self.events = Queue(maxsize=1), Queue()
        self.load_guided_profile()
        self.refresh_guided_view()
        self.worker = PoseCamera(index, profile, self.frames, self.events)
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.camera_input.configure(state='disabled')
        self.size_input.configure(state='disabled')
        self.status.set('YuNet 준비 / 카메라 연결 중 · 다른 얼굴 앱의 카메라를 먼저 중지하세요.')
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop_event.set()
            self.stop_button.configure(state='disabled')
        self.invalidate_tracking('카메라 중지 · 좌표/영점 초기화', clear_manual=True)

    def calibrate(self):
        if self.manual.editing or self.guided.mode in ('collect', 'test'):
            return
        try:
            self.tuner.calibrate(time.monotonic())
        except ValueError as exc:
            self.status.set(str(exc))
        else:
            self.status.set('정면 기준 설정 완료 · 방향을 고르고 고개를 움직이며 임계값을 조절하세요.')

    def reload(self):
        if self.manual.editing or self.guided.mode in ('collect', 'test'):
            return
        try:
            self.tuner.settings = load_guided_settings(self.config)
            self.tuner.trial = None
            self.load_values()
            self.status.set('저장된 임계값을 다시 불러왔습니다. 정면 기준은 유지합니다.')
        except RuntimeError as exc:
            self.status.set(str(exc))

    def save(self):
        if self.manual.editing or self.guided.mode in ('collect', 'test'):
            return
        try:
            self.apply_values()
            self.tuner.save(self.config)
        except (ValueError, RuntimeError, OSError, tk.TclError) as exc:
            self.status.set(f'저장 실패: {exc}')
        else:
            self.status.set('임계값 저장 완료 · 실행 중인 등록/인증 앱은 다시 시작해야 반영됩니다.')

    def show(self, packet):
        self.latest_packet = packet
        if self.manual.editing:
            self.track_manual_source(packet)
            return
        frame, rows, raw, reason, native_size, captured = packet
        fps = 1/max(.001, captured-self.last_packet_at) if self.last_packet_at is not None else 0
        self.last_packet_at = captured
        valid = True
        try:
            self.apply_values()
        except (ValueError, tk.TclError) as exc:
            valid = False
            self.verdict.set(f'설정 입력 오류 · 판정 중지: {exc}')
        s = self.tuner.settings
        size = frame.shape[1], frame.shape[0]
        filtered = self.landmarks.update(rows, size, captured)
        if filtered.reset_baseline:
            self.tuner.reset_baseline()
            self.manual.clear()
            self.manual_info.set('추적 변경 · 수동 보정 초기화')
        stable, corrected, adjusted, pose_error = None, None, None, ''
        self.last_base_row = None
        if filtered.row is not None:
            self.last_base_row = filtered.row.copy() if self.use_stabilization.get() else np.asarray(rows[0]).copy()
            try:
                stable = yunet_pose(filtered.row, size)
                adjusted = self.manual.apply(filtered.row, size)
                corrected = yunet_pose(adjusted, size) if self.manual.active else stable
            except (ValueError, cv2.error) as exc:
                pose_error = str(exc)
        selected = corrected if self.use_stabilization.get() else raw
        if self.manual.active and not self.use_stabilization.get() and filtered.row is not None:
            try:
                selected = yunet_pose(self.manual.apply(rows[0], size), size)
            except (ValueError, cv2.error) as exc:
                selected, pose_error = None, str(exc)
        # An excluded/lost frame cannot borrow the previous smoothed coordinates or hold.
        if filtered.row is None or (self.use_stabilization.get() and pose_error):
            selected = None
        self.guided.observe(selected, captured, self.tuner.settings.max_gap)
        if selected is not None and 'front' in self.guided.records:
            self.tuner.origin = tuple(self.guided.records['front'][:2])
        pose = self.tuner.observe(selected, captured)
        confidence = f' · 신뢰도 {int(rows[0][14]*100)}%' if len(rows) == 1 else ''
        face_width = float(rows[0][2]) if len(rows) == 1 else 0
        self.filter_info.set(f'{filtered.reason}{confidence} · 점 이동: 원본 {int(filtered.raw_step*face_width)}픽셀 / '
                             f'자동 보정 {int(filtered.filtered_step*face_width)}픽셀\n'
                             '하늘색 원: 원본 / 초록 십자: 안정화 · '
                             + ('안정화 좌표로 판정' if self.use_stabilization.get() else '원본 좌표로 판정'))
        personal = self.profile_compatible()
        match, _ = self.guided.profile.classify(selected) if personal else (None, None)
        if self.guided.mode == 'collect':
            self.verdict.set(f'보정 {self.guided.index+1}/{len(ORDER)} · {LABELS[self.guided.target]} 자세에서 P 키를 누르세요.'
                             if selected is not None else f'보정 대기 · {pose_error or filtered.reason}')
        elif personal:
            if self.guided.mode == 'test':
                completed = self.guided.target
                if self.guided.test_frame(selected, captured):
                    self.status.set(f'{LABELS[completed]} 테스트 통과 · 다음 요청을 수행하세요.'
                                    if self.guided.mode == 'test' else '보정 테스트 완료 · 4방향 모두 통과했습니다.')
            if self.guided.mode == 'tested':
                self.verdict.set('보정 테스트 완료 · 4/4 통과')
            elif self.guided.mode == 'test':
                self.verdict.set(f'테스트 요청: {LABELS[self.guided.target]} · 현재 판정: '
                                 f'{LABELS[match] if match else "판정 대기"} · {len(self.guided.results)}/{len(ORDER)} 통과')
            else:
                self.verdict.set(f'개인 기준 판정: {LABELS[match] if match else "기준 사이/판정 대기"}')
        elif valid:
            angle_ok, held = self.tuner.check(self.target.get(), pose, captured)
            trial = self.tuner.trial
            held_for = captured-trial.hold_started if trial and trial.hold_started is not None else 0
            self.verdict.set((pose_error or filtered.reason) if selected is None else
                ('정면을 보고 기준을 설정하세요.' if pose is None else
                '통과 · 유지 시간/연속 프레임 충족' if held else
                f'각도 통과 · {held_for:.2f}/{s.hold_seconds:g}초 · {trial.frames}/{s.min_frames}프레임'
                if angle_ok else '요청 방향의 각도 조건 미달'))
        lines = []
        if raw:
            lines.append(f'원본 각도: 좌우각도 {angle_text(raw.yaw)} / 상하각도 {angle_text(raw.pitch)} / 고개 기울임 {angle_text(raw.roll)}')
        if stable:
            automatic = f'자동 보정: 좌우 {angle_text(stable.yaw)} / 상하 {angle_text(stable.pitch)}'
            if corrected and self.manual.active:
                automatic += f' | 수동 보정: 좌우 {angle_text(corrected.yaw)} / 상하 {angle_text(corrected.pitch)}'
            lines.append(automatic)
        if self.tuner.origin is not None:
            baseline = f'정면 기준각도: 좌우 {angle_text(self.tuner.origin[0])} / 상하 {angle_text(self.tuner.origin[1])}'
            if pose:
                baseline += f' | 정면 보정: 좌우 {angle_text(pose.yaw*s.yaw_sign)} / 상하 {angle_text(pose.pitch*s.pitch_sign)}'
            lines.append(baseline)
        else:
            lines.append('정면 기준각도 미설정 · 정면 기준 설정을 눌러주세요.')
        self.angles.set('\n'.join(lines) if lines else reason or '얼굴 대기 중')
        width, height = frame.shape[1], frame.shape[0]
        self.capture_info.set(f'처리 {width}×{height} · 실제 입력 {native_size[0]}×{native_size[1]} · {int(fps)} FPS · 각도/좌표는 정수 표시')
        bounds = {
            'front': f'|좌우| ≤ {s.front_degrees:g}°, |상하| ≤ {s.front_degrees:g}°',
            'left': f'좌우 −{s.yaw_max:g}~−{s.yaw_min:g}°, |상하| ≤ {s.cross_degrees:g}°',
            'right': f'좌우 +{s.yaw_min:g}~+{s.yaw_max:g}°, |상하| ≤ {s.cross_degrees:g}°',
            'up': f'상하 −{s.pitch_max:g}~−{s.pitch_min:g}°, |좌우| ≤ {s.cross_degrees:g}°',
        }
        for direction in DIRECTIONS:
            if personal:
                center = self.guided.profile.centers[ORDER.index(direction)]
                bounds[direction] = f'P 기록 기준 · 좌우 {angle_text(center[0])} / 상하 {angle_text(center[1])}'
                verdict = '일치' if match == direction else '불일치' if selected is not None else '대기'
            elif self.guided.mode == 'collect':
                verdict = '기록 대기'
            else:
                verdict = '통과' if valid and pose and direction_matches(direction, pose, s) else '미달' if valid and pose else '대기'
            self.table.item(direction, values=(LABELS[direction], verdict, bounds[direction]))
        self.refresh_guided_view()
        scale = min(640/width, 480/height)
        preview = cv2.resize(frame, (round(width*scale), round(height*scale)))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        self.canvas.delete('all')
        ox, oy = (640-preview.shape[1])//2, (480-preview.shape[0])//2
        self.canvas.create_image(ox, oy, anchor='nw', image=self.photo)
        for row in rows:
            x, y, w, h = row[:4]
            self.canvas.create_rectangle(x*scale+ox, y*scale+oy, (x+w)*scale+ox, (y+h)*scale+oy, outline='#facc15', width=2)
            if self.show_raw_points.get():
                for x, y in row[4:14].reshape(5, 2):
                    x, y = x*scale+ox, y*scale+oy
                    self.canvas.create_oval(x-4, y-4, x+4, y+4, outline='#38bdf8', width=2, tags='raw_point')
        if filtered.row is not None:
            for index, (x, y) in enumerate(filtered.row[4:14].reshape(5, 2)):
                x, y = x*scale+ox, y*scale+oy
                if self.show_raw_points.get():
                    rx, ry = rows[0][4+index*2:6+index*2]
                    self.canvas.create_line(rx*scale+ox, ry*scale+oy, x, y, fill='#64748b', tags='correction')
                self.canvas.create_line(x-5, y, x+5, y, fill='#4ade80', width=2, tags='stable_point')
                self.canvas.create_line(x, y-5, x, y+5, fill='#4ade80', width=2, tags='stable_point')
            point_row = adjusted if adjusted is not None else filtered.row
            self.update_point_table(point_row)
            if self.manual.active and adjusted is not None:
                for x, y in adjusted[4:14].reshape(5, 2):
                    x, y = x*scale+ox, y*scale+oy
                    self.canvas.create_oval(x-5, y-5, x+5, y+5, outline='#facc15', width=2, tags='manual_point')
        requested = self.guided.target if self.guided.mode in ('collect', 'test') else self.target.get() if not personal else None
        banner = (f'요청: {LABELS[requested]}' + (' · P 키로 기록' if self.guided.mode == 'collect' else '')
                  if requested else '4방향 테스트 완료' if self.guided.mode == 'tested' else '보정 완료 · 테스트 버튼으로 확인')
        self.canvas.create_text(320, 25, text=banner, fill='white', font=('맑은 고딕', 16, 'bold'))
        vectors = {'up': (320, 105, 320, 55),
                   'left': (345, 80, 295, 80), 'right': (295, 80, 345, 80)}
        if requested in vectors:
            self.canvas.create_line(*vectors[requested], fill='#facc15', width=5,
                                    arrow=tk.LAST, arrowshape=(14, 18, 8))

    def poll(self):
        if self.closing:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind in ('status', 'error'):
                    self.status.set(value)
                elif kind == 'stopped':
                    self.worker = None
                    self.invalidate_tracking('카메라 중지 · 좌표/영점 초기화', clear_manual=True)
                    self.start_button.configure(state='normal')
                    self.stop_button.configure(state='disabled')
                    self.camera_input.configure(state='normal')
                    self.size_input.configure(state='readonly')
        except Empty:
            pass
        try:
            if self.worker and not self.worker.stop_event.is_set():
                self.show(self.frames.get_nowait())
        except Empty:
            pass
        if (self.worker and self.last_packet_at is not None
                and time.monotonic()-self.last_packet_at > self.tuner.settings.max_gap):
            if self.manual.editing:
                self.edit_tracking_ok = False
                self.manual_apply_button.configure(state='disabled')
                self.manual_info.set('영상 입력 지연 · 적용 불가 · 취소하고 다시 시작하세요.')
            else:
                self.invalidate_tracking('영상 입력 지연 · 판정 중지/영점 초기화', clear_manual=True)
        self.root.after(40, self.poll)

    def close(self):
        self.closing = True
        self.stop()
        self._finish_close()

    def _finish_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(50, self._finish_close)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='YuNet 상하좌우 각도 및 유지 조건 수동 테스트')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    root = tk.Tk()
    try:
        PoseTestApp(root, camera=args.camera, config=args.config)
    except (RuntimeError, OSError, ValueError) as exc:
        messagebox.showerror('방향 테스트 준비 오류', str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == '__main__':
    main()
