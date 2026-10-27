"""웹캠 대기/접근 전환 테스트. 영상은 저장하지 않습니다."""
import importlib.util
from pathlib import Path
from collections import deque
import time
import tkinter as tk
from tkinter import messagebox, ttk

spec = importlib.util.spec_from_file_location('standby', Path(__file__).with_name('1-standby.py'))
standby = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standby)


class Preview:
    def __init__(self, root):
        self.root = root
        self.camera = None
        self.gate_settings = standby.GateSettings()
        self.ready = False
        self.consecutive = 0
        self.scanner = standby.FaceWatcher(self.gate_settings)
        self.started = time.monotonic()
        self.previous_frame = None
        self.frame_times = deque(maxlen=30)
        self.work_times = deque(maxlen=30)
        self.root.title('Stage 1 - 웹캠 대기 / 카메라 준비 테스트')
        self.root.configure(bg='#17202b')
        self.root.resizable(False, False)
        tk.Label(root, text='STAGE 1  |  카메라 대기 테스트', font=('Malgun Gothic', 18, 'bold'),
                 bg='#17202b', fg='white').pack(padx=24, pady=(18, 8))
        controls = tk.Frame(root, bg='#17202b')
        controls.pack(padx=24, pady=8)
        tk.Label(controls, text='목표 FPS', bg='#17202b', fg='white').pack(side='left', padx=6)
        self.fps_var = tk.StringVar(value=str(standby.CHECK_FPS))
        tk.Spinbox(controls, from_=1, to=60, textvariable=self.fps_var, width=5).pack(side='left')
        tk.Label(controls, text='검출 해상도', bg='#17202b', fg='white').pack(side='left', padx=6)
        self.size_var = tk.StringVar(value=f'{standby.DETECT_SIZE[0]}x{standby.DETECT_SIZE[1]}')
        ttk.Combobox(controls, textvariable=self.size_var, width=13,
                     values=['320x180', '426x240', '480x270', '640x360', '800x450', '960x540', '1280x720']).pack(side='left')
        tk.Button(controls, text='적용 / 다시 대기', command=self.apply_settings).pack(side='left', padx=8)
        tk.Button(controls, text='ver1 복원', command=self.restore_ver1).pack(side='left', padx=4)
        tuning = tk.Frame(root, bg='#17202b')
        tuning.pack(padx=24, pady=4)
        defaults = self.gate_settings
        self.gate_vars = {}
        fields = [
            ('min_face_ratio', '최소 얼굴 너비 (%)', 0, 50, 1, defaults.min_face_ratio*100),
            ('front_tolerance', '정면 허용오차 (%)', 5, 100, 1, defaults.front_tolerance*100),
            ('center_range', '중앙 허용영역 (%)', 10, 100, 1, defaults.center_range*100),
            ('min_confidence', '최소 신뢰도 (%)', 50, 99, 1, defaults.min_confidence*100),
            ('hold_seconds', '유지 시간 (초)', 0, 2, 0.1, defaults.hold_seconds),
        ]
        for key, label, low, high, step, initial in fields:
            variable = tk.DoubleVar(value=initial)
            self.gate_vars[key] = variable
            tk.Scale(tuning, label=label, variable=variable, from_=low, to=high,
                     resolution=step, orient='horizontal', length=155,
                     bg='#17202b', fg='white', troughcolor='#334155', highlightthickness=0).pack(side='left', padx=4)
        tk.Label(root, text='값을 바꾼 뒤 [적용 / 다시 대기] 클릭 | 최소 크기·신뢰도·유지 시간: 높일수록 엄격 / 정면 오차·중앙 영역: 낮출수록 엄격',
                 bg='#17202b', fg='#cbd5e1', font=('Malgun Gothic', 9)).pack()
        self.continuous_var = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text='계속 검출하여 성능 비교 (해제하면 준비 완료 상태 유지)',
                       variable=self.continuous_var, bg='#17202b', fg='white',
                       selectcolor='#263341', activebackground='#17202b', activeforeground='white').pack()
        self.capture_info = tk.Label(root, text='설정 적용 시 카메라에도 요청합니다. 실제 수신 규격은 아래 표시됩니다.',
                                     bg='#17202b', fg='#cbd5e1', font=('Malgun Gothic', 9))
        self.capture_info.pack(pady=4)
        self.status = tk.Label(root, font=('Malgun Gothic', 13), bg='#17202b', fg='white')
        self.status.pack()
        self.led = tk.Canvas(root, width=48, height=48, bg='#17202b', highlightthickness=0)
        self.led.pack(pady=6)
        self.bulb = self.led.create_oval(7, 7, 41, 41, fill='white', outline='#64748b')
        panel = tk.Frame(root, bg='#17202b')
        panel.pack(padx=24)
        self.video = tk.Label(panel, bg='black', width=standby.DETECT_SIZE[0], height=standby.DETECT_SIZE[1])
        self.video.pack(side='left')
        self.crop_video = tk.Label(panel, text='얼굴 crop 대기 중', bg='#263341', fg='white', width=32, height=20)
        self.crop_video.pack(side='left', padx=(12, 0))
        self.info = tk.Label(root, text=f'{standby.DETECT_SIZE[0]}×{standby.DETECT_SIZE[1]} 전체 화면 / 목표 {standby.CHECK_FPS}fps | 조절한 조건으로 촬영 준비 판단',
                             bg='#17202b', fg='#cbd5e1', font=('Malgun Gothic', 10))
        self.info.pack(pady=8)
        tk.Label(root, text='흰색: 1초 켜짐 / 1초 꺼짐   ·   노란색: 준비 완료\n왼쪽: 전체 화면 / 오른쪽: 얼굴 주변 crop. 하늘색: 눈 / 분홍색: 코 / 초록색: 입. 특징점은 모델 추정이며 시선 판별은 아닙니다.',
                 bg='#17202b', fg='#cbd5e1', font=('Malgun Gothic', 10)).pack()
        buttons = tk.Frame(root, bg='#17202b')
        buttons.pack(pady=16)
        tk.Button(buttons, text='다시 대기', command=self.reset, width=16).pack(side='left', padx=8)
        tk.Button(buttons, text='종료', command=self.close, width=16).pack(side='left', padx=8)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Escape>', lambda event: self.close())
        self.reset()
        try:
            standby._detector()
            self.camera = standby.open_usb_camera()
            self.requested_size = standby.DETECT_SIZE
            self.requested_fps = standby.CHECK_FPS
        except Exception as exc:
            messagebox.showerror('카메라 준비 실패', str(exc), parent=root)
            self.close()
            return
        root.after(0, self.update)
        root.after(0, self.update_led)

    def reset(self):
        self.ready = False
        self.consecutive = 0
        self.scanner = standby.FaceWatcher(self.gate_settings)
        self.started = time.monotonic()
        self.previous_frame = None
        self.frame_times.clear()
        self.work_times.clear()
        self.status.config(text='대기 중 - 카메라를 바라봐 주세요', fg='white')

    @staticmethod
    def parse_settings(fps_text, size_text):
        fps = int(fps_text)
        width, height = map(int, size_text.lower().replace('×', 'x').split('x'))
        if not 1 <= fps <= 60 or not 160 <= width <= 1920 or not 90 <= height <= 1080:
            raise ValueError('FPS는 1~60, 해상도는 가로 160~1920 / 세로 90~1080 범위로 입력하세요.')
        return fps, (width, height)

    def restore_ver1(self):
        try:
            preset = standby.load_ver1()
        except RuntimeError as exc:
            messagebox.showerror('ver1 설정 오류', str(exc), parent=self.root)
            return
        self.fps_var.set(str(preset['fps']))
        self.size_var.set(f"{preset['resolution'][0]}x{preset['resolution'][1]}")
        for key, variable in self.gate_vars.items():
            variable.set(preset[key] if key == 'hold_seconds' else preset[key]*100)
        self.apply_settings()

    def apply_settings(self):
        try:
            fps, size = self.parse_settings(self.fps_var.get(), self.size_var.get())
            gate_values = {key: variable.get() for key, variable in self.gate_vars.items()}
            gate = standby.GateSettings(**{key: value if key == 'hold_seconds' else value/100
                                           for key, value in gate_values.items()})
        except (ValueError, TypeError):
            messagebox.showerror('입력 확인', 'FPS: 1~60 / 해상도: 160x90~1920x1080\n예: 10, 480x270', parent=self.root)
            return
        if self.camera is None:
            return
        self.gate_settings = gate
        standby.CHECK_FPS, standby.DETECT_SIZE = fps, size
        if getattr(self, 'requested_size', None) != size:
            self.camera.set(standby.cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.camera.set(standby.cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.requested_size = size
        if getattr(self, 'requested_fps', None) != fps:
            self.camera.set(standby.cv2.CAP_PROP_FPS, fps)
            self.requested_fps = fps
        self.reset()
        self.info.config(text=f'{size[0]}×{size[1]} / 목표 {fps}fps / 측정 중...')

    def update_led(self):
        color = '#facc15' if self.ready else ('white' if int(time.monotonic() - self.started) % 2 == 0 else '#263341')
        self.led.itemconfig(self.bulb, fill=color)
        self.root.after(50, self.update_led)

    def update(self):
        started = time.monotonic()
        try:
            ok, frame = self.camera.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError('카메라 영상 수신이 끊겼습니다.')
            if self.scanner.shape is not None and self.scanner.shape != frame.shape[:2]:
                self.reset()
            if self.continuous_var.get() and self.scanner.ready:
                self.scanner.ready = False
            self.scanner.update(frame)
            self.ready = self.scanner.ready
            display = frame.copy()
            frame_h, frame_w = frame.shape[:2]
            area = self.gate_settings.center_range
            standby.cv2.rectangle(display,
                                  (round(frame_w*(1-area)/2), round(frame_h*(1-area)/2)),
                                  (round(frame_w*(1+area)/2)-1, round(frame_h*(1+area)/2)-1),
                                  (130, 100, 70), 1)
            if self.scanner.roi is not None:
                x, y, w, h = self.scanner.roi
                crop = frame[y:y+h, x:x+w].copy()
                for index, (px, py) in enumerate(self.scanner.landmarks):
                    color = (255, 255, 0) if index < 2 else ((255, 0, 255) if index == 2 else (0, 255, 0))
                    standby.cv2.circle(display, (px, py), 3, color, -1)
                    standby.cv2.circle(crop, (px-x, py-y), 3, color, -1)
                scale = min(280 / w, 280 / h)
                crop = standby.cv2.resize(crop, (max(1, round(w*scale)), max(1, round(h*scale))))
                self.crop_image = self.photo(crop)
                self.crop_video.config(image=self.crop_image, text='', width=280, height=300)
                color = (0, 255, 255) if self.ready else (0, 165, 255)
                standby.cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            else:
                self.crop_video.config(image='', text='얼굴 crop 대기 중', width=32, height=20)
            if self.ready:
                self.status.config(text='조건 통과 - 다음 단계 전달 가능', fg='#facc15')
            elif self.scanner.face is not None:
                self.status.config(text=self.scanner.reason, fg='#fb923c')
            else:
                self.status.config(text=self.scanner.reason, fg='white')
            if self.previous_frame is not None:
                self.frame_times.append(started - self.previous_frame)
            if self.frame_times:
                fps = len(self.frame_times) / sum(self.frame_times)
                work_ms = 1000 * sum(self.work_times) / len(self.work_times) if self.work_times else 0
                mode = '연속 검출' if self.continuous_var.get() else ('준비 완료 유지' if self.ready else '검출 중')
                self.info.config(text=f'{standby.DETECT_SIZE[0]}×{standby.DETECT_SIZE[1]} / 목표 {standby.CHECK_FPS}fps / 실제 평균 {fps:.1f}fps / 처리 {work_ms:.1f}ms | {mode}\n얼굴 너비 {self.scanner.face_ratio*100:.1f}% / 검출 신뢰도 {self.scanner.score*100:.1f}% / 정면 오차 {self.scanner.front_error*100:.1f}% / 유지 {self.scanner.hold_elapsed:.2f}초')
            reported_fps = self.camera.get(standby.cv2.CAP_PROP_FPS)
            self.capture_info.config(text=f'실제 수신: {frame.shape[1]}×{frame.shape[0]} / 카메라 보고 {reported_fps:.1f}fps | 화면은 검출 해상도로 표시')
            self.previous_frame = started
            # 큰 해상도도 화면 안에서 비교할 수 있도록 표시 크기만 제한한다.
            scale = min(1, 800 / standby.DETECT_SIZE[0], 450 / standby.DETECT_SIZE[1])
            display_size = tuple(max(1, round(value * scale)) for value in standby.DETECT_SIZE)
            display = standby.cv2.resize(display, display_size)
            self.image = self.photo(display)
            self.video.config(image=self.image, width=display_size[0], height=display_size[1])
        except Exception as exc:
            messagebox.showerror('카메라 오류', str(exc), parent=self.root)
            self.close()
            return
        elapsed = time.monotonic() - started
        self.work_times.append(elapsed)
        self.root.after(max(1, round((1 / standby.CHECK_FPS - elapsed) * 1000)), self.update)

    @staticmethod
    def photo(frame):
        frame = standby.cv2.flip(frame, 1)  # 화면과 얼굴 crop을 거울처럼 표시
        rgb = standby.cv2.cvtColor(frame, standby.cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        return tk.PhotoImage(data=f'P6 {width} {height} 255\n'.encode() + rgb.tobytes(), format='PPM')

    def close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    Preview(root)
    root.mainloop()
