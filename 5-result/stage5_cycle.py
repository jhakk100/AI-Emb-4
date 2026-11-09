"""임시통합 app.py의 door_cycle.DoorCycle 자리에 들어가는 5단계 연결부.

인증 시도가 생기면 5-result.py의 ResultHandler.handle()을 별도 스레드에서 실행하고(문 열림 3초·버튼 대기
동안 화면이 멈추지 않게), 화면은 phase / led_state() / advance()로 진행 상황을 읽는다.
LED·버튼은 실제 GPIO 장치가 있으면 같이 쓰고, 없으면(노트북) 화면 LED·화면 버튼으로만 동작한다.
"""

import importlib.util
from pathlib import Path
import sys
import threading
import time


def load_stage5():
    if 'stage5_result' in sys.modules:
        return sys.modules['stage5_result']
    path = Path(__file__).with_name('5-result.py')
    spec = importlib.util.spec_from_file_location('stage5_result', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['stage5_result'] = module
    spec.loader.exec_module(module)
    return module


stage5 = load_stage5()


class VirtualLED:
    """5단계가 켜는 LED 색을 기억해 화면에 표시. 실제 RGBLED가 있으면 같은 명령을 그대로 전달."""
    def __init__(self, real=None):
        self.real = real
        self._lock = threading.Lock()
        self._state = ('off', None)

    @property
    def color(self):
        return self.describe()[0]

    @color.setter
    def color(self, value):
        with self._lock:
            self._state = ('solid', tuple(float(v) for v in value))
        if self.real is not None:
            self.real.color = value

    def off(self):
        with self._lock:
            self._state = ('off', None)
        if self.real is not None:
            self.real.off()

    def blink(self, on_time=1, off_time=1, on_color=(1, 1, 1), off_color=(0, 0, 0), background=True, **_):
        with self._lock:
            self._state = ('blink', tuple(on_color))
        if self.real is not None:
            self.real.blink(on_time=on_time, off_time=off_time, on_color=on_color,
                            off_color=off_color, background=background)

    def describe(self):
        """(RGB 0~1, 이름) - 이름: off / green / red / red_blink / other"""
        with self._lock:
            kind, value = self._state
        if kind == 'off':
            return (0, 0, 0), 'off'
        key = tuple(round(v) for v in value)
        if kind == 'blink':
            return value, 'red_blink' if key == (1, 0, 0) else 'other'
        return value, {(0, 1, 0): 'green', (1, 0, 0): 'red'}.get(key, 'other')

    def close(self):
        if self.real is not None:
            self.real.close()


class PanelButton:
    """화면의 '수동 개방 버튼'. 실제 버튼이 있으면 둘 중 어느 쪽을 눌러도 같은 처리."""
    def __init__(self, real=None):
        self.real = real
        self._callback = None

    @property
    def when_pressed(self):
        return self._callback

    @when_pressed.setter
    def when_pressed(self, callback):
        self._callback = callback
        if self.real is not None:
            self.real.when_pressed = callback

    @property
    def waiting(self):
        return self._callback is not None

    def press(self):
        callback = self._callback
        if callback is None:      # 5단계 수동 처리 대기 중이 아니면 무시(문이 열리지 않음)
            return False
        callback()
        return True

    def close(self):
        self._callback = None
        if self.real is not None:
            self.real.close()


class Stage5Cycle:
    """DoorCycle과 같은 사용법(phase / completed / mode / reset / advance)에 5단계 처리를 연결."""
    def __init__(self, mode='continuous', use_gpio=True, enable_voice=None):
        settings = dict(stage5.SET5)
        if enable_voice is not None:
            settings['enable_voice'] = enable_voice
        led, buzzer, button = stage5.make_devices(settings, enabled=use_gpio)
        self.led, self.button = VirtualLED(led), PanelButton(button)
        self.handler = stage5.ResultHandler(self.led, buzzer, self.button, settings=settings)
        self.settings = settings
        self.mode = mode
        self.last_action = None
        self.last_result = None
        self._worker = None
        self._outcome = {}
        self.manual_started = None
        self.reset()

    def reset(self):
        # 화면 쪽 상태만 초기화. 진행 중인 5단계 처리(문 열림·잠금)는 스레드에서 끝까지 마친다.
        self.phase = 'waiting'
        self.completed = False

    def start(self, now, name, score, frame=None, auth_ms=None):
        """4단계 결과 (이름, 점수)로 5단계 시작. 이미 처리 중이면 False."""
        if self.phase != 'waiting' or self.completed or (self._worker and self._worker.is_alive()):
            return False
        auth = stage5.from_stage4((name, score), frame=frame, auth_ms=auth_ms)
        self.last_result = auth
        self._outcome = {}
        self.manual_started = None

        def work():
            try:
                self._outcome['action'] = self.handler.handle(auth)
            except BaseException as exc:  # 5단계 오류도 화면에 전달
                self._outcome['error'] = exc

        self.phase = 'processing'
        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        return True

    def led_state(self):
        """처리 중 화면 표시용: (LED 색 이름, 깜빡임 여부, 문구)"""
        _, name = self.led.describe()
        if name == 'green':
            return 'green', False, '5단계 · 문 열림'
        if name == 'red':
            return 'red', False, f'5단계 · 인증 실패 {self.handler.fail_count}/{self.settings["max_retries"]} · 다시 시도'
        if name == 'red_blink':
            if self.manual_started is None:
                self.manual_started = time.monotonic()
            left = max(0, self.settings['manual_timeout'] - (time.monotonic() - self.manual_started))
            return 'red', True, f'5단계 · 3회 실패 · 수동 개방 버튼 대기 {left:.0f}초'
        return 'yellow', False, '5단계 · 결과 처리 중'

    def advance(self, now):
        """처리가 끝났으면 'retry' / 'reset' / 'finished' 중 하나, 아직이면 None."""
        if self.phase != 'processing' or (self._worker and self._worker.is_alive()):
            return None
        self.phase = 'waiting'
        if 'error' in self._outcome:
            raise RuntimeError(f'5단계 처리 오류: {self._outcome["error"]}')
        self.last_action = self._outcome.get('action')
        if self.last_action == stage5.Action.RETRY:
            return 'retry'
        self.handler.reset_session()
        self.completed = self.mode == 'once'
        return 'finished' if self.completed else 'reset'

    def press_manual(self):
        return self.button.press()

    def close(self):
        self.handler.close()
