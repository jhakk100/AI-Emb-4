# Raspberry Pi 5 8GB / Windows: set5.json 설정 사용. 출입 시도마다 결과(OPENED 등) 한 줄 출력.
# 필수 모듈: 없음. OpenCV(cv2)는 결과 사진 저장용, gpiozero RGBLED·Buzzer·Button은 선택 사항.
# from_stage4((이름 또는 None, 점수 또는 None), frame) -> AuthResult, handle(AuthResult) -> Action.
# 성공: 초록 지속. 실패: 빨강 + 부저. 3회 실패: 빨강 깜빡임 + 버튼 대기. (대기 흰색·준비 노랑은 1단계)

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import csv
import json
import math
import numbers
import queue
import re
import subprocess
import sys
import threading
import time
try:
    import cv2
except ModuleNotFoundError as exc:
    if exc.name != 'cv2':
        raise
    cv2 = None

PRESET_PATH = Path(__file__).with_name('set5.json')


class AuthStatus(Enum):
    SUCCESS = 'SUCCESS'
    FAIL = 'FAIL'


class Action(Enum):
    OPENED = 'OPENED'            # 인증 성공, 문 열림
    RETRY = 'RETRY'              # 인증 실패, 1단계 대기부터 다시
    MANUAL_OPEN = 'MANUAL_OPEN'  # 재시도 초과 후 버튼으로 수동 개방
    LOCKED = 'LOCKED'            # 재시도 초과 후 버튼 대기 시간 초과, 잠금 유지


@dataclass
class AuthResult:
    """4단계 결과. frame(결과 사진)·auth_ms(인증 처리 시간)는 부르는 쪽에서 채움(선택)."""
    status: AuthStatus
    user_id: str = 'unknown'
    score: float = -1.0
    frame: object = None
    auth_ms: float = None
    message: str = ''      # 로그 note에 기록할 메모(선택)


def from_stage4(result, frame=None, auth_ms=None):
    # 4단계 authenticate_face() 반환값 (이름 또는 None, 점수 또는 None) -> AuthResult
    # 이름이 있으면 성공, None이면 실패(기준값 미달 또는 등록자 없음). 판정 자체는 4단계 몫.
    # 등록된 사람이 없으면 4단계가 (None, None)을 돌려줌 -> 실패, 점수 -1.0
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError('4단계 결과는 (이름 또는 None, 점수 또는 None) 형식이어야 합니다.')
    name, score = result
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise RuntimeError('4단계 결과의 이름은 문자열 또는 None이어야 합니다.')
    if score is not None and (isinstance(score, bool) or not isinstance(score, numbers.Real)
                              or not math.isfinite(score)):
        raise RuntimeError('4단계 결과의 점수는 숫자 또는 None이어야 합니다.')
    if name is not None and score is None:
        raise RuntimeError('4단계가 성공(이름)을 줬는데 점수가 없습니다.')
    if score is None:
        message = 'no registered user'
    elif name is None:
        message = 'below threshold'
    else:
        message = ''
    return AuthResult(AuthStatus.SUCCESS if name is not None else AuthStatus.FAIL,
                      user_id=name if name is not None else 'unknown',
                      score=-1.0 if score is None else float(score),
                      frame=frame, auth_ms=auth_ms, message=message)


def load_set5(path=PRESET_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise ValueError('설정은 JSON 객체여야 합니다.')
        result = {}
        pins = ('pin_led_red', 'pin_led_green', 'pin_led_blue', 'pin_buzzer', 'pin_button')
        for key in pins:
            value = data[key]
            if type(value) is not int or not 0 <= value <= 27:
                raise ValueError(f'{key}는 BCM 핀 번호 0~27 정수여야 합니다.')
            result[key] = value
        if len({result[key] for key in pins}) != len(pins):
            raise ValueError('핀 번호가 서로 겹칩니다.')
        integers = {'max_retries': (1, 10), 'alarm_beeps': (1, 10), 'max_snapshots': (0, 100000)}
        for key, (low, high) in integers.items():
            value = data[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key}는 {low}~{high} 정수여야 합니다.')
            result[key] = value
        ranges = {'unlock_seconds': (0.5, 30), 'session_timeout': (5, 600),
                  'manual_timeout': (1, 120), 'fail_beep_seconds': (0.05, 3)}
        for key, (low, high) in ranges.items():
            value = data[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not low <= value <= high):
                raise ValueError(f'{key}는 {low}~{high} 범위의 숫자여야 합니다.')
            result[key] = float(value)
        for key in ('led_active_high', 'button_pull_up', 'enable_voice',
                    'snapshot_on_success', 'snapshot_on_fail'):
            if type(data[key]) is not bool:
                raise ValueError(f'{key}는 true/false여야 합니다.')
            result[key] = data[key]
        if data['buzzer_type'] not in ('active', 'passive'):
            raise ValueError('buzzer_type은 "active" 또는 "passive"여야 합니다.')
        result['buzzer_type'] = data['buzzer_type']
        if not isinstance(data['log_path'], str) or not data['log_path']:
            raise ValueError('log_path는 비어 있지 않은 문자열이어야 합니다.')
        result['log_path'] = Path(__file__).resolve().parent / data['log_path']  # 상대 경로는 이 폴더 기준
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'set5 설정을 읽을 수 없습니다: {path} ({exc})') from exc


try:
    SET5 = load_set5()
except RuntimeError as exc:
    if __name__ == '__main__':
        print(f'안내: {exc}', file=sys.stderr)
        raise SystemExit(1)
    raise

LOG_FIELDS = ['time', 'event', 'user_id', 'score', 'retry', 'auth_ms', 'snapshot', 'note']
SOUND_DIR = Path(__file__).with_name('sounds')  # <키>.wav가 있으면 espeak-ng 대신 재생
MESSAGES = {
    'success': '인증되었습니다. 출입하세요.',
    'retry': '인증에 실패했습니다. 다시 시도해 주세요.',
    'manual': '인증에 실패했습니다. 버튼을 눌러 수동 처리해 주세요.',
    'locked': '출입이 제한되었습니다.',
}


def make_devices(settings=None, enabled=True):
    # gpiozero가 없거나(Windows 등) enabled=False(--no-gpio)면 (None, None, None).
    # 핀·극성·부저 종류는 set5.json 기준. 돌려준 led는 1단계 prepare_camera(led=led)에도 그대로 넘김.
    settings = settings or SET5
    if not enabled:
        return None, None, None
    try:
        from gpiozero import RGBLED, Buzzer, TonalBuzzer, Button
    except ModuleNotFoundError:
        return None, None, None
    created = []
    try:
        led = RGBLED(settings['pin_led_red'], settings['pin_led_green'], settings['pin_led_blue'],
                     active_high=settings['led_active_high'])
        created.append(led)
        buzzer = (TonalBuzzer(settings['pin_buzzer']) if settings['buzzer_type'] == 'passive'
                  else Buzzer(settings['pin_buzzer']))
        created.append(buzzer)
        button = Button(settings['pin_button'], pull_up=settings['button_pull_up'], bounce_time=0.05)
    except Exception as exc:
        for device in reversed(created):  # 일부만 만들어졌으면 먼저 정리
            device.close()
        raise RuntimeError(f'GPIO 장치를 열 수 없습니다. 핀 번호와 배선을 확인하세요. ({exc})') from exc
    return led, buzzer, button


class KeyboardButton:
    """버튼이 없을 때(Windows 테스트) 엔터를 버튼으로 사용. 입력이 끝나면(EOF) 절대 눌림 아님.
    gpiozero Button처럼 when_pressed에 함수를 넣으면 엔터를 칠 때 호출된다."""
    def __init__(self):
        self.when_pressed = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for _ in sys.stdin:       # 엔터 한 번 = 한 줄
            callback = self.when_pressed
            if callback is not None:  # 수동 처리 대기 중이 아닐 때 친 엔터는 무시
                callback()

    def close(self):
        self.when_pressed = None


class Speaker:
    """음성을 하나씩 순서대로 재생(겹침 방지). 스피커/TTS가 없어도 처리는 계속된다."""
    def __init__(self, enabled):
        self.enabled = enabled and sys.platform != 'win32'
        self._queue = queue.Queue()
        self._warned = False
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def say(self, key):
        if self.enabled:
            self._queue.put(key)

    def _worker(self):
        while True:
            key = self._queue.get()
            if key is None:  # close() 신호
                return
            wav = SOUND_DIR / f'{key}.wav'
            command = (['aplay', '-q', str(wav)] if wav.is_file()
                       else ['espeak-ng', '-v', 'ko', MESSAGES[key]])
            try:
                subprocess.run(command, timeout=15, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=True)
            except (OSError, subprocess.SubprocessError) as exc:
                if not self._warned:  # 같은 안내는 한 번만
                    self._warned = True
                    print(f'안내: 음성 출력 실패 ({exc})', file=sys.stderr)

    def close(self):
        # 남은 안내까지 재생한 뒤 종료(각 재생 최대 15초).
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None


class ResultHandler:
    """인증 결과에 따라 LED·부저·음성·버튼·보안 로그·사진을 처리."""
    def __init__(self, led=None, buzzer=None, button=None, settings=None, keyboard_button=False):
        self.settings = settings or SET5
        self.led, self.buzzer = led, buzzer
        # 실제 버튼이 없으면 기본은 '버튼 없음'(= 수동 개방 불가). 엔터 대체는 테스트에서만 명시적으로 켬.
        self.button = button if button is not None else (KeyboardButton() if keyboard_button else None)
        self.voice = Speaker(self.settings['enable_voice'])
        self.fail_count = 0
        self.last_fail = None
        self._log_lock = threading.Lock()
        self.snapshot_dir = self.settings['log_path'].parent / 'snapshots'
        log_path = self.settings['log_path']
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():  # utf-8-sig: 엑셀에서 한글 깨짐 방지
            with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(LOG_FIELDS)

    def reset_session(self):
        # 사람이 떠났을 때(초음파 등) 직접 호출해도 됨. 연속 실패 횟수 초기화.
        self.fail_count = 0
        self.last_fail = None

    def handle(self, result):
        if not isinstance(result, AuthResult) or not isinstance(result.status, AuthStatus):
            raise RuntimeError('4단계 결과는 AuthResult(status=AuthStatus.…)여야 합니다.')
        if (self.last_fail is not None
                and time.monotonic() - self.last_fail > self.settings['session_timeout']):
            self.reset_session()  # 이전 사람의 실패 횟수가 다음 사람에게 넘어가지 않게
        if result.status == AuthStatus.SUCCESS:
            return self._success(result)
        return self._fail(result)

    def _success(self, result):
        self.reset_session()
        self._color((0, 1, 0))
        self.voice.say('success')
        snap = self._snapshot('success', result) if self.settings['snapshot_on_success'] else ''
        self._log('AUTH_SUCCESS', result, retry=0, snapshot=snap, note=result.message or 'door open')
        time.sleep(self.settings['unlock_seconds'])
        self._off()
        return Action.OPENED

    def _fail(self, result):
        self.fail_count += 1
        self.last_fail = time.monotonic()
        self._color((1, 0, 0))
        self._beep(self.settings['fail_beep_seconds'])
        snap = self._snapshot('fail', result) if self.settings['snapshot_on_fail'] else ''
        self._log('AUTH_FAIL', result, retry=self.fail_count, snapshot=snap, note=result.message)
        if self.fail_count < self.settings['max_retries']:
            self.voice.say('retry')
            time.sleep(1)
            self._off()
            return Action.RETRY
        return self._manual()

    def _manual(self):
        # 재시도 초과: 경보 -> 버튼 대기 -> 수동 개방 또는 잠금 유지. 버튼이 없으면 무조건 잠금.
        # 경보음이 울리는 동안 누른 버튼도 인정(경보 시작부터 버튼 입력을 받음).
        self._log('RETRY_EXCEEDED', retry=self.fail_count, note='alarm')
        pressed_event = threading.Event()
        if self.button is not None:
            self.button.when_pressed = lambda *_: pressed_event.set()
        try:
            if self.led is not None:
                self.led.blink(on_time=0.3, off_time=0.3, on_color=(1, 0, 0),
                               off_color=(0, 0, 0), background=True)
            self.voice.say('manual')
            for _ in range(self.settings['alarm_beeps']):
                if pressed_event.is_set():
                    break
                self._beep(0.2)
                time.sleep(0.2)
            if self.button is not None:
                pressed = pressed_event.wait(timeout=self.settings['manual_timeout'])
            else:
                time.sleep(self.settings['manual_timeout'])
                pressed = False
        finally:
            if self.button is not None:
                self.button.when_pressed = None
        self._off()
        self.reset_session()
        if pressed:
            self._color((0, 1, 0))
            self._log('MANUAL_OPEN', note='button override')
            time.sleep(self.settings['unlock_seconds'])
            self._off()
            return Action.MANUAL_OPEN
        self.voice.say('locked')
        self._log('MANUAL_TIMEOUT', note='no button' if self.button is None else 'stay locked')
        return Action.LOCKED

    def _color(self, color):
        if self.led is not None:
            self.led.color = color

    def _off(self):
        if self.led is not None:
            self.led.off()

    def _beep(self, seconds):
        if self.buzzer is None:
            time.sleep(seconds)
            return
        if self.settings['buzzer_type'] == 'passive':
            self.buzzer.play(880.0)  # 수동 부저는 주파수를 줘야 소리남
            time.sleep(seconds)
            self.buzzer.stop()
        else:
            self.buzzer.on()
            time.sleep(seconds)
            self.buzzer.off()

    def _log(self, event, result=None, retry='', snapshot='', note=''):
        user_id = score = auth_ms = ''
        if result is not None:
            user_id = result.user_id
            score = f'{result.score:.3f}' if isinstance(result.score, (int, float)) else result.score
            auth_ms = '' if result.auth_ms is None else f'{result.auth_ms:.1f}'
        row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), event, user_id, score,
               retry, auth_ms, snapshot, note]
        with self._log_lock, open(self.settings['log_path'], 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(row)

    def _snapshot(self, kind, result):
        # 과제 결과용 사진. 1단계는 준비 완료 후 프레임을 버리므로 인증에 쓴 frame을 받아서 저장.
        # frame이 있을 때만 저장하고 로그에는 파일 이름을 남김. max_snapshots장 넘으면 오래된 것부터 삭제.
        if result.frame is None:
            return ''
        if cv2 is None:
            print('안내: OpenCV(cv2)가 없어 결과 사진을 저장하지 않습니다.', file=sys.stderr)
            return ''
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            user = re.sub(r'[^\w-]', '_', str(result.user_id))[:30] or 'unknown'
            path = self.snapshot_dir / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{kind}_{user}.jpg'
            ok, encoded = cv2.imencode('.jpg', result.frame)  # 한글 경로 대응
            if not ok:
                return ''
            path.write_bytes(encoded.tobytes())
            self._prune_snapshots()
            return path.name
        except Exception as exc:
            print(f'안내: 사진 저장 실패 ({exc})', file=sys.stderr)
            return ''

    def _prune_snapshots(self):
        limit = self.settings['max_snapshots']
        if limit == 0:
            return
        files = sorted(self.snapshot_dir.glob('*.jpg'))  # 파일명이 시각 순
        for old in files[:max(0, len(files) - limit)]:
            old.unlink(missing_ok=True)

    def close(self):
        try:
            self._off()
            for device in (self.led, self.buzzer, self.button):
                if device is not None:
                    device.close()
        finally:
            self.voice.close()


def run_demo(handler):
    # 카메라·모델 없이 5단계만 확인. 4단계가 주는 형식 그대로 (이름, 점수)를 넣음.
    # 성공 1회 -> 연속 실패 max_retries회 -> 버튼(엔터) 대기.
    yield handler.handle(from_stage4(('demo', 0.80)))
    for _ in range(handler.settings['max_retries']):
        yield handler.handle(from_stage4((None, 0.20)))


if __name__ == '__main__':
    handler = None
    try:
        if '--demo' not in sys.argv:
            raise RuntimeError('단독 실행은 --demo만 지원합니다. (python 5-result/5-result.py --demo --no-gpio)')
        handler = ResultHandler(*make_devices(enabled='--no-gpio' not in sys.argv), keyboard_button=True)
        for action in run_demo(handler):
            print(action.value, flush=True)
    except RuntimeError as exc:
        print(f'안내: {exc}', file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        pass
    finally:
        if handler is not None:
            handler.close()
