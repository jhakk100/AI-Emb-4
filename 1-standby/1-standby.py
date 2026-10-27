# Raspberry Pi 5 8GB / Windows: 같은 폴더의 ver1.json 설정으로 실행.
# 필수 모듈: OpenCV(cv2). gpiozero.RGBLED 객체는 선택 사항.
# prepare_camera(device=None, led=None) -> (열린 카메라, 고정 ROI 좌표).
# 준비 중: 흰색 1초 켜짐/1초 꺼짐. 준비 완료: 노란색 지속.
# ROI=(x, y, 너비, 높이). 다음 단계에서 같은 영역을 자르고 camera.release() 호출.

from dataclasses import dataclass
from functools import cache
from pathlib import Path
import time
import json
import math
import sys
try:
    import cv2
except ModuleNotFoundError as exc:
    if exc.name != 'cv2':
        raise
    cv2 = None

PRESET_PATH = Path(__file__).with_name('set1.json')


def load_ver1(path=PRESET_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise ValueError('설정은 JSON 객체여야 합니다.')
        fps = data['fps']
        resolution = data['resolution']
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError('fps는 1~60 정수여야 합니다.')
        if (not isinstance(resolution, list) or len(resolution) != 2
                or any(type(value) is not int for value in resolution)
                or not 160 <= resolution[0] <= 1920 or not 90 <= resolution[1] <= 1080):
            raise ValueError('resolution은 [가로, 세로], 160x90~1920x1080 범위여야 합니다.')
        ranges = {'min_face_ratio': (0, 0.5), 'front_tolerance': (0.05, 1),
                  'center_range': (0.1, 1), 'min_confidence': (0.5, 0.99),
                  'hold_seconds': (0, 2)}
        result = {'fps': fps, 'resolution': tuple(resolution)}
        for key, (low, high) in ranges.items():
            value = data[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not low <= value <= high):
                raise ValueError(f'{key}는 {low}~{high} 범위의 숫자여야 합니다.')
            result[key] = float(value)
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'ver1 설정을 읽을 수 없습니다: {path} ({exc})') from exc


try:
    VER1 = load_ver1()
except RuntimeError as exc:
    if __name__ == '__main__':
        print(f'안내: {exc}')
        raise SystemExit(1)
    raise

CHECK_FPS = VER1['fps']
DETECT_SIZE = VER1['resolution']
FACE_MARGIN = 0.25  # 얼굴 상자 각 방향에 25% 여유
MODEL_PATH = Path(__file__).with_name('models') / 'face_detection_yunet_2023mar.onnx'


@dataclass
class GateSettings:
    min_face_ratio: float = VER1['min_face_ratio']
    front_tolerance: float = VER1['front_tolerance']
    center_range: float = VER1['center_range']  # 화면 중앙 허용 영역의 너비/높이 비율
    min_confidence: float = VER1['min_confidence']
    hold_seconds: float = VER1['hold_seconds']


def _require_opencv():
    if cv2 is None:
        raise RuntimeError(
            '필수 모듈 OpenCV(cv2)가 없습니다. '
            '현재 가상환경에서 python -m pip install "opencv-python<5"을 실행하세요.')


def open_usb_camera(device=None):
    _require_opencv()
    if sys.platform == 'win32':
        index = 0 if device is None else device
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, DETECT_SIZE[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DETECT_SIZE[1])
            camera.set(cv2.CAP_PROP_FPS, CHECK_FPS)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return camera
        camera.release()
        raise RuntimeError('웹캠을 열 수 없습니다. 카메라 권한과 다른 앱의 사용 여부를 확인하세요.')
    paths = [Path(device)] if device is not None else sorted(
        Path('/dev/v4l/by-id').glob('usb-*-video-index*'))
    if device is None:
        # by-id가 없는 USB도 확인. CSI/코덱 노드는 후보에서 제외한다.
        for node in sorted(Path('/dev').glob('video[0-9]*')):
            parent = (Path('/sys/class/video4linux') / node.name / 'device').resolve()
            if any((p / 'idVendor').exists() for p in (parent, *parent.parents)):
                paths.append(node)
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        camera = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
        try:
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, DETECT_SIZE[0])
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DETECT_SIZE[1])
                camera.set(cv2.CAP_PROP_FPS, CHECK_FPS)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = camera.read()
                if ok and frame is not None and frame.size:
                    return camera
        except BaseException:
            camera.release()
            raise
        camera.release()
    raise RuntimeError('사용 가능한 USB 카메라가 없습니다. 연결/장치 권한을 확인하세요.')



@cache
def _detector():
    _require_opencv()
    if not hasattr(cv2, 'FaceDetectorYN'):
        raise RuntimeError('얼굴 특징점 모듈이 없습니다. python -m pip install "opencv-python<5"을 실행하세요.')
    if not MODEL_PATH.is_file():
        raise RuntimeError(f'얼굴 특징점 모델이 없습니다: {MODEL_PATH}')
    try:
        return cv2.FaceDetectorYN.create(str(MODEL_PATH), '', DETECT_SIZE, 0.5, 0.3, 5000)
    except cv2.error as exc:
        raise RuntimeError('얼굴 특징점 모델을 읽을 수 없습니다. 모델 파일과 OpenCV 설치를 확인하세요.') from exc


def nearby_faces(frame):
    if frame is None or not frame.size:
        raise RuntimeError('카메라 영상을 받을 수 없습니다.')
    detector = _detector()
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    return [] if faces is None else faces


def features_present(box, points):
    # 모델이 추정한 다섯 점의 범위와 기본 배치만 확인한다.
    x, y, w, h = box
    if w <= 0 or h <= 0 or len(points) != 5:
        return False
    if not all(math.isfinite(v) for point in points for v in point):
        return False
    if not all(x <= px <= x+w and y <= py <= y+h for px, py in points):
        return False
    eye1, eye2, nose, mouth1, mouth2 = points
    return (abs(eye1[0] - eye2[0]) > w * 0.1
            and abs(mouth1[0] - mouth2[0]) > w * 0.05
            and max(eye1[1], eye2[1]) < nose[1] < min(mouth1[1], mouth2[1]))


def face_crop(box, width, height):
    x, y, w, h = box
    left, top = max(0, round(x-w*FACE_MARGIN)), max(0, round(y-h*FACE_MARGIN))
    right, bottom = min(width, round(x+w*(1+FACE_MARGIN))), min(height, round(y+h*(1+FACE_MARGIN)))
    return left, top, right-left, bottom-top


def overlap_ratio(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax+aw, bx+bw)-max(ax, bx)) * max(0, min(ay+ah, by+bh)-max(ay, by))
    return intersection / max(1, aw*ah + bw*bh - intersection)


def face_front_error(points):
    eye1, eye2, nose, mouth1, mouth2 = points
    eye_span = max(1, abs(eye1[0]-eye2[0]))
    eye_mid = (eye1[0]+eye2[0])/2
    mouth_mid = (mouth1[0]+mouth2[0])/2
    # 얼굴 방향의 근사 지표. 각도나 실제 시선을 측정하는 값은 아니다.
    return max(abs(nose[0]-eye_mid), abs(nose[0]-mouth_mid),
                      abs(eye1[1]-eye2[1])) / eye_span


def check_gate(face, points, settings):
    x, y, w, h = map(float, face[:4])
    width, height = DETECT_SIZE
    if not all(math.isfinite(float(v)) for v in face[:15]):
        return '특징점 데이터 불완전'
    if not features_present((x,y,w,h), points):
        return '눈·코·입 배치 불완전'
    if float(face[14]) < settings.min_confidence:
        return '검출 신뢰도 부족'
    if w / width < settings.min_face_ratio:
        return '얼굴이 작음 - 가까이 와 주세요'
    if x < 0 or y < 0 or x+w > width or y+h > height:
        return '얼굴이 화면 밖으로 잘림'
    if (abs((x+w/2)/width - 0.5) > settings.center_range/2
            or abs((y+h/2)/height - 0.5) > settings.center_range/2):
        return '중앙 허용범위 밖'
    front_error = face_front_error(points)
    if front_error > settings.front_tolerance:
        return '정면 허용오차 초과'
    return ''


class FaceWatcher:
    """UI에서 조절한 얼굴 크기·위치·특징점·유지 시간으로 촬영 준비 판단."""
    def __init__(self, settings=None):
        self.settings = settings or GateSettings()
        self.ready = False
        self.roi = None
        self.face = None
        self.landmarks = []
        self.shape = None
        self.reason = '얼굴 대기 중'
        self.valid_since = None
        self.last_valid = None
        self.previous_face = None
        self.hold_elapsed = 0.0
        self.score = 0.0
        self.face_ratio = 0.0
        self.front_error = 0.0

    def reset_hold(self):
        self.valid_since = self.last_valid = self.previous_face = None
        self.hold_elapsed = 0.0

    def update(self, frame):
        if frame is None or not frame.size:
            raise RuntimeError('카메라 영상을 받을 수 없습니다.')
        shape = frame.shape[:2]
        if self.shape is None:
            self.shape = shape
        elif shape != self.shape:
            raise RuntimeError('영상 해상도가 변경되었습니다. 대기를 다시 시작하세요.')
        if self.ready:
            return
        height, width = shape
        small = cv2.resize(frame, DETECT_SIZE)
        faces = nearby_faces(small)
        self.face = self.roi = None
        self.landmarks = []
        self.score = self.face_ratio = self.front_error = 0.0
        if len(faces) == 0:
            self.reason = '얼굴 대기 중'
            self.reset_hold()
            return
        candidates = []
        for face in faces:
            points = [(float(face[i]), float(face[i+1])) for i in range(4, 14, 2)]
            if not all(math.isfinite(float(v)) for v in face[:15]):
                continue
            reason = check_gate(face, points, self.settings)
            candidates.append((reason, face, points))
        if not candidates:
            self.reason = '특징점 데이터 불완전'
            self.reset_hold()
            return
        # 통과 후보 우선, 유지 중에는 기존 위치와 겹치는 후보 우선.
        reason, face, points = max(candidates, key=lambda item: (
            not bool(item[0]),
            overlap_ratio(self.previous_face, item[1][:4]) if self.previous_face else 0,
            item[1][2]*item[1][3]))
        self.front_error = face_front_error(points)
        self.score = float(face[14])
        self.face_ratio = float(face[2]) / DETECT_SIZE[0]
        sx, sy = width / DETECT_SIZE[0], height / DETECT_SIZE[1]
        x, y, w, h = face[:4]
        left, top = max(0, min(width-1, round(x*sx))), max(0, min(height-1, round(y*sy)))
        right = min(width, max(left+1, round((x+w)*sx)))
        bottom = min(height, max(top+1, round((y+h)*sy)))
        self.face = (left, top, right-left, bottom-top)
        self.roi = face_crop(self.face, width, height)
        self.landmarks = [(round(px*sx), round(py*sy)) for px, py in points]
        if reason:
            self.reason = reason
            self.reset_hold()
            return
        now = time.monotonic()
        box = tuple(map(float, face[:4]))
        if (self.valid_since is None or self.previous_face is None
                or overlap_ratio(self.previous_face, box) < 0.3
                or now-self.last_valid > max(0.5, 2/CHECK_FPS)):
            self.valid_since = now
        self.previous_face, self.last_valid = box, now
        self.hold_elapsed = now-self.valid_since
        self.ready = self.hold_elapsed >= self.settings.hold_seconds
        self.reason = '조건 통과 - 준비 완료' if self.ready else f'유지 확인 {self.hold_elapsed:.2f}/{self.settings.hold_seconds:.2f}초'


def prepare_camera(device=None, led=None):
    camera = None
    try:
        if led is not None:
            led.blink(on_time=1, off_time=1, on_color=(1, 1, 1),
                      off_color=(0, 0, 0), background=True)
        print(f'ver1 설정: {DETECT_SIZE[0]}x{DETECT_SIZE[1]}, {CHECK_FPS}fps')
        print('LED: 흰색 1초 켜짐 / 1초 꺼짐 반복 : 눈·코·입 확인 대기 중')
        _detector()
        camera = open_usb_camera(device)
        scanner = FaceWatcher()
        while True:
            started = time.monotonic()
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('카메라 영상을 받을 수 없습니다.')
            scanner.update(frame)
            if scanner.ready:
                if led is not None:
                    led.color = (1, 1, 0)
                print('LED: 노란색 지속 : 눈·코·입 확인, 카메라 준비 완료')
                return camera, scanner.roi
            time.sleep(max(0, 1 / CHECK_FPS - (time.monotonic() - started)))
    except BaseException:
        if camera is not None:
            camera.release()
        if led is not None:
            led.off()
        raise


if __name__ == '__main__':
    camera = None
    try:
        camera, roi = prepare_camera()
        print(f'다음 단계 전달 가능. 고정 ROI: {roi}')
    except RuntimeError as exc:
        print(f'안내: {exc}')
    except KeyboardInterrupt:
        print('종료')
    finally:
        if camera is not None:
            camera.release()
