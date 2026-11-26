"""Pi preflight; camera and model execution are explicit opt-in diagnostics, no DB writes."""

import argparse
import importlib
import platform
from pathlib import Path
import sys
import time


def validate_platform(system, machine, version):
    if system != 'Linux' or machine.lower() != 'aarch64':
        raise RuntimeError('64비트 Raspberry Pi OS(aarch64)가 필요합니다.')
    if tuple(version[:2]) != (3, 13) or tuple(version[:3]) < (3, 13, 5):
        raise RuntimeError(
            '현재 Pi 설치 조합은 Python 3.13.5 이상인 3.13 계열용입니다. '
            'Python 3.13.5와 같은 인터프리터용 libcamera/Picamera2를 사용하세요.')


def check_platform():
    validate_platform(platform.system(), platform.machine(), sys.version_info)
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(int(v) for v in version.split('.')[:2]) < (2, 28):
        raise RuntimeError('MediaPipe ARM64 배포 파일은 glibc 2.28 이상이 필요합니다.')
    print(f'환경 확인: {platform.machine()} / Python {platform.python_version()}')


def check_dependencies():
    errors = []
    for name in ('numpy', 'cv2', 'mediapipe', 'PIL', 'picamera2', 'libcamera', 'gpiozero', 'tkinter'):
        try:
            module = importlib.import_module(name)
            if name == 'numpy' and module.__version__ != '2.5.4':
                raise RuntimeError('NumPy 2.5.4가 필요합니다. setup_pi.sh를 다시 실행하세요.')
            if name == 'mediapipe' and module.__version__ != '1.2.3':
                raise RuntimeError('MediaPipe 1.2.3이 필요합니다. setup_pi.sh를 다시 실행하세요.')
            if name == 'cv2' and (not hasattr(module, 'FaceDetectorYN') or not hasattr(module, 'FaceRecognizerSF')):
                raise RuntimeError('YuNet/SFace를 지원하는 OpenCV contrib가 필요합니다.')
            print(f'정상: {name} {getattr(module, "__version__", "")}')
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    return errors


def check_models():
    import numpy as np
    from face_core import FaceEngine
    from liveness import FaceMeshEstimator
    from models import ensure_models
    engine = FaceEngine(*ensure_models())
    mesh = FaceMeshEstimator(print)
    try:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.detect(blank)
        mesh.observe(blank, time.monotonic())
        print('정상: YuNet / SFace / Face Mesh 모델 로딩 및 검출 실행')
    finally:
        mesh.close()


def check_camera(index):
    from face_core import open_camera
    capture = open_camera(index, size=(640, 480), fps=20)
    try:
        ok, frame = capture.read()
        if not ok or frame is None or not frame.size:
            raise RuntimeError('카메라에서 영상을 받지 못했습니다.')
        print(f'정상: 카메라 {index} / {frame.shape[1]}×{frame.shape[0]} / 촬영 후 해제')
    finally:
        capture.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description='라즈베리파이 설치·모델·카메라 점검 (얼굴 DB 저장 없음)')
    parser.add_argument('--platform-only', action='store_true', help='설치 전 OS/Python만 확인')
    parser.add_argument('--models', action='store_true', help='모델 준비 후 빈 영상으로 실행 확인')
    parser.add_argument('--camera', type=int, help='지정한 카메라의 프레임을 읽고 해제')
    args = parser.parse_args(argv)
    if args.camera is not None and args.camera < 0:
        parser.error('--camera는 0 이상이어야 합니다.')
    try:
        check_platform()
        if args.platform_only:
            return 0
        errors = check_dependencies()
        if errors:
            raise RuntimeError('\n'.join(errors))
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'stages'))
        if args.models:
            check_models()
        if args.camera is not None:
            check_camera(args.camera)
        print('점검 완료. GPIO 핀 배선·권한과 실제 인증 동작은 기기에서 확인하세요.')
        return 0
    except Exception as exc:
        print(f'점검 실패: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
