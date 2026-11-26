# Raspberry Pi 5 8GB / Windows: 1~5단계 통합 실행 (5단계 담당: 결과 처리 + 통합).
# 1~4단계와 단일 카메라 동작 검사는 같은 폴더 stages/의 모듈을 쓰고,
# 결과는 5-result.py로 처리한다. 임시통합 폴더가 없어도 5-result 폴더만으로 동작한다.
#   1단계 standby.FaceWatcher      : 320×180 / 5fps 절전 검사, 위치·정면·크기·0.8초 유지
#   2단계 detection.detect_faces   : 640×480 활성 모드 YuNet 검출, 원본 좌표 박스 + 특징점 5개
#   3단계 face_core.FaceEngine     : 특징점 정렬 + SFace 128차원 임베딩
#   4단계 authenticate_face        : 등록 사용자(stages/data/faces.sqlite3)와 코사인 유사도 비교 -> (이름|None, 점수)
#   5단계 5-result.py              : from_stage4((이름, 점수), frame) -> handle() -> OPENED/RETRY/MANUAL_OPEN/LOCKED
#   (1~4단계 한 프레임 처리는 stages/pipeline.py의 FacePipeline.process()를 그대로 사용)
# 화면은 임시통합 app.py와 같게 좌우 반전해서 처리(등록·인증 모두 같은 설정, --no-flip으로 끔).
# 출력: 인증 시도마다 결과(OPENED 등) 한 줄. 안내·오류는 stderr, 오류 시 종료 코드 1.
#
# 준비: 5-result/setup.bat (가상환경 .venv + 패키지 + 모델 다운로드).
#       Pi: pip install -r 5-result/requirements.txt 후 python 5-result/stages/models.py
# 사용법 (레포 최상위에서, 5-result/.venv의 python):
#   python 5-result/integrate.py enroll --name 홍길동 [--show]   정면·상·하·좌·우 3장씩 DB 등록
#   python 5-result/integrate.py users                          등록 목록
#   python 5-result/integrate.py delete --name 홍길동            등록 삭제
#   python 5-result/integrate.py run [--show] [--once] [--no-gpio] [--no-voice] [--camera 0]
#   (노트북: run.bat = run --show --no-gpio --no-voice / Pi: run)

from pathlib import Path
from contextlib import ExitStack
import argparse
import importlib.util
import math
import re
import sys
import threading
import time

STAGES_DIR = Path(__file__).resolve().parent / 'stages'


def load_stage5():
    path = Path(__file__).with_name('5-result.py')
    spec = importlib.util.spec_from_file_location('stage5_result', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['stage5_result'] = module
    spec.loader.exec_module(module)
    return module


stage5 = load_stage5()


class Stages:
    """stages/ 검출·임베딩·동작 검사·인증 모듈 묶음."""
    def __init__(self, threshold=None):
        if not (STAGES_DIR / 'pipeline.py').is_file():
            raise RuntimeError(f'1~4단계 모듈 폴더가 없습니다: {STAGES_DIR}')
        if str(STAGES_DIR) not in sys.path:
            sys.path.insert(0, str(STAGES_DIR))  # stages 모듈끼리 서로 import하므로 경로 추가
        try:
            import cv2
            import capture_control
            import database
            import face_core
            import models
            import pipeline
        except ImportError as exc:
            raise RuntimeError(f'1~4단계 모듈을 불러올 수 없습니다 ({exc}). 5-result/setup.bat을 먼저 실행하세요.') from exc
        self.cv2, self.capture_control, self.database = cv2, capture_control, database
        self.face_core, self.pipeline_module = face_core, pipeline
        paths = models.ensure_models(lambda text: print(f'  {text}', file=sys.stderr))
        self.engine = face_core.FaceEngine(*paths)
        self.store = database.initialize_database()
        self.threshold = face_core.DEFAULT_THRESHOLD if threshold is None else threshold

    def new_pipeline(self, source, purpose="authenticate", enrollment_name=None):
        # The legacy interval argument remains compatible; a successful auth pair is stored atomically.
        return self.pipeline_module.FacePipeline(self.engine, store=self.store, source=source,
                                                 save_interval=1.0, threshold=self.threshold, purpose=purpose,
                                                 enrollment_name=enrollment_name)


class Camera:
    """임시통합 app.py와 같은 방식: 절전/활성 해상도 전환, 소프트웨어로 정확한 분석 해상도 맞춤."""
    def __init__(self, stages, index, flip=True, profile=None):
        self.stages, self.flip = stages, flip
        self.cv2, self.control = stages.cv2, stages.capture_control
        self.profile = profile or self.control.STANDBY_PROFILE
        self.capture = stages.face_core.open_camera(index, size=self.profile.size, fps=self.profile.fps)
        self.failures = 0
        self.resolution_gate = self.control.EnrollmentResolutionGate()

    def read(self, profile):
        if profile != self.profile:
            self.control.configure_capture(self.capture, profile.size, profile.fps)
            self.profile = profile
            self.resolution_gate.reset()
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.failures += 1
            if self.failures >= 10:
                raise RuntimeError('카메라 영상이 끊겼습니다. 다른 앱의 카메라 사용 여부를 확인하세요.')
            return None
        self.failures = 0
        if not self.resolution_gate.accept(frame, profile):
            return None
        if self.flip:
            frame = self.cv2.flip(frame, 1)
        return self.control.prepare_frame(frame, profile.size)

    def flush(self, count=5):
        # 5단계 처리(문 열림 3초 등) 동안 쌓인 오래된 프레임 버림
        for _ in range(count):
            self.capture.grab()

    def release(self):
        self.capture.release()


class Preview:
    """노트북 테스트용 창(--show). 화면 글자는 영어, 한글 상태는 바뀔 때만 콘솔 출력. ESC/Q 종료."""
    WINDOW = 'Face Access 1-5 (ESC/Q: quit)'

    def __init__(self, cv2):
        self.cv2 = cv2
        self.last = None
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 640, 480)

    def status(self, text):
        kind = re.sub(r'[\d.]+/[\d.]+초', '', text)
        if kind != self.last:
            self.last = kind
            print(f'  상태: {text}', file=sys.stderr, flush=True)

    def show(self, frame, result, banner=None, color=None, wait_ms=1):
        cv2 = self.cv2
        image = frame.copy()
        boxes = [face.box for face in result.faces] or (result.detection_boxes or [])
        box_color = color or ((0, 255, 255) if result.ready else (0, 165, 255) if boxes else (200, 200, 200))
        for x, y, w, h in boxes:
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), box_color, 2)
        text = banner or f'{result.power_mode} {result.processing_size[0]}x{result.processing_size[1]} | faces {result.detected_count}'
        cv2.rectangle(image, (0, 0), (image.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(image, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
        direction = getattr(result, 'direction', None)
        if direction:
            cx = image.shape[1] // 2
            offsets = {'up': (0, -25), 'down': (0, 25), 'left': (-25, 0), 'right': (25, 0)}
            cv2.rectangle(image, (cx-80, 30), (cx+80, 130), (0, 0, 0), -1)
            if direction in offsets:
                dx, dy = offsets[direction]
                cv2.arrowedLine(image, (cx-dx, 75-dy), (cx+dx, 75+dy), (0, 255, 255), 4, tipLength=.3)
            else:
                cv2.circle(image, (cx, 75), 15, (0, 255, 255), 3)
            cv2.putText(image, direction.upper(), (cx-35, 120), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.imshow(self.WINDOW, image)
        if cv2.waitKey(wait_ms) & 0xFF in (27, ord('q'), ord('Q')):
            raise KeyboardInterrupt
        if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt

    def pump(self):
        # 창 이벤트만 처리(5단계 처리 중에는 ESC로 끊지 않음 - 문 열림·잠금 처리를 끝까지 마침)
        self.cv2.waitKey(1)

    def close(self):
        try:
            self.cv2.destroyWindow(self.WINDOW)
            self.cv2.waitKey(1)
        except self.cv2.error:
            pass


def set_wait_led(led, power_mode, last):
    # 1단계 LED 규칙: 절전 대기 = 흰색 1초 깜빡임, 조건 통과(활성 인증) = 노란색
    if led is None or power_mode == last:
        return power_mode
    if power_mode == 'standby':
        led.blink(on_time=1, off_time=1, on_color=(1, 1, 1), off_color=(0, 0, 0), background=True)
    else:
        led.color = (1, 1, 0)
    return power_mode


def run_stage5(handler, auth, preview=None):
    # 5단계는 문 열림 3초·버튼 대기 10초 동안 멈춰 있으므로, 미리보기 창이 있으면
    # 별도 스레드에서 처리하고 그동안 창을 계속 갱신(Windows '응답 없음' 방지).
    if preview is None:
        return handler.handle(auth)
    outcome = {}

    def work():
        try:
            outcome['action'] = handler.handle(auth)
        except BaseException as exc:  # 5단계 오류는 메인 스레드로 넘겨서 그대로 처리
            outcome['error'] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        preview.pump()
        worker.join(0.05)
    if 'error' in outcome:
        raise outcome['error']
    return outcome['action']


def wait_departure(camera, pipeline, seconds, preview=None):
    # 처리된 사람이 화면에서 seconds초 동안 사라져야 다음 사람 대기 시작(같은 사람 반복 인증 방지)
    pipeline.set_purpose('departure')
    absent_since = None
    while True:
        started = time.monotonic()
        frame = camera.read(pipeline.profile)
        if frame is None:
            time.sleep(0.05)
            continue
        result = pipeline.process(frame)
        if preview is not None:
            preview.status('다음 사람: 현재 사용자는 화면에서 벗어나 주세요.')
            preview.show(frame, result, banner='LEAVE (step away)', color=(255, 255, 0))
        now = time.monotonic()
        if result.detected_count:
            absent_since = None
        elif absent_since is None:
            absent_since = now
        elif now - absent_since >= seconds:
            pipeline.set_purpose('authenticate')
            return
        time.sleep(camera.control.frame_delay(started, time.monotonic(), pipeline.profile.fps))


def cmd_run(args):
    stages = Stages(args.threshold)
    if not stages.database.get_users(stages.store):
        raise RuntimeError('등록된 사람이 없습니다. enroll(enroll.bat)로 먼저 등록하세요.')
    settings = dict(stage5.SET5)
    if args.no_voice:
        settings['enable_voice'] = False
    handler = camera = preview = pipeline = None
    try:
        handler = stage5.ResultHandler(*stage5.make_devices(settings, enabled=not args.no_gpio),
                                       settings=settings)
        camera = Camera(stages, args.camera, flip=not args.no_flip)
        pipeline = stages.new_pipeline(f'camera:{args.camera}')
        preview = Preview(stages.cv2) if args.show else None
        led_mode = None
        last_prompt = None
        print(f'인증 대기 (기준 유사도 {stages.threshold:.2f}): 화면 중앙에서 정면을 봐 주세요. 종료: Ctrl+C',
              file=sys.stderr, flush=True)
        while True:
            started = time.monotonic()
            profile = pipeline.profile
            frame = camera.read(profile)
            if frame is None:
                pipeline.reset_standby()
                time.sleep(0.05)
                continue
            result = pipeline.process(frame)                      # 1~4단계
            auth_ms = (time.monotonic() - started) * 1000
            led_mode = set_wait_led(handler.led, result.power_mode, led_mode)
            if result.status != last_prompt:
                print(f'  {result.status}', file=sys.stderr, flush=True)
                if result.liveness_step:
                    handler.voice.say_text(result.liveness_step)
                last_prompt = result.status
            if preview is not None:
                preview.status(result.status)
                preview.show(frame, result)
            attempted = result.authentication_attempted
            if not attempted:
                time.sleep(stages.capture_control.frame_delay(started, time.monotonic(), profile.fps))
                continue
            auth = stage5.from_stage4((result.authenticated_name, result.match_score),
                                      frame=frame, auth_ms=auth_ms)
            if result.authenticated_name is not None and not result.liveness_passed:
                raise RuntimeError('동작 검사 없는 인증 성공은 허용하지 않습니다.')
            auth.message = result.status
            if preview is not None:
                ok = auth.status == stage5.AuthStatus.SUCCESS
                preview.show(frame, result, wait_ms=300, color=(0, 200, 0) if ok else (0, 0, 255),
                             banner=f'{auth.status.value} {auth.user_id} score {auth.score:.3f}')
            action = run_stage5(handler, auth, preview)          # 5단계
            print(action.value, flush=True)
            led_mode = None
            camera.flush()
            if action == stage5.Action.RETRY:
                pipeline.reset_standby()                          # 1단계 유지 조건부터 다시
                continue
            if args.once:
                return
            wait_departure(camera, pipeline, args.departure, preview)
            handler.reset_session()
    finally:
        with ExitStack() as cleanup:
            if preview is not None:
                cleanup.callback(preview.close)
            if handler is not None:
                cleanup.callback(handler.close)
            if pipeline is not None:
                cleanup.callback(pipeline.close)
            if camera is not None:
                cleanup.callback(camera.release)


def cmd_enroll(args):
    # The worker/pipeline owns all 15 snapshots and commits a complete profile atomically.
    stages = Stages(args.threshold)
    target = stages.face_core.REGISTRATION_SAMPLES
    camera = preview = pipeline = None
    last_status = None
    try:
        camera = Camera(stages, args.camera, flip=not args.no_flip,
                        profile=stages.capture_control.ENROLLMENT_PROFILE)
        pipeline = stages.new_pipeline(f'camera:{args.camera}', purpose='enroll', enrollment_name=args.name)
        preview = Preview(stages.cv2) if args.show else None
        print(f'등록: {args.name} - 정면·상·하·좌·우마다 3장씩, 총 {target}장',
              file=sys.stderr, flush=True)
        while True:
            started = time.monotonic()
            profile = pipeline.profile
            frame = camera.read(profile)
            if frame is None:
                time.sleep(0.05)
                continue
            result = pipeline.process(frame)
            status = result.status
            kind = re.sub(r'[\d.]+/[\d.]+초', '', status)  # 유지 시간 숫자만 바뀌면 다시 출력 안 함
            if kind != last_status:
                last_status = kind
                print(f'  {status}', file=sys.stderr, flush=True)
            if result.enrollment_error:
                raise RuntimeError(result.enrollment_error)
            if result.enrollment_completed:
                break
            if preview is not None:
                preview.show(frame, result, banner=f'ENROLL {result.enrollment_count}/{target}', color=(255, 0, 255))
            time.sleep(stages.capture_control.frame_delay(started, time.monotonic(), profile.fps))
    finally:
        with ExitStack() as cleanup:
            if preview is not None:
                cleanup.callback(preview.close)
            if pipeline is not None:
                cleanup.callback(pipeline.close)
            if camera is not None:
                cleanup.callback(camera.release)
    print(f'등록 완료: {args.name}', flush=True)


def cmd_users(args):
    stages = Stages(args.threshold)
    users = stages.database.get_users(stages.store)
    print('\n'.join(users) if users else '(등록된 사람 없음)', flush=True)


def cmd_delete(args):
    stages = Stages(args.threshold)
    if args.name not in stages.database.get_users(stages.store):
        raise RuntimeError(f'등록되지 않은 이름입니다: {args.name}')
    stages.database.delete_user(stages.store, args.name)
    print(f'삭제 완료: {args.name}', flush=True)


def parser():
    result = argparse.ArgumentParser(description='1~5단계 통합 실행 (1~4: stages, 5: 5-result.py)')
    result.add_argument('--threshold', type=float, help='인증 기준 유사도 0~1 (기본: face_core 0.45)')
    commands = result.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='통합 실행')
    run.add_argument('--once', action='store_true', help='한 사람 처리 후 종료')
    run.add_argument('--no-gpio', action='store_true', help='GPIO 장치 생성 안 함(노트북)')
    run.add_argument('--no-voice', action='store_true', help='음성 안내 끔')
    run.add_argument('--departure', type=float, default=1.0, help='다음 사람 대기 전 화면 비움 시간(초)')
    enroll = commands.add_parser('enroll', help='정면·상·하·좌·우 3장씩 사진·Mesh·임베딩 등록')
    enroll.add_argument('--name', required=True)
    commands.add_parser('users', help='등록 목록')
    delete = commands.add_parser('delete', help='등록 삭제')
    delete.add_argument('--name', required=True)
    for command in (run, enroll):
        command.add_argument('--camera', type=int, default=0, help='카메라 번호(기본 0)')
        command.add_argument('--show', action='store_true', help='노트북 미리보기 창')
        command.add_argument('--no-flip', action='store_true', help='좌우 반전 끄기(등록과 인증을 같은 설정으로)')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        print('안내: --threshold는 0~1입니다.', file=sys.stderr)
        return 1
    departure = getattr(args, 'departure', 1.0)
    if not math.isfinite(departure) or departure <= 0:
        print('안내: --departure는 0보다 커야 합니다.', file=sys.stderr)
        return 1
    commands = {'run': cmd_run, 'enroll': cmd_enroll, 'users': cmd_users, 'delete': cmd_delete}
    try:
        commands[args.command](args)
        return 0
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f'안내: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
