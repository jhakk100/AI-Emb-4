from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


def save_image(path, image):
    """한글 경로에서도 이미지 저장."""
    success, encoded = cv2.imencode(path.suffix, image)

    if not success:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")

    path.write_bytes(encoded.tobytes())


def detect_faces(detector, image, detect_width=320, detect_height=320):
    """얼굴을 검출하고 원본 크기 기준의 박스와 얼굴 Crop을 반환."""
    original_height, original_width = image.shape[:2]

    # 1. 비율을 유지하며 크기 변경
    scale = min(
        detect_width / original_width,
        detect_height / original_height
    )

    resized_width = max(
        1, min(detect_width, round(original_width * scale))
    )
    resized_height = max(
        1, min(detect_height, round(original_height * scale))
    )

    resized = cv2.resize(
        image, (resized_width, resized_height)
    )

    # 2. 남는 영역에 검은색 패딩 추가
    pad_x = (detect_width - resized_width) // 2
    pad_y = (detect_height - resized_height) // 2

    detect_image = np.zeros(
        (detect_height, detect_width, 3), dtype=np.uint8
    )

    detect_image[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width
    ] = resized

    # 3. 얼굴 검출
    _, faces = detector.detect(detect_image)

    display = image.copy()
    crops = []

    if faces is None:
        return display, crops

    # 반올림된 Resize 크기를 기준으로 좌표 복원
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    for face in faces:
        x, y, width, height = map(float, face[:4])
        confidence = float(face[-1])

        if width <= 0 or height <= 0:
            continue

        # 4. 검출 좌표 → 원본 좌표
        x1 = (x - pad_x) / scale_x
        y1 = (y - pad_y) / scale_y
        x2 = (x + width - pad_x) / scale_x
        y2 = (y + height - pad_y) / scale_y

        # 영상 범위로 제한
        x1 = max(0, min(original_width, int(np.floor(x1))))
        y1 = max(0, min(original_height, int(np.floor(y1))))
        x2 = max(0, min(original_width, int(np.ceil(x2))))
        y2 = max(0, min(original_height, int(np.ceil(y2))))

        if x2 <= x1 or y2 <= y1:
            continue

        # 박스를 그리지 않은 원본에서 Crop
        crop = image[y1:y2, x1:x2].copy()
        crops.append(crop)

        face_number = len(crops)

        # 5. 표시용 이미지에 박스와 신뢰도 표시
        cv2.rectangle(
            display,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Face {face_number}: {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return display, crops


def main():
    project_dir = Path(__file__).resolve().parent

    model_path = (
        project_dir / "models" / "face_detection_yunet_2023mar.onnx"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}"
        )

    # 1. 모델 로딩
    model_buffer = np.frombuffer(
        model_path.read_bytes(), dtype=np.uint8
    )

    detector = cv2.FaceDetectorYN.create(
        "onnx",
        model_buffer,
        np.empty(0, dtype=np.uint8),
        (320, 320),
        0.8,   # 검출 신뢰도 임계값
        0.3,   # NMS 임계값
        5000
    )

    # 2. 노트북 기본 카메라 열기
    # 카메라가 여러 대라면 0을 1 또는 2로 변경
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "카메라를 열 수 없습니다. "
            "카메라 권한과 다른 프로그램의 카메라 사용 여부를 확인하세요."
        )

    # 요청 해상도: 실제 적용 여부는 카메라에 따라 다름
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("카메라 얼굴 검출 시작")
    print("S: 현재 화면과 얼굴 Crop 저장")
    print("Q 또는 ESC: 종료")

    try:
        while True:
            # 3. 카메라에서 프레임 읽기
            success, frame = cap.read()

            if not success:
                print("카메라에서 영상을 읽을 수 없습니다.")
                break

            # 거울처럼 표시
            # 실제 카메라 방향을 유지하려면 다음 줄을 삭제
            frame = cv2.flip(frame, 1)

            # 4. 현재 프레임에서 얼굴 검출
            display, crops = detect_faces(detector, frame)

            cv2.putText(
                display,
                f"Faces: {len(crops)} | S: Save | Q / ESC: Quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow("Webcam Face Detection", display)

            key = cv2.waitKey(1) & 0xFF

            # 5. 종료
            if key in (ord("q"), ord("Q"), 27):
                break

            # 6. S를 누르면 현재 프레임과 얼굴 Crop 저장
            if key in (ord("s"), ord("S")):
                run_name = (
                    f"webcam_{datetime.now():%Y%m%d_%H%M%S_%f}"
                )

                result_dir = project_dir / "results" / run_name
                result_dir.mkdir(parents=True, exist_ok=False)

                # 박스가 없는 원본 프레임 저장
                save_image(result_dir / "original.jpg", frame)

                # 박스가 표시된 프레임 저장
                save_image(
                    result_dir / "face_detection.jpg",
                    display
                )

                # 검출된 얼굴별 저장
                for index, crop in enumerate(crops, start=1):
                    crop_path = (
                        result_dir / f"face_crop_{index}.jpg"
                    )
                    save_image(crop_path, crop)

                print(f"\n저장 위치: {result_dir}")
                print(f"저장된 얼굴 Crop 개수: {len(crops)}")

    finally:
        # 오류가 발생해도 카메라와 창 정리
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
