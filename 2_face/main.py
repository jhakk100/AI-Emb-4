from pathlib import Path
import cv2
import numpy as np
from datetime import datetime



def read_image(path):
    """한글 경로에서도 이미지 읽기."""
    image_bytes = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")

    return image


def save_image(path, image):
    """한글 경로에서도 이미지 저장."""
    success, encoded = cv2.imencode(path.suffix, image)

    if not success:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")

    path.write_bytes(encoded.tobytes())


def main():
    project_dir = Path(__file__).resolve().parent

    model_path = (
        project_dir / "models" / "face_detection_yunet_2023mar.onnx"
    )
    image_path = project_dir / "test_data" / "people.jpg"
    # 입력 파일명과 실행 시각으로 고유한 결과 폴더 생성
    run_name = (
       f"{image_path.stem}_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    )

    result_dir = project_dir / "results" / run_name


    for path in (model_path, image_path):
        if not path.is_file():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")


    # 1. 모델 로딩
    model_buffer = np.frombuffer(
        model_path.read_bytes(), dtype=np.uint8
    )

    detect_width = 320
    detect_height = 320

    detector = cv2.FaceDetectorYN.create(
        "onnx",
        model_buffer,
        np.empty(0, dtype=np.uint8),
        (detect_width, detect_height),
        0.8,
        0.3,
        5000
    )

        # 2. 원본 사진 읽기
    image = read_image(image_path)
    original_height, original_width = image.shape[:2]

    # 사진을 정상적으로 읽은 후 결과 폴더 생성
    result_dir.mkdir(parents=True, exist_ok=False)

    # 3. 비율을 유지하며 Resize하고 남는 영역에 패딩 추가
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

    pad_x = (detect_width - resized_width) // 2
    pad_y = (detect_height - resized_height) // 2

    detect_image = np.zeros(
        (detect_height, detect_width, 3), dtype=np.uint8
    )

    detect_image[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width
    ] = resized

    # 4. 얼굴 검출
    _, faces = detector.detect(detect_image)

    # 박스는 표시용 복사본에만 그림
    display = image.copy()
    saved_count = 0

    if faces is None:
        print("얼굴이 검출되지 않았습니다.")
    else:
        # 반올림된 Resize 크기를 기준으로 좌표 복원
        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        for face in faces:
            x, y, width, height = map(float, face[:4])
            confidence = float(face[-1])

            if width <= 0 or height <= 0:
                continue

            # 검출 좌표 → 원본 좌표
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

            # 5. 박스가 그려지지 않은 원본에서 얼굴 Crop
            crop = image[y1:y2, x1:x2].copy()

            saved_count += 1
            crop_path = result_dir / f"face_crop_{saved_count}.jpg"
            save_image(crop_path, crop)

            # 6. 표시용 이미지에 Bounding Box 그리기
            cv2.rectangle(
                display, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

            cv2.putText(
                display,
                f"Face {saved_count}: {confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(
                f"얼굴 {saved_count}: "
                f"검출 신뢰도={confidence:.3f}, "
                f"좌표=({x1}, {y1}, {x2}, {y2})"
            )
            print(f"Crop 저장: {crop_path}")

    # 7. 박스 표시 결과 저장
    box_path = result_dir / "face_detection.jpg"
    save_image(box_path, display)

    print(f"저장된 얼굴 Crop 개수: {saved_count}")
    print(f"검출 결과 이미지: {box_path}")

    # 큰 사진도 화면에서 확인할 수 있도록 축소 표시
    preview_scale = min(
        1.0, 1000 / original_width, 700 / original_height
    )
    preview = cv2.resize(
        display,
        (
            max(1, round(original_width * preview_scale)),
            max(1, round(original_height * preview_scale))
        )
    )

    cv2.imshow("Face Detection - press any key to close", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
