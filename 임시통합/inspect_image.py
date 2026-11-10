"""Offline stage 2-4 diagnostics; images are saved only with --output."""

import argparse
import json
from pathlib import Path

from authenticate import authenticate_face
from database import initialize_database
from detection import crop_face, read_image, save_image
from face_core import DEFAULT_THRESHOLD, FaceEngine
from models import ensure_models


def main():
    parser = argparse.ArgumentParser(description="이미지의 얼굴 검출·임베딩·DB 인증 검사 (대기 단계 제외)")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, help="지정한 경우에만 얼굴 Crop 저장")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold는 0~1 사이여야 합니다.")
    frame = read_image(args.image)
    engine = FaceEngine(*ensure_models())
    store = initialize_database()
    results = []
    for index, face in enumerate(engine.analyze(frame), 1):
        name, score = authenticate_face(store, face.embedding, args.threshold)
        results.append({"face": index, "box": face.box, "embedding_size": int(face.embedding.size),
                        "name": name, "score": score})
        if args.output:
            save_image(args.output / f"face_crop_{index}.jpg", crop_face(frame, face.box))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
