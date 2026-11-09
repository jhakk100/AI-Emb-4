"""Download the two official OpenCV models and verify their published LFS hashes."""

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODELS = (
    ("face_detection_yunet", "face_detection_yunet_2023mar.onnx",
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    ("face_recognition_sface", "face_recognition_sface_2021dec.onnx",
     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_models(progress=print):
    MODEL_DIR.mkdir(exist_ok=True)
    paths = []
    for folder, filename, expected in MODELS:
        path = MODEL_DIR / filename
        if not path.exists() or sha256(path) != expected:
            url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{filename}"
            partial = path.with_suffix(".part")
            progress(f"모델 다운로드: {filename}")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "LocalFaceDemo/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                    size = int(response.headers.get("Content-Length", 0))
                    received = 0
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                        received += len(chunk)
                        progress(f"{filename}: {received / 1048576:.1f} MB"
                                 + (f" / {size / 1048576:.1f} MB" if size else ""))
                if sha256(partial) != expected:
                    raise ValueError(f"모델 무결성 검증 실패: {filename}")
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        paths.append(path)
    return paths


if __name__ == "__main__":
    ensure_models()
    print("Models ready.")
