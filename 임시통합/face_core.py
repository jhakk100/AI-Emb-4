"""Face embeddings, local enrollment storage, and registration quality checks."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from models import ROOT
from detection import detect_faces

MODEL_ID = "opencv-sface-2021dec"
EMBEDDING_SIZE = 128
DEFAULT_THRESHOLD = 0.45
REGISTRATION_SAMPLES = 15


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size != EMBEDDING_SIZE or not np.isfinite(vector).all():
        raise ValueError("올바른 128차원 임베딩이 아닙니다.")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("빈 임베딩은 사용할 수 없습니다.")
    return vector / norm


class FaceStore:
    def __init__(self, path=ROOT / "data" / "embeddings.json"):
        self.path = Path(path)
        self.people = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if raw["version"] != 1 or raw["model"] != MODEL_ID:
                    raise ValueError("지원하지 않는 저장 형식 또는 모델입니다.")
                for name, person in raw["people"].items():
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("잘못된 등록 이름입니다.")
                    self.people[name] = {
                        "embedding": normalize(person["embedding"]).tolist(),
                        "samples": int(person["samples"]),
                        "updated_at": str(person["updated_at"]),
                    }
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"등록 파일을 읽지 못했습니다: {self.path}\n{exc}") from exc

    def _write(self, people):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "model": MODEL_ID, "people": people}
        # Commit both disk and memory only after a complete, atomic write.
        fd, temp = tempfile.mkstemp(prefix=".embeddings-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        finally:
            Path(temp).unlink(missing_ok=True)
        self.people = people

    def enroll(self, name, samples):
        name = name.strip()
        if not name or len(name) > 40:
            raise ValueError("이름은 1~40자로 입력하세요.")
        if len(samples) < REGISTRATION_SAMPLES:
            raise ValueError(f"등록에는 최소 {REGISTRATION_SAMPLES}개 샘플이 필요합니다.")
        embedding = normalize(np.mean([normalize(x) for x in samples], axis=0))
        people = dict(self.people)
        people[name] = {"embedding": embedding.tolist(), "samples": len(samples),
                        "updated_at": datetime.now(timezone.utc).isoformat()}
        self._write(people)

    def delete(self, name):
        people = dict(self.people)
        del people[name]
        self._write(people)

    def match(self, embedding, threshold=DEFAULT_THRESHOLD):
        if not 0 <= threshold <= 1:
            raise ValueError("인식 기준은 0~1 사이여야 합니다.")
        if not self.people:
            return None, None
        query = normalize(embedding)
        scores = [(name, float(np.clip(query @ np.asarray(person["embedding"], dtype=np.float32), -1, 1)))
                  for name, person in self.people.items()]
        name, score = max(scores, key=lambda item: item[1])
        return (name if score >= threshold else None), score


@dataclass
class Face:
    box: tuple
    embedding: np.ndarray
    sharpness: float


class FaceEngine:
    def __init__(self, detector_path, recognizer_path):
        # Buffer APIs also work when the project path contains Korean characters.
        empty = np.empty(0, dtype=np.uint8)
        self.detector = cv2.FaceDetectorYN.create(
            "onnx", np.frombuffer(Path(detector_path).read_bytes(), dtype=np.uint8),
            empty, (640, 480), 0.5, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(
            "onnx", np.frombuffer(Path(recognizer_path).read_bytes(), dtype=np.uint8), empty)

    def detect(self, frame, size=(640, 480)):
        return detect_faces(self.detector, frame, size)

    def analyze(self, frame, detections=None):
        if detections is None:
            detections = self.detect(frame)
        faces = []
        if detections is not None:
            for detection in detections:
                aligned = self.recognizer.alignCrop(frame, detection)
                feature = normalize(self.recognizer.feature(aligned).copy())
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                faces.append(Face(tuple(float(x) for x in detection[:4]), feature, sharpness))
        return faces


class Enrollment:
    def __init__(self, name):
        self.name = name
        self.samples = []
        self.last_sample = -float("inf")
        self.started = time.monotonic()

    def accept(self, faces, now):
        if len(faces) != 1:
            return "화면에 얼굴이 한 명만 나오도록 해주세요."
        face = faces[0]
        if min(face.box[2:]) < 90:
            return "카메라에 조금 더 가까이 와주세요."
        if face.sharpness < 50:
            return "얼굴이 흐립니다. 밝은 곳에서 잠시 멈춰주세요."
        if self.samples and float(self.samples[0] @ face.embedding) < 0.363:
            return "등록을 시작한 얼굴을 유지해주세요. 필요하면 취소 후 다시 등록하세요."
        if now - self.last_sample >= 0.3 and len(self.samples) < REGISTRATION_SAMPLES:
            self.samples.append(face.embedding.copy())
            self.last_sample = now
        return f"등록 중 {len(self.samples)}/{REGISTRATION_SAMPLES} · 고개를 좌우로 조금씩 움직여주세요."


def open_camera(index, size=(640, 480), fps=20):
    from capture_control import configure_capture
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if os.name == "nt" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened():
                configure_capture(capture, size, fps)
                # Some Windows devices open successfully but never deliver a frame.
                for _ in range(10):
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        return capture
        except Exception:
            capture.release()
            raise
        capture.release()
    raise RuntimeError(f"카메라 {index}를 열 수 없습니다. Windows 카메라 권한과 카메라 번호를 확인하세요.")
