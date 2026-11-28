"""Photo analysis shared by otherwise independent enrollment and authentication stages."""

import cv2

from detection import crop_face
from guided_capture import CaptureSample, normalized_mesh
from standby import overlap_ratio


class PhotoCapture:
    def _analyze(self, snapshot, detections):
        faces = self.engine.analyze(snapshot, detections)
        if len(faces) != 1 or faces[0].sharpness < 50 or min(faces[0].box[2:]) < 90:
            raise ValueError('얼굴이 작거나 흐립니다. 가까이에서 잠시 멈춰주세요.')
        return faces[0]

    def _photo_mesh(self, snapshot, detection, now):
        estimate = self.mesh.observe(snapshot, now)
        if estimate is None or overlap_ratio(tuple(detection[:4]), estimate.box) < .25:
            raise ValueError('촬영 사진의 Mesh와 YuNet 얼굴이 일치하지 않습니다.')
        return estimate

    def _snapshot(self, frame, detection, pose, now, *, mesh_estimate=None):
        snapshot = frame.copy()
        face = self._analyze(snapshot, [detection])
        estimate = mesh_estimate if mesh_estimate is not None else self._photo_mesh(snapshot, detection, now)
        size = snapshot.shape[1], snapshot.shape[0]
        geometry = normalized_mesh(estimate.landmarks, size)
        ok, jpeg = cv2.imencode('.jpg', crop_face(snapshot, detection[:4]))
        if not ok:
            raise ValueError('사진 인코딩 실패')
        index = sum(s.direction == self.challenge.target for s in self.challenge.samples)
        return face, CaptureSample(self.challenge.target, index, now, face.embedding.copy(),
            estimate.landmarks.copy(), geometry, pose, size, detection.copy(), jpeg.tobytes(),
            pose_basis=self.pose_guide.preset.basis if getattr(self, 'pose_guide', None) is not None else None)

    def close(self):
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None
