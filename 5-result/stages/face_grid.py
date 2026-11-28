"""Four directional surface grids, each formed from three normalized Face Mesh photos."""

from functools import cache
import json
import math
from pathlib import Path

import cv2
import numpy as np

from face_core import normalize, REGISTRATION_SAMPLES
from guided_capture import (DIRECTIONS, MESH_AUTH_POSE_KIND, MESH_MODEL, enrollment_sample_direction_valid,
                            mesh_auth_pose_profile, mesh_errors,
                            sample_direction_matches, template_basis_matches)
from liveness import Pose
from pose_calibration import MAX_PROFILE_ROLL, ORDER

GRID_MODEL = 'four-pose-face-grid-468-v1'
FUSED_MODEL = 'fused-four-pose-face-468-v1'
TOPOLOGY = 'mediapipe-face-mesh-468-triangles-v1'


def _centered_mesh(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (468, 3) or not np.isfinite(vertices).all():
        raise ValueError('얼굴 grid는 유효한 468개 입체 좌표가 필요합니다.')
    return vertices-vertices.mean(axis=0)


def _rotation(source, target):
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def aligned_vertices(vertices, reference):
    source, target = _centered_mesh(vertices), _centered_mesh(reference)
    return source @ _rotation(source, target)


class MeshPoseGuide:
    """Read live orientation in the coordinate system of the enrolled frontal mesh."""

    def __init__(self, grids, settings):
        self.reference = _centered_mesh(grids['front']['vertices'])
        centers = [(0., 0., 0.)]
        centers.extend(self.angles(grids[direction]['vertices']) for direction in ORDER[1:])
        self.profile = mesh_auth_pose_profile(centers, settings)
        self.basis = dict(version=1, kind=MESH_AUTH_POSE_KIND, centers=self.profile.centers.tolist())
        self.neutral = None

    def angles(self, mesh):
        rotation = _rotation(self.reference, _centered_mesh(mesh))
        pitch, yaw, roll = cv2.RQDecomp3x3(rotation.T)[0]
        return -yaw, pitch, roll

    def pose(self, mesh, box):
        angles = self.angles(mesh)
        if self.neutral is not None:
            angles = tuple(angle-origin for angle, origin in zip(angles, self.neutral))
        return Pose(*angles, box)

    def set_neutral(self, pose):
        # A relaxed front check must not shift the pitch origin of later directions
        # beyond what the original frontal window could have accepted.
        yaw_ratio = pose.yaw / self.profile.scale[0]
        pitch_limit = self.profile.scale[1] * math.sqrt(max(0., self.profile.radii[0]**2-yaw_ratio**2))
        pitch = max(-pitch_limit, min(pitch_limit, pose.pitch))
        self.neutral = float(pose.yaw), float(pitch), float(pose.roll)

    def matches(self, direction, pose):
        return self.profile.classify(pose)[0] == direction

    def correction_hint(self, direction, pose):
        """Suggest one small correction when the requested direction is already the closest."""
        if (pose is None or direction not in self.profile.directions
                or not np.isfinite([pose.yaw, pose.pitch, pose.roll]).all()):
            return None
        nearest, acceptance = self.profile.distances(pose)
        index = self.profile.directions.index(direction)
        ratio = acceptance[index] / self.profile.radii[index]
        # This margin controls hint visibility only; classify() still decides acceptance.
        if int(nearest.argmin()) != index or ratio > 1.4:
            return None
        roll_error = abs(pose.roll-self.profile.centers[0, 2])
        if roll_error > MAX_PROFILE_ROLL:
            return 'level_head' if roll_error <= MAX_PROFILE_ROLL+5 and ratio <= 1 else None
        if ratio <= 1:
            return None
        yaw, pitch = self.profile.centers[index, :2]
        yaw_error = (pose.yaw-yaw) / self.profile.scale[0]
        pitch_error = (pose.pitch-pitch) / (self.profile.scale[1]*self.profile.pitch_scales[index])
        if abs(pitch_error) >= abs(yaw_error):
            return 'raise_chin' if pitch_error > 0 else 'lower_chin'
        return 'turn_left' if yaw_error > 0 else 'turn_right'


def build_grids(samples, settings):
    if len(samples) != REGISTRATION_SAMPLES:
        raise ValueError('얼굴 grid 생성에는 4방향 12장의 사진이 필요합니다.')
    grids = {}
    bases = [sample.pose_basis for sample in samples]
    if any(basis != bases[0] for basis in bases):
        raise ValueError('하나의 얼굴 grid에는 같은 각도 보정 기준을 사용해야 합니다.')
    for direction in DIRECTIONS:
        selected = sorted((sample for sample in samples if sample.direction == direction), key=lambda sample: sample.index)
        if [sample.index for sample in selected] != [0, 1, 2]:
            raise ValueError('각 방향의 grid에는 서로 다른 사진 3장이 필요합니다.')
        if any(sample.model != MESH_MODEL or not enrollment_sample_direction_valid(sample, settings) for sample in selected):
            raise ValueError('grid 사진의 모델 또는 방향이 일치하지 않습니다.')
        reference = np.asarray(selected[0].mesh, dtype=float)
        aligned = np.stack([aligned_vertices(sample.mesh, reference) for sample in selected])
        vertices = aligned.mean(axis=0).astype(np.float32)
        if any(mesh_errors(sample.mesh, vertices)[0] > settings.mesh_limit for sample in selected):
            raise ValueError('같은 방향의 얼굴 형상이 일치하지 않습니다. 다시 등록하세요.')
        grids[direction] = dict(direction=direction, vertices=vertices,
            embedding=normalize(np.mean([normalize(sample.embedding) for sample in selected], axis=0)),
            samples=3, grid_model=GRID_MODEL, mesh_model=MESH_MODEL, pose_basis=bases[0])
    return grids


def compare_grid(sample, grid, threshold, settings, trace=None):
    if (grid is None or grid['direction'] != sample.direction or grid['samples'] != 3
            or grid['grid_model'] != GRID_MODEL or grid['mesh_model'] != sample.model
            or not template_basis_matches(grid['pose_basis'], sample.pose_basis)
            or not sample_direction_matches(sample, settings)):
        return None
    score = float(normalize(sample.embedding) @ normalize(grid['embedding']))
    error, depth = mesh_errors(sample.mesh, grid['vertices'])
    passed = score >= threshold and error <= settings.mesh_limit and depth <= settings.depth_limit
    if trace is not None:
        trace('grid', score=score, shape_error=error, depth_error=depth, passed=passed)
    return (score, error, depth) if passed else None


def fuse_grids(grids):
    if set(grids) != set(DIRECTIONS):
        raise ValueError('통합 3D 모델에는 완성된 4방향 grid가 필요합니다.')
    if any(grid['grid_model'] != GRID_MODEL or grid['mesh_model'] != MESH_MODEL or grid['samples'] != 3
           or grid['pose_basis'] != grids['front']['pose_basis'] for grid in grids.values()):
        raise ValueError('같은 버전·보정 기준의 4방향 grid만 통합할 수 있습니다.')
    reference = grids['front']['vertices']
    aligned = np.stack([aligned_vertices(grids[direction]['vertices'], reference) for direction in DIRECTIONS])
    vertices = np.median(aligned, axis=0)
    vertices -= vertices.mean(axis=0)
    eye_scale = np.linalg.norm(vertices[33]-vertices[263])
    if not np.isfinite(vertices).all() or eye_scale < .1:
        raise ValueError('통합 3D 모델의 얼굴 형상이 올바르지 않습니다.')
    vertices = (vertices/eye_scale).astype(np.float32)
    return dict(vertices=vertices, embedding=normalize(np.mean([grids[d]['embedding'] for d in DIRECTIONS], axis=0)),
                directions=len(DIRECTIONS), model=FUSED_MODEL, mesh_model=MESH_MODEL, topology=TOPOLOGY,
                pose_basis=grids['front']['pose_basis'],
                source_errors={d: list(mesh_errors(grids[d]['vertices'], vertices)) for d in DIRECTIONS})


def compare_fused(sample, model, threshold, settings, trace=None):
    if (model is None or model['model'] != FUSED_MODEL or model['mesh_model'] != sample.model
            or model['directions'] != len(DIRECTIONS) or model['topology'] != TOPOLOGY
            or not template_basis_matches(model['pose_basis'], sample.pose_basis)
            or not sample_direction_matches(sample, settings)):
        return None
    score = float(normalize(sample.embedding) @ normalize(model['embedding']))
    error, depth = mesh_errors(sample.mesh, model['vertices'])
    passed = score >= threshold and error <= settings.mesh_limit and depth <= settings.depth_limit
    if trace is not None:
        trace('fused', score=score, shape_error=error, depth_error=depth, passed=passed)
    return (score, error, depth) if passed else None


@cache
def grid_triangles():
    payload = json.loads(Path(__file__).with_name('face_topology.json').read_text(encoding='utf-8'))
    triangles = np.asarray(payload['triangles'], dtype=np.int32)
    if payload['topology'] != TOPOLOGY or triangles.shape != (852, 3) or triangles.min() < 0 or triangles.max() >= 468:
        raise ValueError('얼굴 grid 연결 정보가 올바르지 않습니다.')
    return tuple(tuple(int(index) for index in triangle) for triangle in triangles)


@cache
def grid_edges():
    return tuple(sorted({tuple(sorted((triangle[first], triangle[second]))) for triangle in grid_triangles()
                         for first, second in ((0, 1), (1, 2), (2, 0))}))


def draw_mesh_preview(image, landmarks):
    """Draw only on the already resized preview; never alter the captured model input."""
    height, width = image.shape[:2]
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (478, 3) or not np.isfinite(points).all():
        raise ValueError('미리보기에 표시할 Mesh 좌표가 올바르지 않습니다.')
    pixels = np.rint(points[:468, :2]*[width, height]).astype(np.int32)
    for first, second in grid_edges():
        cv2.line(image, tuple(pixels[first]), tuple(pixels[second]), (60, 210, 120), 1, cv2.LINE_AA)
    return image


def render_grids(grids, cell=190):
    image = np.full((cell*2, cell*3, 3), (30, 25, 20), np.uint8)
    positions = {'up': (1, 0), 'left': (0, 1), 'front': (1, 1), 'right': (2, 1)}
    for direction, (column, row) in positions.items():
        x, y = column*cell, row*cell
        cv2.rectangle(image, (x+3, y+3), (x+cell-3, y+cell-3), (90, 90, 90), 1)
        cv2.putText(image, direction.upper(), (x+10, y+23), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
        if direction not in grids:
            continue
        vertices = np.asarray(grids[direction]['vertices'])
        xy = vertices[:, :2].copy()
        xy -= xy.min(axis=0)
        scale = (cell-35)/max(np.ptp(xy, axis=0).max(), 1e-6)
        xy *= scale
        xy += [x+(cell-np.ptp(xy[:, 0]))/2, y+30]
        pixels = np.rint(xy).astype(np.int32)
        for first, second in grid_edges():
            cv2.line(image, tuple(pixels[first]), tuple(pixels[second]), (90, 220, 130), 1, cv2.LINE_AA)
    return image


def render_fused_model(model, cell=190):
    image = np.full((cell*2, cell*3, 3), (30, 25, 20), np.uint8)
    vertices = np.asarray(model['vertices'])
    for column, angle in enumerate((-35, 0, 35)):
        theta = np.radians(angle)
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        points = vertices @ rotation.T
        xy = points[:, :2].copy()
        xy -= xy.min(axis=0)
        xy *= (cell*1.5)/max(np.ptp(xy[:, 1]), np.ptp(xy[:, 0])*2, 1e-6)
        xy += [column*cell+(cell-np.ptp(xy[:, 0]))/2, 40]
        pixels = np.rint(xy).astype(np.int32)
        cv2.putText(image, f'VIEW {angle:+d}', (column*cell+20, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
        for first, second in grid_edges():
            cv2.line(image, tuple(pixels[first]), tuple(pixels[second]), (230, 190, 80), 1, cv2.LINE_AA)
    return image
