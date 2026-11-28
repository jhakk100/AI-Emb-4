"""Named default pose calibration and the shared stabilized pose guide."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from guided_capture import yunet_pose
from landmark_filter import LandmarkFilter, LandmarkFilterSettings
from liveness import Pose
from models import ROOT
from pose_calibration import PoseProfile


@dataclass
class PosePreset:
    name: str
    profile: PoseProfile
    context: dict
    path: Path

    @property
    def basis(self):
        return dict(version=1, name=self.name, centers=self.profile.centers.tolist())


def load_default_preset(directory=ROOT):
    directory = Path(directory).resolve()
    config = directory / 'pose_defaults.json'
    if not config.exists():
        return None
    data = json.loads(config.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.get('version') != 1 or type(data.get('enabled')) is not bool:
        raise ValueError('기본 보정 설정 형식이 올바르지 않습니다.')
    if not data['enabled']:
        return None
    if not isinstance(data.get('preset'), str) or not data['preset']:
        raise ValueError('기본 보정 파일 이름이 필요합니다.')
    path = (directory / data['preset']).resolve()
    if not path.is_relative_to(directory):
        raise ValueError('기본 보정 파일은 설정 폴더 안에 있어야 합니다.')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict) or not isinstance(payload.get('context'), dict):
        raise ValueError('기본 보정의 카메라 설정이 올바르지 않습니다.')
    context = payload['context']
    profile = PoseProfile.load(path, context)
    if (not isinstance(data.get('name'), str) or not data['name'].strip()
            or not context.get('mirrored') or context.get('solver') != 'yunet-5pt-sqpnp-v1'
            or context.get('manual_offsets') is not None or not context.get('stabilized')):
        raise ValueError('기본 보정은 좌표 안정화·좌우 반전 설정으로 기록해야 합니다.')
    size = context.get('size')
    if size != [1920, 1080]:
        raise ValueError('등록·인증 기본 보정은 FHD 1920×1080 기준이어야 합니다.')
    if not isinstance(context.get('filtering'), dict) or type(context.get('camera')) is not int:
        raise ValueError('기본 좌표 안정화/카메라 설정을 확인하세요.')
    LandmarkFilterSettings(**context['filtering'])
    return PosePreset(data['name'], profile, context, path)


class DefaultPoseGuide:
    def __init__(self, preset, source='camera:0'):
        if source.startswith('camera:') and source != f'camera:{preset.context["camera"]}':
            raise ValueError(f'기본값 {preset.name}의 카메라 번호와 다릅니다. 해당 카메라로 다시 보정하세요.')
        self.preset = preset
        self.filter = LandmarkFilter(LandmarkFilterSettings(**preset.context['filtering']))
        self.reset()

    def reset(self):
        self.filter.reset()
        self.shift = None

    def observe(self, row, size, now):
        if list(size) != self.preset.context['size']:
            raise ValueError('기본 보정값과 같은 FHD 영상이 필요합니다.')
        result = self.filter.update([row], size, now)
        return None if result.row is None else yunet_pose(result.row, size)

    def calibrate_neutral(self, pose):
        self.shift = np.array([pose.yaw, pose.pitch])-self.preset.profile.centers[0, :2]

    def align(self, pose):
        if self.shift is None:
            raise ValueError('정면 준비를 먼저 완료하세요.')
        return Pose(pose.yaw-self.shift[0], pose.pitch-self.shift[1], pose.roll, pose.box)

    def matches(self, direction, pose):
        return self.preset.profile.classify(pose)[0] == direction

    def after_photo(self, now):
        if self.filter.last_seen is not None:
            self.filter.last_seen = self.filter.last_accepted = now
