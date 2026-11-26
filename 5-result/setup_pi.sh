#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${FACE_ACCESS_VENV:-$SCRIPT_DIR/.venv-pi}"

if [[ "${1:-}" == "--help" ]]; then
    echo '사용법: bash setup_pi.sh'
    echo '64비트 Raspberry Pi OS, Python 3.11/3.12에서 실행하세요.'
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo '알 수 없는 옵션입니다. bash setup_pi.sh --help' >&2
    exit 2
fi
trap 'echo "설치가 중단되었습니다. 위 오류를 해결한 뒤 bash setup_pi.sh를 다시 실행하세요." >&2' ERR

# Reject unsupported interpreters before making any system changes.
"$PYTHON_BIN" check_pi.py --platform-only
if [[ -e "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Linux용 가상환경이 아닙니다: $VENV_DIR" >&2
    echo 'FACE_ACCESS_VENV에 새 폴더를 지정하세요. 기존 데이터는 자동 삭제하지 않습니다.' >&2
    exit 2
fi

APT=(apt-get)
if [[ $EUID -ne 0 ]]; then
    APT=(sudo apt-get)
fi
"${APT[@]}" update
"${APT[@]}" install -y python3-venv python3-tk python3-picamera2 \
    python3-gpiozero python3-lgpio espeak-ng alsa-utils libportaudio2 libgl1 libglib2.0-0
"$PYTHON_BIN" -c 'import picamera2, libcamera; print("정상: 시스템 Python에서 Picamera2/libcamera 사용 가능")'

# libcamera/Picamera2 come from apt and must remain visible inside the venv.
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip uninstall -y opencv-python opencv-python-headless \
    opencv-contrib-python opencv-contrib-python-headless
"$PYTHON" -m pip install --only-binary=:all: -r requirements.txt
"$PYTHON" stages/models.py
"$PYTHON" check_pi.py --models
echo '설치 완료. 실행: bash run_pi.sh'
echo '카메라 확인: bash run_pi.sh check --camera 0'
