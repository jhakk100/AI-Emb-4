#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"
VENV_DIR="${FACE_ACCESS_VENV:-$SCRIPT_DIR/.venv-pi}"
PYTHON="$VENV_DIR/bin/python"

usage() {
    echo '사용법: bash run_pi.sh [gui|run|enroll|users|delete|check|test] [옵션]'
    echo '옵션 없이 실행하면 화면이 있을 때 GUI, 없을 때 콘솔 인증을 실행합니다.'
    echo '등록: bash run_pi.sh enroll --name "이름"'
    echo '카메라 진단: bash run_pi.sh check --camera 0'
}

MODE="${1:-auto}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    -h|--help) usage; exit 0 ;;
    auto|gui|run|enroll|users|delete|check|test) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ ! -x "$PYTHON" ]]; then
    echo 'Pi용 설치를 먼저 실행하세요: bash setup_pi.sh' >&2
    exit 2
fi
"$PYTHON" check_pi.py --platform-only
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
if [[ "$MODE" == auto ]]; then
    if [[ -n "${DISPLAY:-}" ]]; then MODE=gui; else MODE=run; fi
fi
case "$MODE" in
    gui)
        if [[ -z "${DISPLAY:-}" ]]; then
            echo '화면 연결이 없습니다. bash run_pi.sh run을 사용하세요.' >&2
            exit 2
        fi
        exec "$PYTHON" app.py "$@" ;;
    run|enroll|users|delete) exec "$PYTHON" integrate.py "$MODE" "$@" ;;
    check) exec "$PYTHON" check_pi.py "$@" ;;
    test) exec "$PYTHON" -m unittest discover -s tests -v "$@" ;;
esac
