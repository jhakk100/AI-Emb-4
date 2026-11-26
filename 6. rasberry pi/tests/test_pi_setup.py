"""Unsupported Pi environments must fail before installation or camera access."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stages'))

import check_pi


class PiSetupTests(unittest.TestCase):
    def test_supported_python_versions_and_64_bit_os(self):
        for version in ((3, 13, 5), (3, 13, 14)):
            check_pi.validate_platform('Linux', 'aarch64', version)

    def test_unsupported_platforms_fail_with_actionable_reason(self):
        cases = [('Windows', 'AMD64', (3, 11), '64비트'),
                 ('Linux', 'armv7l', (3, 11), '64비트'),
                 ('Linux', 'aarch64', (3, 13, 4), '3.13.5'),
                 ('Linux', 'aarch64', (3, 12, 9), '3.13.5'),
                 ('Linux', 'aarch64', (3, 14, 0), '3.13.5')]
        for system, machine, version, reason in cases:
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, reason):
                check_pi.validate_platform(system, machine, version)

    def test_target_platform_and_libc_are_accepted(self):
        with patch('check_pi.platform.system', return_value='Linux'), \
                patch('check_pi.platform.machine', return_value='aarch64'), \
                patch('check_pi.sys.version_info', (3, 13, 5)), \
                patch('check_pi.platform.libc_ver', return_value=('glibc', '2.41')), \
                redirect_stdout(StringIO()):
            check_pi.check_platform()

    def test_old_glibc_is_rejected_before_binary_imports(self):
        with patch('check_pi.platform.system', return_value='Linux'), \
                patch('check_pi.platform.machine', return_value='aarch64'), \
                patch('check_pi.sys.version_info', (3, 13, 5)), \
                patch('check_pi.platform.libc_ver', return_value=('glibc', '2.27')), \
                self.assertRaisesRegex(RuntimeError, 'glibc 2.28'):
            check_pi.check_platform()

    def test_platform_only_does_not_import_runtime_or_touch_camera(self):
        with patch('check_pi.check_platform'), \
                patch('check_pi.check_dependencies') as deps, \
                patch('check_pi.check_camera') as camera:
            self.assertEqual(check_pi.main(['--platform-only']), 0)
        deps.assert_not_called()
        camera.assert_not_called()

    def test_bad_platform_stops_before_dependency_import(self):
        with patch('check_pi.check_platform', side_effect=RuntimeError('unsupported')), \
                patch('check_pi.check_dependencies') as deps, redirect_stderr(StringIO()):
            self.assertEqual(check_pi.main(['--models', '--camera', '0']), 1)
        deps.assert_not_called()

    def test_camera_read_failure_still_releases_handle(self):
        camera = Mock()
        camera.read.return_value = False, None
        with patch('face_core.open_camera', return_value=camera):
            with self.assertRaises(RuntimeError):
                check_pi.check_camera(0)
        camera.release.assert_called_once()

    def test_missing_dependency_is_reported_and_remaining_imports_checked(self):
        names = []
        def importing(name):
            names.append(name)
            if name == 'picamera2':
                raise ImportError('libcamera unavailable')
            version = {'numpy': '2.5.4', 'mediapipe': '1.2.3'}.get(name, 'test')
            return SimpleNamespace(__version__=version, FaceDetectorYN=object, FaceRecognizerSF=object)
        with patch('check_pi.importlib.import_module', side_effect=importing), redirect_stdout(StringIO()):
            errors = check_pi.check_dependencies()
        self.assertEqual(len(errors), 1)
        self.assertIn('picamera2', errors[0])
        self.assertIn('tkinter', names)


class PiLauncherTests(unittest.TestCase):
    def setUp(self):
        self.bash = shutil.which('bash')
        if not self.bash and Path('C:/Program Files/Git/bin/bash.exe').is_file():
            self.bash = 'C:/Program Files/Git/bin/bash.exe'
        if not self.bash:
            self.skipTest('Bash is unavailable')
        scratch = ROOT / 'work'
        scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix='pi launcher ', dir=scratch)
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        executable = self.directory / 'bin/python'
        executable.parent.mkdir()
        executable.write_text(
            '#!/usr/bin/env bash\n'
            'printf "<call>\\n" >> "$PI_TEST_LOG"\n'
            'printf "%s\\n" "$@" >> "$PI_TEST_LOG"\n', encoding='utf-8', newline='\n')
        executable.chmod(0o755)
        self.log = self.directory / 'arguments.log'
        self.env = dict(os.environ, FACE_ACCESS_VENV=self.directory.as_posix(),
                        PI_TEST_LOG=self.log.as_posix(), DISPLAY='')

    def run_launcher(self, args):
        return subprocess.run([self.bash, (ROOT / 'run_pi.sh').as_posix(), *args],
                              cwd=self.directory, env=self.env, capture_output=True, timeout=10)

    def test_headless_default_uses_console_authentication(self):
        completed = self.run_launcher([])
        self.assertEqual(completed.returncode, 0, completed.stderr.decode('utf-8', errors='replace'))
        lines = self.log.read_text(encoding='utf-8').splitlines()
        self.assertEqual(lines, ['<call>', 'check_pi.py', '--platform-only',
                                 '<call>', 'integrate.py', 'run'])

    def test_enrollment_preserves_name_with_spaces_and_non_ascii(self):
        completed = self.run_launcher(['enroll', '--name', '홍 길동', '--camera', '1'])
        self.assertEqual(completed.returncode, 0, completed.stderr.decode('utf-8', errors='replace'))
        self.assertEqual(self.log.read_text(encoding='utf-8').splitlines()[-4:],
                         ['--name', '홍 길동', '--camera', '1'])


if __name__ == '__main__':
    unittest.main()
