"""Local, opt-in JSONL diagnostics for a batch of authentication attempts."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4


ATTEMPT_LIMIT = 10
LOG_DIR = Path(__file__).resolve().parents[1] / 'logs' / 'auth_diagnostics'


class AuthDiagnostics:
    def __init__(self, directory=LOG_DIR):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = directory / f'auth-{stamp}-{uuid4().hex[:8]}.jsonl'
        self.completed = 0
        self.error = None
        self.stream = None
        self.started = None
        with self.path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps({'event': 'batch_start', 'schema': 1, 'attempt_limit': ATTEMPT_LIMIT}) + '\n')

    @property
    def active(self):
        return self.stream is not None

    def begin(self, **fields):
        if self.active or self.completed >= ATTEMPT_LIMIT or self.error:
            return False
        try:
            self.stream = self.path.open('a', encoding='utf-8', newline='\n', buffering=1)
        except OSError as exc:
            self.error = str(exc)
            return False
        self.started = monotonic()
        self.write('attempt_start', **fields)
        return not self.error

    def write(self, event, **fields):
        if not self.active or self.error:
            return
        row = dict(event=event, attempt=self.completed+1, elapsed_ms=round((monotonic()-self.started)*1000, 2),
                   **fields)
        try:
            self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        except (OSError, TypeError, ValueError) as exc:
            self.error = str(exc)

    def finish(self, outcome, reason):
        if not self.active:
            return
        self.write('attempt_end', outcome=outcome, reason=reason)
        if self.completed+1 == ATTEMPT_LIMIT:
            self.write('batch_end', completed=ATTEMPT_LIMIT)
        try:
            self.stream.close()
        except OSError as exc:
            self.error = str(exc)
        finally:
            self.stream = None
        if not self.error:
            self.completed += 1
