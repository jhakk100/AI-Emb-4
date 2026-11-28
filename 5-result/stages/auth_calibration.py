"""Record authentication attempts and estimate the frontal score threshold from reviewed comparisons."""

from bisect import bisect_left
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import sqlite3

import numpy as np

from face_core import normalize
from face_grid import FUSED_MODEL
from guided_capture import DIRECTIONS, MANUAL_POSE_BASIS


MIN_CONFIRMED = 10
SCORE_WINDOW = 200


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def equal_error_threshold(genuine, impostor):
    """Pick the nearest attainable FAR/FRR crossing, with a midpoint between score steps."""
    if not genuine or not impostor:
        raise ValueError('본인 점수와 타인 점수가 모두 필요합니다.')
    genuine, impostor = sorted(genuine), sorted(impostor)
    cuts = sorted({0., 1., *(score for score in (*genuine, *impostor) if 0 < score < 1)})
    candidates = [0., 1., *((low+high)/2 for low, high in zip(cuts, cuts[1:]))]
    points = []
    for threshold in candidates:
        far = (len(impostor)-bisect_left(impostor, threshold))/len(impostor)
        frr = bisect_left(genuine, threshold)/len(genuine)
        points.append((threshold, far, frr))
    return min(points, key=lambda point: (abs(point[1]-point[2]), point[1]+point[2], point[1]))


@dataclass
class AuthenticationAttempt:
    id: int
    threshold: float
    threshold_source: str
    references: dict
    diagnostic: bool
    scored: bool = False


class AuthenticationCalibration:
    def __init__(self, connect):
        self.connect = connect
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
                    source TEXT NOT NULL, diagnostic INTEGER NOT NULL,
                    gallery_revision TEXT NOT NULL, threshold REAL NOT NULL, threshold_source TEXT NOT NULL,
                    score_candidates TEXT, outcome TEXT, reason TEXT, matched_name TEXT,
                    verified_kind TEXT CHECK(verified_kind IN ('registered','visitor')),
                    verified_name TEXT, verified_at TEXT
                );
                CREATE INDEX IF NOT EXISTS auth_attempt_review
                    ON auth_attempts(gallery_revision,id DESC) WHERE verified_at IS NOT NULL;
                CREATE TABLE IF NOT EXISTS auth_eer_policy (
                    gallery_revision TEXT PRIMARY KEY, updated_at TEXT NOT NULL,
                    genuine_count INTEGER NOT NULL, impostor_count INTEGER NOT NULL,
                    threshold REAL, far REAL, frr REAL
                );
            """)

    @staticmethod
    def _references(db):
        # Match the actual first-stage score source: manual frontal grid or legacy representative.
        rows = db.execute("""
            SELECT people.name, people.updated_at, models.pose_basis, front.embedding, people.embedding
            FROM people JOIN face_models AS models ON models.person=people.name
            JOIN face_grids AS front ON front.person=people.name AND front.direction='front'
            WHERE models.model=? AND models.directions=? ORDER BY people.name
        """, (FUSED_MODEL, len(DIRECTIONS))).fetchall()
        references = {}
        versions = []
        for name, updated, basis, grid, representative in rows:
            manual = basis is not None and json.loads(basis) == MANUAL_POSE_BASIS
            encoded = grid if manual else representative
            references[name] = normalize(json.loads(encoded))
            versions.append((name, updated, encoded))
        # A new or re-enrolled gallery starts calibration again. Prior attempt rows stay reviewable.
        version = sha256(json.dumps(versions).encode()).hexdigest()
        return version, references

    @staticmethod
    def _enabled(db):
        row = db.execute("SELECT value FROM metadata WHERE key='auth_eer_enabled'").fetchone()
        return row is None or row[0] == '1'

    @staticmethod
    def _policy(db, revision):
        row = db.execute("SELECT genuine_count,impostor_count,threshold,far,frr FROM auth_eer_policy "
                         "WHERE gallery_revision=?", (revision,)).fetchone()
        return row or (0, 0, None, None, None)

    def status(self):
        with closing(self.connect()) as db:
            revision, _ = self._references(db)
            genuine, impostor, threshold, far, frr = self._policy(db, revision)
            return dict(enabled=self._enabled(db), genuine=genuine, impostor=impostor,
                        threshold=threshold, far=far, frr=frr)

    def set_enabled(self, enabled):
        with closing(self.connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('auth_eer_enabled',?)", ('1' if enabled else '0',))

    def begin(self, source, baseline, diagnostic=False, allow_adaptive=True):
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            revision, references = self._references(db)
            _, _, adaptive, _, _ = self._policy(db, revision)
            active = allow_adaptive and adaptive is not None and self._enabled(db)
            threshold, threshold_source = (adaptive, 'eer') if active else (baseline, 'baseline')
            row = db.execute("""INSERT INTO auth_attempts
                (started_at,source,diagnostic,gallery_revision,threshold,threshold_source) VALUES (?,?,?,?,?,?)""",
                (_utc_now(), source, int(diagnostic), revision, threshold, threshold_source))
            return AuthenticationAttempt(row.lastrowid, threshold, threshold_source, references, diagnostic)

    def record_scores(self, attempt, embedding):
        if attempt.scored or not attempt.references:
            return
        query = normalize(embedding)
        scores = {name: float(np.clip(query @ vector, -1, 1)) for name, vector in attempt.references.items()}
        with closing(self.connect()) as db, db:
            db.execute('UPDATE auth_attempts SET score_candidates=? WHERE id=? AND ended_at IS NULL '
                       'AND score_candidates IS NULL', (json.dumps(scores, ensure_ascii=False), attempt.id))
        attempt.scored = True

    def finish(self, attempt, outcome, reason, matched_name=None):
        with closing(self.connect()) as db, db:
            db.execute('UPDATE auth_attempts SET ended_at=?,outcome=?,reason=?,matched_name=? '
                       'WHERE id=? AND ended_at IS NULL', (_utc_now(), outcome, reason, matched_name, attempt.id))

    def recent_attempts(self, limit=100):
        with closing(self.connect()) as db:
            db.row_factory = sqlite3.Row
            attempts = db.execute('SELECT * FROM auth_attempts ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row)|{'score_candidates': json.loads(row['score_candidates'] or '{}')} for row in attempts]

    def review(self, attempt_id, kind, name=None):
        if kind not in (None, 'registered', 'visitor') or (kind == 'registered') != bool(name):
            raise ValueError('확인한 등록자 이름 또는 미등록 타인을 선택하세요.')
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT gallery_revision,score_candidates FROM auth_attempts WHERE id=? '
                             'AND ended_at IS NOT NULL', (attempt_id,)).fetchone()
            if row is None:
                raise ValueError('끝난 인증 시도를 선택하세요.')
            revision, scores = row
            if name and scores and name not in json.loads(scores):
                raise ValueError('선택한 사람의 점수가 이 시도에 없습니다.')
            if name and not scores and not db.execute('SELECT 1 FROM people WHERE name=?', (name,)).fetchone():
                raise ValueError('등록된 사람을 선택하세요.')
            db.execute('UPDATE auth_attempts SET verified_kind=?,verified_name=?,verified_at=? WHERE id=?',
                       (kind, name, _utc_now() if kind else None, attempt_id))
            self._recalculate(db, revision)

    @staticmethod
    def _recalculate(db, revision):
        genuine, impostor = [], []
        rows = db.execute('SELECT verified_kind,verified_name,score_candidates FROM auth_attempts '
                          'WHERE gallery_revision=? AND verified_at IS NOT NULL AND score_candidates IS NOT NULL '
                          'ORDER BY id DESC', (revision,))
        for kind, name, encoded in rows:
            scores = json.loads(encoded)
            if kind == 'registered' and name in scores and len(genuine) < SCORE_WINDOW:
                genuine.append(scores[name])
            # One negative per attempt: the strongest competing comparison, not every gallery entry.
            if len(impostor) < SCORE_WINDOW:
                other = max((score for person, score in scores.items() if person != name), default=None)
                if other is not None:
                    impostor.append(other)
            if len(genuine) == len(impostor) == SCORE_WINDOW:
                break
        point = (None, None, None)
        if len(genuine) >= MIN_CONFIRMED and len(impostor) >= MIN_CONFIRMED:
            point = equal_error_threshold(genuine, impostor)
        db.execute('INSERT OR REPLACE INTO auth_eer_policy VALUES (?,?,?,?,?,?,?)',
                   (revision, _utc_now(), len(genuine), len(impostor), *point))
