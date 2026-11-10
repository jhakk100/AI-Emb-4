"""Stage 4: separate enrollment and observation tables in one SQLite DB."""

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import cv2

from detection import crop_face
from face_core import DEFAULT_THRESHOLD, FaceStore, MODEL_ID, normalize
from models import ROOT

DEFAULT_DATABASE_PATH = ROOT / "data" / "faces.sqlite3"


class FaceDatabase(FaceStore):
    """FaceStore-compatible enrollment plus an independent stream of crops."""

    def __init__(self, path=DEFAULT_DATABASE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS people (
                    name TEXT PRIMARY KEY, embedding TEXT NOT NULL,
                    samples INTEGER NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL, source TEXT NOT NULL,
                    box TEXT NOT NULL, detection_score REAL NOT NULL,
                    crop_jpeg BLOB NOT NULL, embedding TEXT,
                    gate_passed INTEGER NOT NULL, embedding_error TEXT,
                    matched_name TEXT, match_score REAL, match_threshold REAL
                );
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(observations)")}
            for name, kind in (("matched_name", "TEXT"), ("match_score", "REAL"), ("match_threshold", "REAL")):
                if name not in columns:
                    connection.execute(f"ALTER TABLE observations ADD COLUMN {name} {kind}")
            existing = connection.execute("SELECT value FROM metadata WHERE key='model'").fetchone()
            if existing is not None and existing[0] != MODEL_ID:
                raise ValueError("DB의 임베딩 모델이 현재 모델과 다릅니다.")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('model', ?)", (MODEL_ID,))
        self.refresh()

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def refresh(self):
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT name, embedding, samples, updated_at FROM people").fetchall()
        self.people = {name: {"embedding": normalize(json.loads(embedding)).tolist(),
                              "samples": samples, "updated_at": updated}
                       for name, embedding, samples, updated in rows}

    def _write(self, people):
        with closing(self._connect()) as connection, connection:
            # Only explicit enrollment/deletion touches this table.
            connection.execute("DELETE FROM people")
            connection.executemany("INSERT INTO people VALUES (?, ?, ?, ?)", [
                (name, json.dumps(person["embedding"], allow_nan=False),
                 person["samples"], person["updated_at"]) for name, person in people.items()])
        self.people = people

    def record_authenticated(self, frame, detection, embedding, source="camera:0",
                             gate_passed=False, threshold=DEFAULT_THRESHOLD):
        """Only a gate-passing match may write a crop; commit crop+embedding together."""
        if not gate_passed:
            return None
        self.refresh()
        name, score = self.match(embedding, threshold)
        if name is None:
            return None
        crop = crop_face(frame, detection[:4])
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            raise RuntimeError("얼굴 Crop을 JPEG로 인코딩하지 못했습니다.")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute("""INSERT INTO observations
                (captured_at, source, box, detection_score, crop_jpeg, embedding, gate_passed,
                 matched_name, match_score, match_threshold)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (datetime.now(timezone.utc).isoformat(), source,
                 json.dumps([float(v) for v in detection[:4]]), float(detection[14]),
                 encoded.tobytes(), json.dumps(normalize(embedding).tolist(), allow_nan=False),
                 name, score, threshold))
            return cursor.lastrowid

    def count_observations(self):
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    def recent_observations(self, limit=10):
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute("""SELECT id, captured_at, source,
                gate_passed, embedding IS NOT NULL AS has_embedding, embedding_error,
                matched_name, match_score, match_threshold
                FROM observations ORDER BY id DESC LIMIT ?""", (limit,))]

    def observation_crop(self, sample_id):
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT crop_jpeg FROM observations WHERE id=?", (sample_id,)).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return row[0]


def initialize_database(path=DEFAULT_DATABASE_PATH):
    return FaceDatabase(path)


def add_user(store, name, samples):
    store.enroll(name, samples)


def delete_user(store, name):
    store.delete(name)


def get_users(store):
    return list(store.people.keys())


def get_database_path():
    return DEFAULT_DATABASE_PATH
