"""Stage 4: separate enrollment and observation tables in one SQLite DB."""

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

import cv2
import numpy as np

from detection import crop_face
from face_core import DEFAULT_THRESHOLD, FaceStore, MODEL_ID, normalize
from models import ROOT
from guided_capture import (DIRECTIONS, MESH_MODEL, compare_sample, normalized_mesh,
                            direction_matches, load_guided_settings)

DEFAULT_DATABASE_PATH = ROOT / "data" / "faces.sqlite3"


class FaceDatabase(FaceStore):
    """Legacy embedding compatibility plus directional profiles and paired observations."""

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
                CREATE TABLE IF NOT EXISTS directional_samples (
                    person TEXT NOT NULL REFERENCES people(name) ON DELETE CASCADE,
                    direction TEXT NOT NULL CHECK(direction IN ('front','up','down','left','right')),
                    sample_index INTEGER NOT NULL CHECK(sample_index BETWEEN 0 AND 2),
                    embedding TEXT NOT NULL, landmarks TEXT NOT NULL, mesh TEXT NOT NULL,
                    pose TEXT NOT NULL, image_size TEXT NOT NULL, crop_jpeg BLOB NOT NULL,
                    mesh_model TEXT NOT NULL, captured_at TEXT NOT NULL,
                    PRIMARY KEY(person, direction, sample_index)
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
            for name, kind in (("matched_name", "TEXT"), ("match_score", "REAL"), ("match_threshold", "REAL"),
                               ("direction", "TEXT"), ("mesh_error", "REAL"), ("depth_error", "REAL"),
                               ("auth_session", "TEXT")):
                if name not in columns:
                    connection.execute(f"ALTER TABLE observations ADD COLUMN {name} {kind}")
            existing = connection.execute("SELECT value FROM metadata WHERE key='model'").fetchone()
            if existing is not None and existing[0] != MODEL_ID:
                raise ValueError("DB의 임베딩 모델이 현재 모델과 다릅니다.")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('model', ?)", (MODEL_ID,))
        self.refresh()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=3)
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    def refresh(self):
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT name, embedding, samples, updated_at FROM people").fetchall()
        self.people = {name: {"embedding": normalize(json.loads(embedding)).tolist(),
                              "samples": samples, "updated_at": updated}
                       for name, embedding, samples, updated in rows}

    def _write(self, people):
        with closing(self._connect()) as connection, connection:
            # Delete only removed people: rebuilding the table would cascade every pose profile.
            existing = {r[0] for r in connection.execute('SELECT name FROM people')}
            for name in existing - people.keys():
                connection.execute('DELETE FROM people WHERE name=?', (name,))
            for name, person in people.items():
                if name not in self.people or person != self.people[name]:
                    connection.execute('DELETE FROM directional_samples WHERE person=?', (name,))
            connection.executemany("""INSERT INTO people VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET embedding=excluded.embedding,
                samples=excluded.samples, updated_at=excluded.updated_at""", [
                (name, json.dumps(person["embedding"], allow_nan=False),
                 person["samples"], person["updated_at"]) for name, person in people.items()])
        self.people = people

    def enroll_directional(self, name, samples, settings=None):
        settings = settings or load_guided_settings()
        name = name.strip()
        if not name or len(name) > 40:
            raise ValueError('이름은 1~40자여야 합니다.')
        if len(samples) != 15 or any(sorted(s.index for s in samples if s.direction == d) != [0, 1, 2]
                                     for d in DIRECTIONS):
            raise ValueError('정면·상·하·좌·우마다 정확히 3장씩 필요합니다.')
        rows, embeddings = [], []
        timestamp = datetime.now(timezone.utc).isoformat()
        for sample in samples:
            vector = normalize(sample.embedding)
            mesh = normalized_mesh(sample.landmarks, sample.size)
            if sample.model != MESH_MODEL or not np.allclose(mesh, sample.mesh, atol=1e-5):
                raise ValueError('등록 Mesh 형식 또는 모델이 다릅니다.')
            if not direction_matches(sample.direction, sample.pose, settings):
                raise ValueError('등록 방향과 촬영 각도가 일치하지 않습니다.')
            angles = [sample.pose.yaw, sample.pose.pitch, sample.pose.roll]
            if not np.isfinite(angles).all() or not sample.crop_jpeg:
                raise ValueError('등록 사진 또는 각도 데이터 오류')
            if embeddings and float(embeddings[0] @ vector) < .363:
                raise ValueError('등록 도중 다른 사람의 얼굴이 섞였습니다.')
            embeddings.append(vector)
            rows.append((name, sample.direction, sample.index,
                json.dumps(vector.tolist(), allow_nan=False),
                json.dumps(sample.landmarks.tolist(), allow_nan=False),
                json.dumps(mesh.tolist(), allow_nan=False), json.dumps(angles),
                json.dumps(list(sample.size)), sample.crop_jpeg, sample.model, timestamp))
        mean = normalize(np.mean(embeddings, axis=0))
        with closing(self._connect()) as connection, connection:
            connection.execute("""INSERT INTO people VALUES (?, ?, 15, ?)
                ON CONFLICT(name) DO UPDATE SET embedding=excluded.embedding,
                samples=15, updated_at=excluded.updated_at""", (name, json.dumps(mean.tolist()), timestamp))
            connection.execute('DELETE FROM directional_samples WHERE person=?', (name,))
            connection.executemany('INSERT INTO directional_samples VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)
        self.refresh()

    def directional_samples(self, name, direction=None, connection=None):
        if connection is None:
            with closing(self._connect()) as owned:
                return self.directional_samples(name, direction, owned)
        query = 'SELECT direction, sample_index, embedding, landmarks, mesh, pose, image_size, crop_jpeg, mesh_model FROM directional_samples WHERE person=?'
        params = [name]
        if direction is not None:
            query += ' AND direction=?'
            params.append(direction)
        return [dict(direction=d, index=i, embedding=normalize(json.loads(e)),
                     landmarks=np.asarray(json.loads(l), dtype=np.float32),
                     mesh=np.asarray(json.loads(m), dtype=np.float32), pose=json.loads(p),
                     size=tuple(json.loads(size)), crop_jpeg=jpeg, model=model)
                for d, i, e, l, m, p, size, jpeg, model in connection.execute(query, params)]

    def has_directional_profile(self, name, connection=None):
        if connection is None:
            with closing(self._connect()) as owned:
                return self.has_directional_profile(name, owned)
        counts = dict(connection.execute('SELECT direction, COUNT(*) FROM directional_samples '
            'WHERE person=? AND mesh_model=? GROUP BY direction', (name, MESH_MODEL)))
        return counts == {d: 3 for d in DIRECTIONS}

    def record_directional_pair(self, name, samples, source, threshold, settings):
        if len(samples) != 2 or len({s.direction for s in samples}) != 2:
            return None
        session = uuid.uuid4().hex
        proofs = []
        with closing(self._connect()) as connection, connection:
            # Matching and both writes see the same registration version.
            connection.execute('BEGIN IMMEDIATE')
            if not self.has_directional_profile(name, connection):
                return None
            for sample in samples:
                templates = self.directional_samples(name, sample.direction, connection)
                proof = compare_sample(sample, templates, threshold, settings)
                if proof is None:
                    return None
                proofs.append(proof)
            for sample, (score, error, depth) in zip(samples, proofs):
                connection.execute("""INSERT INTO observations
                    (captured_at,source,box,detection_score,crop_jpeg,embedding,gate_passed,
                     matched_name,match_score,match_threshold,direction,mesh_error,depth_error,auth_session)
                    VALUES (?,?,?,?,?,?,1,?,?,?,?,?,?,?)""",
                    (datetime.now(timezone.utc).isoformat(), source,
                     json.dumps([float(v) for v in sample.detection[:4]]), float(sample.detection[14]),
                     sample.crop_jpeg, json.dumps(normalize(sample.embedding).tolist()),
                     name, score, threshold, sample.direction, error, depth, session))
        return proofs

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
    """Legacy vectors only; new registration must use enroll_directional()."""
    store.enroll(name, samples)


def delete_user(store, name):
    store.delete(name)


def get_users(store):
    return list(store.people.keys())


def get_database_path():
    return DEFAULT_DATABASE_PATH
