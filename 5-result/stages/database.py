"""Stage 4: separate enrollment and observation tables in one SQLite DB."""

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid
from types import SimpleNamespace

import cv2
import numpy as np

from detection import crop_face
from auth_calibration import AuthenticationCalibration
from face_core import DEFAULT_THRESHOLD, FaceStore, MODEL_ID, normalize, REGISTRATION_SAMPLES
from models import ROOT
from guided_capture import (DIRECTIONS, MANUAL_POSE_BASIS, MESH_MODEL, compare_sample, enrollment_sample_direction_valid,
                            load_guided_settings, normalized_mesh, template_basis_matches)
from liveness import Pose
from face_grid import (GRID_MODEL, FUSED_MODEL, TOPOLOGY, build_grids, compare_grid,
                       fuse_grids, compare_fused)

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
                CREATE TABLE IF NOT EXISTS face_grids (
                    person TEXT NOT NULL REFERENCES people(name) ON DELETE CASCADE,
                    direction TEXT NOT NULL, vertices TEXT NOT NULL, embedding TEXT NOT NULL,
                    sample_count INTEGER NOT NULL CHECK(sample_count=3),
                    grid_model TEXT NOT NULL, mesh_model TEXT NOT NULL, pose_basis TEXT,
                    updated_at TEXT NOT NULL, PRIMARY KEY(person,direction)
                );
                CREATE TABLE IF NOT EXISTS face_models (
                    person TEXT PRIMARY KEY REFERENCES people(name) ON DELETE CASCADE,
                    vertices TEXT NOT NULL, embedding TEXT NOT NULL, directions INTEGER NOT NULL CHECK(directions IN (4,5)),
                    model TEXT NOT NULL, mesh_model TEXT NOT NULL, topology TEXT NOT NULL,
                    pose_basis TEXT, source_errors TEXT NOT NULL, updated_at TEXT NOT NULL
                );
            """)
            self._migrate_model_schema(connection)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(observations)")}
            for name, kind in (("matched_name", "TEXT"), ("match_score", "REAL"), ("match_threshold", "REAL"),
                               ("direction", "TEXT"), ("mesh_error", "REAL"), ("depth_error", "REAL"),
                               ("auth_session", "TEXT"), ("grid_model", "TEXT"), ("grid_error", "REAL"),
                               ("grid_depth_error", "REAL"), ("fused_model", "TEXT"),
                               ("fused_error", "REAL"), ("fused_depth_error", "REAL")):
                if name not in columns:
                    connection.execute(f"ALTER TABLE observations ADD COLUMN {name} {kind}")
            columns = {row[1] for row in connection.execute('PRAGMA table_info(directional_samples)')}
            if 'pose_basis' not in columns:
                connection.execute('ALTER TABLE directional_samples ADD COLUMN pose_basis TEXT')
            if 'capture_mode' not in columns:
                connection.execute("ALTER TABLE directional_samples ADD COLUMN capture_mode TEXT NOT NULL DEFAULT 'auto'")
            existing = connection.execute("SELECT value FROM metadata WHERE key='model'").fetchone()
            if existing is not None and existing[0] != MODEL_ID:
                raise ValueError("DB의 임베딩 모델이 현재 모델과 다릅니다.")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('model', ?)", (MODEL_ID,))
            self._migrate_grids(connection)
        self.refresh()
        self.calibration = AuthenticationCalibration(self._connect)

    @staticmethod
    def _migrate_model_schema(connection):
        schema = connection.execute("SELECT sql FROM sqlite_master WHERE name='face_models'").fetchone()[0]
        if 'CHECK(DIRECTIONS=5)' not in ''.join(schema.upper().split()):
            return
        # Preserve existing five-pose models while allowing new four-pose models.
        connection.execute('SAVEPOINT four_pose_model_schema')
        try:
            connection.execute('''CREATE TABLE face_models_four_pose (
                person TEXT PRIMARY KEY REFERENCES people(name) ON DELETE CASCADE,
                vertices TEXT NOT NULL, embedding TEXT NOT NULL,
                directions INTEGER NOT NULL CHECK(directions IN (4,5)),
                model TEXT NOT NULL, mesh_model TEXT NOT NULL, topology TEXT NOT NULL,
                pose_basis TEXT, source_errors TEXT NOT NULL, updated_at TEXT NOT NULL)''')
            connection.execute('INSERT INTO face_models_four_pose SELECT * FROM face_models')
            connection.execute('DROP TABLE face_models')
            connection.execute('ALTER TABLE face_models_four_pose RENAME TO face_models')
        except Exception:
            connection.execute('ROLLBACK TO four_pose_model_schema')
            raise
        finally:
            connection.execute('RELEASE four_pose_model_schema')

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
                    connection.execute('DELETE FROM face_grids WHERE person=?', (name,))
                    connection.execute('DELETE FROM face_models WHERE person=?', (name,))
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
        if len(samples) != REGISTRATION_SAMPLES or any(sorted(s.index for s in samples if s.direction == d) != [0, 1, 2]
                                     for d in DIRECTIONS):
            raise ValueError('정면·위·좌·우마다 정확히 3장씩 필요합니다.')
        rows, embeddings = [], []
        timestamp = datetime.now(timezone.utc).isoformat()
        for sample in samples:
            vector = normalize(sample.embedding)
            mesh = normalized_mesh(sample.landmarks, sample.size)
            if sample.model != MESH_MODEL or not np.allclose(mesh, sample.mesh, atol=1e-5):
                raise ValueError('등록 Mesh 형식 또는 모델이 다릅니다.')
            if not enrollment_sample_direction_valid(sample, settings):
                raise ValueError('등록 방향과 촬영 각도가 일치하지 않습니다.')
            angles = None if sample.pose is None else [sample.pose.yaw, sample.pose.pitch, sample.pose.roll]
            if (angles is not None and not np.isfinite(angles).all()) or not sample.crop_jpeg:
                raise ValueError('등록 사진 또는 각도 데이터 오류')
            if sample.capture_mode not in ('auto', 'manual'):
                raise ValueError('등록 촬영 방식이 올바르지 않습니다.')
            if embeddings and float(embeddings[0] @ vector) < .363:
                raise ValueError('등록 도중 다른 사람의 얼굴이 섞였습니다.')
            embeddings.append(vector)
            rows.append((name, sample.direction, sample.index,
                json.dumps(vector.tolist(), allow_nan=False),
                json.dumps(sample.landmarks.tolist(), allow_nan=False),
                json.dumps(mesh.tolist(), allow_nan=False), json.dumps(angles),
                json.dumps(list(sample.size)), sample.crop_jpeg, sample.model, timestamp,
                json.dumps(sample.pose_basis, allow_nan=False) if sample.pose_basis is not None else None,
                sample.capture_mode))
        grids = build_grids(samples, settings)
        fused = fuse_grids(grids)
        mean = normalize(np.mean(embeddings, axis=0))
        with closing(self._connect()) as connection, connection:
            connection.execute("""INSERT INTO people VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET embedding=excluded.embedding,
                samples=excluded.samples, updated_at=excluded.updated_at""",
                (name, json.dumps(mean.tolist()), len(samples), timestamp))
            connection.execute('DELETE FROM directional_samples WHERE person=?', (name,))
            connection.execute('DELETE FROM face_grids WHERE person=?', (name,))
            connection.execute('DELETE FROM face_models WHERE person=?', (name,))
            connection.executemany('''INSERT INTO directional_samples
                (person,direction,sample_index,embedding,landmarks,mesh,pose,image_size,crop_jpeg,
                 mesh_model,captured_at,pose_basis,capture_mode) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', rows)
            self._write_grids(connection, name, grids, timestamp)
            self._write_model(connection, name, fused, timestamp)
        self.refresh()

    def directional_samples(self, name, direction=None, connection=None):
        if connection is None:
            with closing(self._connect()) as owned:
                return self.directional_samples(name, direction, owned)
        query = 'SELECT direction, sample_index, embedding, landmarks, mesh, pose, image_size, crop_jpeg, mesh_model, pose_basis, capture_mode FROM directional_samples WHERE person=?'
        params = [name]
        if direction is not None:
            query += ' AND direction=?'
            params.append(direction)
        return [dict(direction=d, index=i, embedding=normalize(json.loads(e)),
                     landmarks=np.asarray(json.loads(l), dtype=np.float32),
                     mesh=np.asarray(json.loads(m), dtype=np.float32), pose=json.loads(p),
                     size=tuple(json.loads(size)), crop_jpeg=jpeg, model=model,
                     pose_basis=json.loads(basis) if basis is not None else None, capture_mode=mode)
                for d, i, e, l, m, p, size, jpeg, model, basis, mode in connection.execute(query, params)]

    def _write_grids(self, connection, name, grids, timestamp):
        connection.executemany('INSERT INTO face_grids VALUES (?,?,?,?,?,?,?,?,?)', [
            (name, direction, json.dumps(grid['vertices'].tolist(), allow_nan=False),
             json.dumps(grid['embedding'].tolist(), allow_nan=False), grid['samples'], GRID_MODEL, MESH_MODEL,
             json.dumps(grid['pose_basis'], allow_nan=False) if grid['pose_basis'] is not None else None, timestamp)
            for direction, grid in grids.items()])

    def _migrate_grids(self, connection):
        settings = load_guided_settings()
        people = [row[0] for row in connection.execute('SELECT name FROM people')]
        for name in people:
            if self.has_grid_profile(name, connection):
                if not self.has_fused_model(name, connection):
                    try:
                        model = fuse_grids(self.face_grids(name, connection))
                    except (ValueError, KeyError, np.linalg.LinAlgError):
                        continue
                    self._write_model(connection, name, model, datetime.now(timezone.utc).isoformat())
                continue
            if not self.has_directional_profile(name, connection):
                continue
            try:
                samples = []
                for stored in self.directional_samples(name, connection=connection):
                    if stored['direction'] not in DIRECTIONS:
                        continue  # Retain old down photos, but never use them for authentication.
                    item = SimpleNamespace(**stored)
                    item.pose = None if stored['pose'] is None else Pose(*stored['pose'], (0, 0, 1, 1))
                    if not np.allclose(normalized_mesh(item.landmarks, item.size), item.mesh, atol=1e-5):
                        raise ValueError('저장된 Mesh가 일치하지 않습니다.')
                    samples.append(item)
                grids = build_grids(samples, settings)
                model = fuse_grids(grids)
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
                continue  # Preserve invalid/legacy records; authentication requests re-enrollment.
            connection.execute('DELETE FROM face_grids WHERE person=?', (name,))
            self._write_grids(connection, name, grids, datetime.now(timezone.utc).isoformat())
            self._write_model(connection, name, model, datetime.now(timezone.utc).isoformat())

    def _write_model(self, connection, name, model, timestamp):
        connection.execute('''INSERT INTO face_models VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(person) DO UPDATE SET vertices=excluded.vertices,embedding=excluded.embedding,
            directions=excluded.directions,model=excluded.model,mesh_model=excluded.mesh_model,
            topology=excluded.topology,pose_basis=excluded.pose_basis,source_errors=excluded.source_errors,
            updated_at=excluded.updated_at''', (
            name, json.dumps(model['vertices'].tolist(), allow_nan=False),
            json.dumps(model['embedding'].tolist(), allow_nan=False), model['directions'],
            model['model'], model['mesh_model'], model['topology'],
            json.dumps(model['pose_basis'], allow_nan=False) if model['pose_basis'] is not None else None,
            json.dumps(model['source_errors'], allow_nan=False), timestamp))

    def face_model(self, name, connection=None):
        if connection is None:
            with closing(self._connect()) as owned:
                return self.face_model(name, owned)
        row = connection.execute('SELECT vertices,embedding,directions,model,mesh_model,topology,pose_basis,source_errors '
                                 'FROM face_models WHERE person=?', (name,)).fetchone()
        if row is None:
            return None
        vertices, embedding, directions, model, mesh_model, topology, basis, errors = row
        vertices = np.asarray(json.loads(vertices), dtype=np.float32)
        if vertices.shape != (468, 3) or not np.isfinite(vertices).all():
            raise ValueError('저장된 통합 3D 얼굴 모델이 올바르지 않습니다. 다시 등록하세요.')
        return dict(vertices=vertices, embedding=normalize(json.loads(embedding)), directions=directions,
                    model=model, mesh_model=mesh_model, topology=topology,
                    pose_basis=json.loads(basis) if basis else None, source_errors=json.loads(errors))

    def has_fused_model(self, name, connection=None):
        model = self.face_model(name, connection)
        return (model is not None and model['model'] == FUSED_MODEL and model['mesh_model'] == MESH_MODEL
                and model['directions'] == len(DIRECTIONS) and model['topology'] == TOPOLOGY)

    def manual_auth_names(self):
        with closing(self._connect()) as connection:
            return {name for name, basis in connection.execute(
                'SELECT person,pose_basis FROM face_models WHERE model=? AND directions=?',
                (FUSED_MODEL, len(DIRECTIONS))) if basis and json.loads(basis) == MANUAL_POSE_BASIS}

    def grid_basis_matches(self, name, basis):
        grids, model = self.face_grids(name), self.face_model(name)
        return (set(grids) == set(DIRECTIONS) and model is not None
                and template_basis_matches(model['pose_basis'], basis)
                and all(grid['pose_basis'] == model['pose_basis'] for grid in grids.values()))

    def compare_face_model(self, name, sample, threshold, settings, trace=None):
        return compare_fused(sample, self.face_model(name), threshold, settings, trace)

    def face_grids(self, name, connection=None):
        if connection is None:
            with closing(self._connect()) as owned:
                return self.face_grids(name, owned)
        result = {}
        for direction, vertices, embedding, count, grid_model, mesh_model, basis in connection.execute(
                'SELECT direction,vertices,embedding,sample_count,grid_model,mesh_model,pose_basis '
                'FROM face_grids WHERE person=?', (name,)):
            mesh = np.asarray(json.loads(vertices), dtype=np.float32)
            if mesh.shape != (468, 3) or not np.isfinite(mesh).all():
                raise ValueError('저장된 얼굴 grid가 올바르지 않습니다. 다시 등록하세요.')
            result[direction] = dict(direction=direction, vertices=mesh, embedding=normalize(json.loads(embedding)),
                                     samples=count, grid_model=grid_model, mesh_model=mesh_model,
                                     pose_basis=json.loads(basis) if basis else None)
        return result

    def has_grid_profile(self, name, connection=None):
        grids = self.face_grids(name, connection)
        return (set(grids) == set(DIRECTIONS) and all(grid['samples'] == 3 and grid['grid_model'] == GRID_MODEL
                and grid['mesh_model'] == MESH_MODEL for grid in grids.values()))

    def compare_face_grid(self, name, sample, threshold, settings, trace=None):
        grids = self.face_grids(name)
        if set(grids) != set(DIRECTIONS):
            return None
        return compare_grid(sample, grids.get(sample.direction), threshold, settings, trace)

    def has_directional_profile(self, name, connection=None):
        if connection is None:
            with closing(self._connect()) as owned:
                return self.has_directional_profile(name, owned)
        counts = dict(connection.execute('SELECT direction, COUNT(*) FROM directional_samples '
            'WHERE person=? AND mesh_model=? GROUP BY direction', (name, MESH_MODEL)))
        return all(counts.get(direction) == 3 for direction in DIRECTIONS)

    def record_directional_pair(self, name, samples, source, threshold, settings):
        if len(samples) != 2 or len({s.direction for s in samples}) != 2:
            return None
        session = uuid.uuid4().hex
        proofs = []
        with closing(self._connect()) as connection, connection:
            # Matching and both writes see the same registration version.
            connection.execute('BEGIN IMMEDIATE')
            if (not self.has_directional_profile(name, connection) or not self.has_grid_profile(name, connection)
                    or not self.has_fused_model(name, connection)):
                return None
            grids = self.face_grids(name, connection)
            model = self.face_model(name, connection)
            grid_proofs = []
            fused_proofs = []
            for sample in samples:
                templates = self.directional_samples(name, sample.direction, connection)
                proof = compare_sample(sample, templates, threshold, settings)
                if proof is None:
                    return None
                grid_proof = compare_grid(sample, grids.get(sample.direction), threshold, settings)
                if grid_proof is None:
                    return None
                fused_proof = compare_fused(sample, model, threshold, settings)
                if fused_proof is None:
                    return None
                grid_proofs.append(grid_proof)
                fused_proofs.append(fused_proof)
                proofs.append((min(proof[0], grid_proof[0], fused_proof[0]), max(proof[1], grid_proof[1], fused_proof[1]),
                               max(proof[2], grid_proof[2], fused_proof[2])))
            for sample, (score, error, depth), grid_proof, fused_proof in zip(samples, proofs, grid_proofs, fused_proofs):
                connection.execute("""INSERT INTO observations
                    (captured_at,source,box,detection_score,crop_jpeg,embedding,gate_passed,
                     matched_name,match_score,match_threshold,direction,mesh_error,depth_error,auth_session,
                     grid_model,grid_error,grid_depth_error,fused_model,fused_error,fused_depth_error)
                    VALUES (?,?,?,?,?,?,1,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (datetime.now(timezone.utc).isoformat(), source,
                     json.dumps([float(v) for v in sample.detection[:4]]), float(sample.detection[14]),
                     sample.crop_jpeg, json.dumps(normalize(sample.embedding).tolist()),
                     name, score, threshold, sample.direction, error, depth, session, GRID_MODEL, grid_proof[1], grid_proof[2],
                     FUSED_MODEL, fused_proof[1], fused_proof[2]))
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
