import sqlite3
import json
import math
from pathlib import Path


# 데이터베이스 파일 경로
DB_PATH = Path(__file__).parent / "face_database.db"


def connect_db():
    """데이터베이스 연결"""
    return sqlite3.connect(DB_PATH)


def initialize_database():
    """사용자 임베딩 데이터를 저장할 테이블 생성"""

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            embedding TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def validate_embedding(embedding):
    """임베딩 데이터 유효성 검사"""

    if not isinstance(embedding, (list, tuple)) or len(embedding) == 0:
        raise ValueError("임베딩 데이터가 비어 있거나 올바르지 않습니다.")

    try:
        values = [float(x) for x in embedding]
    except (TypeError, ValueError):
        raise ValueError("임베딩은 숫자 데이터여야 합니다.")

    if not all(math.isfinite(x) for x in values):
        raise ValueError("임베딩에 유효하지 않은 숫자가 있습니다.")

    return values


def add_user(user_id, embedding):
    """새로운 사용자의 얼굴 임베딩 등록"""

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("사용자 ID가 올바르지 않습니다.")

    embedding = validate_embedding(embedding)

    conn = connect_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (user_id, embedding) VALUES (?, ?)",
            (user_id.strip(), json.dumps(embedding))
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        print("이미 등록된 사용자입니다.")
        return False

    finally:
        conn.close()


def get_all_users():
    """등록된 모든 사용자의 임베딩 데이터 조회"""

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, embedding FROM users")
    rows = cursor.fetchall()

    conn.close()

    users = []

    for user_id, embedding in rows:
        users.append({
            "user_id": user_id,
            "embedding": json.loads(embedding)
        })

    return users


def delete_user(user_id):
    """등록된 사용자 삭제"""

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM users WHERE user_id = ?",
        (user_id,)
    )

    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


# 파일을 직접 실행했을 때 데이터베이스 초기화
if __name__ == "__main__":
    initialize_database()
    print("얼굴 임베딩 데이터베이스가 준비되었습니다.")
