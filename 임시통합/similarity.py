"""Similarity utilities for face authentication evaluation."""

import numpy as np


def normalize(vector):
    """L2-normalize an embedding vector."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        raise ValueError("빈 임베딩은 사용할 수 없습니다.")

    return vector / norm


def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    a = normalize(embedding1)
    b = normalize(embedding2)

    if a.shape != b.shape:
        raise ValueError("두 임베딩의 차원이 다릅니다.")

    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def is_match(score, threshold):
    """Return True when similarity score passes the threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold는 0~1 사이여야 합니다.")

    return score >= threshold
