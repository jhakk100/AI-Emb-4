import math


def cosine_similarity(embedding1, embedding2):
    """
    두 얼굴 임베딩 사이의 코사인 유사도를 계산합니다.

    반환값:
        1에 가까울수록 유사함
        0에 가까울수록 관련성이 낮음
        -1에 가까울수록 반대 방향
    """

    # 입력 데이터 확인
    if not embedding1 or not embedding2:
        raise ValueError("임베딩 데이터가 비어 있습니다.")

    if len(embedding1) != len(embedding2):
        raise ValueError("두 임베딩의 길이가 다릅니다.")

    # 숫자 데이터인지 확인하고 실수형으로 변환
    try:
        embedding1 = [float(x) for x in embedding1]
        embedding2 = [float(x) for x in embedding2]
    except (TypeError, ValueError):
        raise ValueError("임베딩은 숫자 데이터여야 합니다.")

    # 무한대 또는 NaN 값 확인
    if not all(math.isfinite(x) for x in embedding1 + embedding2):
        raise ValueError("임베딩에 유효하지 않은 숫자가 있습니다.")

    # 벡터의 내적 계산
    dot_product = sum(
        a * b for a, b in zip(embedding1, embedding2)
    )

    # 각 벡터의 크기 계산
    norm1 = math.sqrt(sum(a * a for a in embedding1))
    norm2 = math.sqrt(sum(b * b for b in embedding2))

    # 0으로 나누는 오류 방지
    if norm1 == 0 or norm2 == 0:
        raise ValueError("영벡터는 비교할 수 없습니다.")

    # 코사인 유사도 계산
    similarity = dot_product / (norm1 * norm2)

    return similarity
