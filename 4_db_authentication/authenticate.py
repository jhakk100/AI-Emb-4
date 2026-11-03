from similarity import cosine_similarity
from database import get_all_users, validate_embedding


def authenticate_face(current_embedding, threshold):
    """
    현재 얼굴 임베딩과 등록된 사용자 DB를 비교하여 인증합니다.

    Args:
        current_embedding: 현재 카메라에서 추출한 얼굴 임베딩
        threshold: 인증 기준 유사도

    Returns:
        인증 결과를 담은 딕셔너리
    """

    # 현재 얼굴 임베딩 검사
    try:
        current_embedding = validate_embedding(current_embedding)
    except ValueError as e:
        return {
            "authenticated": False,
            "user_id": None,
            "similarity": None,
            "message": str(e)
        }

    # Threshold 범위 검사
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Threshold는 0부터 1 사이의 숫자여야 합니다.")

    # 등록된 사용자 DB 조회
    users = get_all_users()

    if not users:
        return {
            "authenticated": False,
            "user_id": None,
            "similarity": None,
            "message": "등록된 사용자가 없습니다."
        }

    best_user = None
    best_similarity = -1.0

    # 모든 등록 사용자와 유사도 비교
    for user in users:

        try:
            similarity = cosine_similarity(
                current_embedding,
                user["embedding"]
            )

        except ValueError:
            # 임베딩 길이가 다르거나 잘못된 데이터는 비교하지 않음
            continue

        # 가장 높은 유사도 찾기
        if similarity > best_similarity:
            best_similarity = similarity
            best_user = user["user_id"]

    # 비교 가능한 사용자가 없는 경우
    if best_user is None:
        return {
            "authenticated": False,
            "user_id": None,
            "similarity": None,
            "message": "비교할 수 있는 임베딩 데이터가 없습니다."
        }

    # Threshold 기준으로 인증 판단
    if best_similarity >= threshold:
        return {
            "authenticated": True,
            "user_id": best_user,
            "similarity": best_similarity,
            "message": "인증 성공"
        }

    return {
        "authenticated": False,
        "user_id": None,
        "similarity": best_similarity,
        "message": "등록된 사용자를 찾을 수 없습니다."
    }
