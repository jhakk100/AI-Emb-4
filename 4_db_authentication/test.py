from database import (
    initialize_database,
    add_user,
    delete_user
)

from similarity import cosine_similarity
from authenticate import authenticate_face


def test_similarity():
    """코사인 유사도 계산 테스트"""

    face1 = [1.0, 0.0, 0.0]
    face2 = [1.0, 0.0, 0.0]

    similarity = cosine_similarity(face1, face2)

    print("\n[테스트 1] 유사도 계산")
    print("유사도:", similarity)

    assert abs(similarity - 1.0) < 1e-6

    print("결과: 성공")


def test_authentication():
    """얼굴 인증 기능 테스트"""

    # 테스트용 가상 사용자 데이터
    student_a = [1.0, 0.0, 0.0]
    student_b = [0.0, 1.0, 0.0]

    test_ids = ["test_student_A", "test_student_B"]

    # 테스트 데이터 중복 방지
    for user_id in test_ids:
        delete_user(user_id)

    # 테스트 사용자 등록
    add_user(test_ids[0], student_a)
    add_user(test_ids[1], student_b)

    print("\n[테스트 2] 등록 사용자 인증")

    result = authenticate_face(
        current_embedding=student_a,
        threshold=0.9
    )

    print("인증 결과:", result)

    assert result["authenticated"] is True
    assert result["user_id"] == "test_student_A"

    print("결과: 성공")

    print("\n[테스트 3] 미등록 사용자 인증 거부")

    unknown_face = [0.0, 0.0, 1.0]

    result = authenticate_face(
        current_embedding=unknown_face,
        threshold=0.9
    )

    print("인증 결과:", result)

    assert result["authenticated"] is False

    print("결과: 성공")

    # 테스트 데이터 삭제
    for user_id in test_ids:
        delete_user(user_id)


def main():
    """전체 테스트 실행"""

    print("===== 얼굴 인증 시스템 테스트 시작 =====")

    initialize_database()

    test_similarity()
    test_authentication()

    print("\n===== 모든 테스트 완료 =====")


if __name__ == "__main__":
    main()
