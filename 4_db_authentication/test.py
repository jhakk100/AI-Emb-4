"""Tests for threshold-based face authentication performance evaluation."""

from evaluation import (
    calculate_far,
    calculate_frr,
    evaluate_thresholds,
    find_eer,
    print_results,
    plot_far_frr,
)


def test_far():
    """Test False Acceptance Rate."""

    # 다른 사람들의 유사도 점수
    impostor_scores = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ]

    threshold = 0.45

    far = calculate_far(
        impostor_scores,
        threshold
    )

    print("[테스트 1] FAR 계산")
    print(f"Threshold: {threshold}")
    print(f"FAR: {far:.2%}")

    assert far == 0.0

    print("결과: 성공\n")


def test_frr():
    """Test False Rejection Rate."""

    # 본인 얼굴들의 유사도 점수
    genuine_scores = [
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    threshold = 0.45

    frr = calculate_frr(
        genuine_scores,
        threshold
    )

    print("[테스트 2] FRR 계산")
    print(f"Threshold: {threshold}")
    print(f"FRR: {frr:.2%}")

    assert frr == 0.0

    print("결과: 성공\n")


def test_threshold_evaluation():
    """Test FAR/FRR across multiple thresholds."""

    # 본인 얼굴 유사도 점수
    genuine_scores = [
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    # 다른 사람 얼굴 유사도 점수
    impostor_scores = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ]

    # 여러 threshold에서 성능 비교
    thresholds = [
        0.30,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
    ]

    results = evaluate_thresholds(
        genuine_scores,
        impostor_scores,
        thresholds
    )

    print("[테스트 3] Threshold별 성능 평가")
    print()

    print_results(results)

    assert len(results) == len(thresholds)

    print("\n결과: 성공\n")

    # FAR / FRR 그래프 출력
    plot_far_frr(results)


def test_eer():
    """Test EER approximation."""

    # EER 계산을 확인하기 위한 테스트 데이터
    genuine_scores = [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    impostor_scores = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
    ]

    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ]

    results = evaluate_thresholds(
        genuine_scores,
        impostor_scores,
        thresholds
    )

    eer_result = find_eer(results)

    print("[테스트 4] EER 계산")

    assert eer_result is not None

    eer = (
        eer_result.far +
        eer_result.frr
    ) / 2

    print(
        f"EER Threshold: "
        f"{eer_result.threshold:.2f}"
    )

    print(
        f"EER: "
        f"{eer:.2%}"
    )

    print("결과: 성공\n")


if __name__ == "__main__":

    test_far()
    test_frr()
    test_threshold_evaluation()
    test_eer()

    print(
        "===== 모든 성능 평가 테스트 완료 ====="
    )
