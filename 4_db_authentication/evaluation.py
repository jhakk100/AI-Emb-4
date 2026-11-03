
import math


def calculate_far(impostor_scores, threshold):
    """
    FAR(False Acceptance Rate) 계산

    등록되지 않은 사람을 잘못 인증한 비율
    """

    if not impostor_scores:
        raise ValueError("타인 비교 데이터가 없습니다.")

    false_accepts = sum(
        score >= threshold for score in impostor_scores
    )

    return false_accepts / len(impostor_scores)


def calculate_frr(genuine_scores, threshold):
    """
    FRR(False Rejection Rate) 계산

    등록된 사용자를 잘못 거부한 비율
    """

    if not genuine_scores:
        raise ValueError("본인 비교 데이터가 없습니다.")

    false_rejects = sum(
        score < threshold for score in genuine_scores
    )

    return false_rejects / len(genuine_scores)


def evaluate_authentication(genuine_scores, impostor_scores, threshold):
    """
    지정한 Threshold에서 FAR과 FRR을 계산합니다.

    genuine_scores:
        동일인의 얼굴 임베딩 비교 점수

    impostor_scores:
        서로 다른 사람의 얼굴 임베딩 비교 점수

    threshold:
        인증 기준값
    """

    validate_scores(genuine_scores)
    validate_scores(impostor_scores)

    far = calculate_far(impostor_scores, threshold)
    frr = calculate_frr(genuine_scores, threshold)

    return {
        "threshold": threshold,
        "FAR": far,
        "FRR": frr
    }


def validate_scores(scores):
    """유사도 데이터의 유효성을 검사합니다."""

    if not scores:
        raise ValueError("평가할 유사도 데이터가 없습니다.")

    for score in scores:
        if not isinstance(score, (int, float)):
            raise ValueError("유사도는 숫자여야 합니다.")

        if not math.isfinite(score):
            raise ValueError("유효하지 않은 유사도 값입니다.")


def calculate_eer(genuine_scores, impostor_scores):
    """
    EER(Equal Error Rate)을 계산합니다.

    FAR과 FRR의 차이가 가장 작은 Threshold를 선택합니다.
    이 방식은 유한한 테스트 데이터에서 EER을 근사합니다.
    """

    validate_scores(genuine_scores)
    validate_scores(impostor_scores)

    all_scores = genuine_scores + impostor_scores

    # 모든 유사도 점수를 정렬
    sorted_scores = sorted(set(all_scores))

    # Threshold 후보 생성
    thresholds = [
        sorted_scores[0] - 0.000001
    ]

    for i in range(len(sorted_scores) - 1):
        midpoint = (
            sorted_scores[i] + sorted_scores[i + 1]
        ) / 2

        thresholds.append(midpoint)

    thresholds.append(sorted_scores[-1] + 0.000001)

    best_result = None
    min_difference = float("inf")

    # 모든 Threshold에서 FAR과 FRR 비교
    for threshold in thresholds:

        far = calculate_far(impostor_scores, threshold)
        frr = calculate_frr(genuine_scores, threshold)

        difference = abs(far - frr)

        if difference < min_difference:
            min_difference = difference

            best_result = {
                "EER": (far + frr) / 2,
                "threshold": threshold,
                "FAR": far,
                "FRR": frr
            }

    return best_result


def print_evaluation_result(result):
    """평가 결과를 보기 쉽게 출력합니다."""

    print("\n========== Authentication Evaluation ==========")

    print(f"Threshold : {result['threshold']:.4f}")
    print(f"FAR       : {result['FAR'] * 100:.2f}%")
    print(f"FRR       : {result['FRR'] * 100:.2f}%")

    if "EER" in result:
        print(f"EER       : {result['EER'] * 100:.2f}%")

    print("================================================")
