"""Threshold-based face authentication performance evaluation."""

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    threshold: float
    far: float
    frr: float


def calculate_far(impostor_scores, threshold):
    """
    Calculate False Acceptance Rate (FAR).

    FAR:
        다른 사람의 얼굴을 등록된 사용자로 잘못 인증한 비율.
    """
    if not impostor_scores:
        return 0.0

    false_accepts = sum(
        score >= threshold
        for score in impostor_scores
    )

    return false_accepts / len(impostor_scores)


def calculate_frr(genuine_scores, threshold):
    """
    Calculate False Rejection Rate (FRR).

    FRR:
        정상 사용자의 얼굴을 잘못 거부한 비율.
    """
    if not genuine_scores:
        return 0.0

    false_rejects = sum(
        score < threshold
        for score in genuine_scores
    )

    return false_rejects / len(genuine_scores)


def evaluate_threshold(
    genuine_scores,
    impostor_scores,
    threshold
):
    """Evaluate FAR and FRR at one threshold."""

    far = calculate_far(
        impostor_scores,
        threshold
    )

    frr = calculate_frr(
        genuine_scores,
        threshold
    )

    return EvaluationResult(
        threshold=threshold,
        far=far,
        frr=frr
    )


def evaluate_thresholds(
    genuine_scores,
    impostor_scores,
    thresholds
):
    """Evaluate FAR and FRR over multiple thresholds."""

    results = []

    for threshold in thresholds:
        result = evaluate_threshold(
            genuine_scores,
            impostor_scores,
            threshold
        )

        results.append(result)

    return results


def find_eer(results):
    """
    Find the threshold where FAR and FRR are closest.

    This is a discrete approximation of EER.
    """

    if not results:
        return None

    return min(
        results,
        key=lambda result: abs(result.far - result.frr)
    )


def print_results(results):
    """Print threshold evaluation results."""

    print(
        f"{'Threshold':>10} "
        f"{'FAR':>10} "
        f"{'FRR':>10}"
    )

    print("-" * 34)

    for result in results:
        print(
            f"{result.threshold:>10.2f} "
            f"{result.far * 100:>9.2f}% "
            f"{result.frr * 100:>9.2f}%"
        )

    eer_result = find_eer(results)

    if eer_result is not None:
        eer = (eer_result.far + eer_result.frr) / 2

        print()
        print("EER approximation")
        print(f"Threshold : {eer_result.threshold:.2f}")
        print(f"EER       : {eer * 100:.2f}%")
