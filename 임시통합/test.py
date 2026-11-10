"""Tests for face authentication performance evaluation."""

from evaluation import (
    calculate_far,
    calculate_frr,
    evaluate_thresholds,
    print_results,
)


def test_far():

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

    assert far == 0.0

    print("FAR test passed")


def test_frr():

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

    assert frr == 0.0

    print("FRR test passed")


def test_threshold_evaluation():

    genuine_scores = [
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    impostor_scores = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ]

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

    print()
    print("Threshold performance evaluation")
    print()

    print_results(results)

    assert len(results) == len(thresholds)

    print()
    print("Threshold evaluation test passed")


if __name__ == "__main__":

    test_far()
    test_frr()
    test_threshold_evaluation()

    print()
    print("All tests passed.")
