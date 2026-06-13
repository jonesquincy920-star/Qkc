"""behavioral_drift's profile mean is the threat class closest to nominal's,
making it the highest-risk class for false positives if telemetry noise
grows beyond the calibrated 0.15 std. These tests lock in that margin so a
future profile change can't silently erode it."""

import numpy as np

from eamis import FEATURE_NAMES, THREAT_CLASSES, classify, default_class_profiles

THREAT_SIGNATURES = [c for c in THREAT_CLASSES if c != "nominal"]
CALIBRATED_NOISE_STD = 0.15
STRESS_NOISE_STD = 0.75


def _margin(profiles, threat_class):
    return float(np.linalg.norm(profiles[threat_class].means - profiles["nominal"].means))


def test_behavioral_drift_is_nearest_threat_to_nominal():
    profiles = default_class_profiles()
    drift_margin = _margin(profiles, "behavioral_drift")
    other_margins = [_margin(profiles, c) for c in THREAT_SIGNATURES
                      if c != "behavioral_drift"]
    assert drift_margin < min(other_margins)


def test_behavioral_drift_margin_exceeds_calibrated_noise_floor():
    profiles = default_class_profiles()
    drift_margin = _margin(profiles, "behavioral_drift")
    assert drift_margin / CALIBRATED_NOISE_STD > 5.0


def test_behavioral_drift_false_positive_rate_zero_at_calibrated_noise():
    rng = np.random.default_rng(7)
    profiles = default_class_profiles()
    nominal_mean = profiles["nominal"].means
    false_positives = sum(
        max(classify(nominal_mean + rng.normal(0.0, CALIBRATED_NOISE_STD,
                                                size=len(FEATURE_NAMES)), profiles
                      ).items(), key=lambda kv: kv[1])[0] == "behavioral_drift"
        for _ in range(1000)
    )
    assert false_positives == 0


def test_behavioral_drift_false_positive_rate_bounded_under_noise_stress():
    rng = np.random.default_rng(8)
    profiles = default_class_profiles()
    nominal_mean = profiles["nominal"].means
    false_positives = sum(
        max(classify(nominal_mean + rng.normal(0.0, STRESS_NOISE_STD,
                                                size=len(FEATURE_NAMES)), profiles
                      ).items(), key=lambda kv: kv[1])[0] == "behavioral_drift"
        for _ in range(1000)
    )
    assert false_positives / 1000 < 0.05
