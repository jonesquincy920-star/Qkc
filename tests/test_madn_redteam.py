"""Adversarial red-team check: every non-nominal threat-class signature
should be classified as something other than nominal (i.e. flagged), and
classified as its own class under low-noise observation. Mirrors the
proposal's claim of 7/7 adversarial detection."""

import numpy as np
import pytest

from orrerynav.madn import THREAT_CLASSES, default_class_profiles, monte_carlo_classify

THREAT_SIGNATURES = [name for name in THREAT_CLASSES if name != "nominal"]


@pytest.mark.parametrize("threat_class", THREAT_SIGNATURES)
def test_redteam_signature_is_flagged_as_non_nominal(threat_class):
    rng = np.random.default_rng(hash(threat_class) % (2**32))
    profile = default_class_profiles()[threat_class]
    noise = np.full(profile.means.shape, 0.15)

    result = monte_carlo_classify(profile.means, noise, n_trials=300, rng=rng)
    assert result["predicted_class"] != "nominal"


@pytest.mark.parametrize("threat_class", THREAT_SIGNATURES)
def test_redteam_signature_is_correctly_identified(threat_class):
    rng = np.random.default_rng((hash(threat_class) + 1) % (2**32))
    profile = default_class_profiles()[threat_class]
    noise = np.full(profile.means.shape, 0.15)

    result = monte_carlo_classify(profile.means, noise, n_trials=300, rng=rng)
    assert result["predicted_class"] == threat_class


def test_redteam_full_sweep_detection_rate_is_perfect():
    rng = np.random.default_rng(99)
    detections = 0
    for threat_class in THREAT_SIGNATURES:
        profile = default_class_profiles()[threat_class]
        noise = np.full(profile.means.shape, 0.15)
        result = monte_carlo_classify(profile.means, noise, n_trials=300, rng=rng)
        if result["predicted_class"] != "nominal":
            detections += 1

    assert detections == len(THREAT_SIGNATURES)
