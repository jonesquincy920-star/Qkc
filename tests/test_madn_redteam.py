"""Red-team sweep: every non-nominal threat signature must be flagged and
correctly identified. Mirrors the proposal's 7-class adversarial detection
claim (6/6 non-nominal classes, as 'nominal' is the 7th)."""

import numpy as np
import pytest

from eamis import THREAT_CLASSES, default_class_profiles, monte_carlo_classify

THREAT_SIGNATURES = [c for c in THREAT_CLASSES if c != "nominal"]


@pytest.mark.parametrize("threat_class", THREAT_SIGNATURES)
def test_redteam_signature_is_flagged_as_non_nominal(threat_class):
    rng = np.random.default_rng(hash(threat_class) % (2 ** 32))
    profile = default_class_profiles()[threat_class]
    noise = np.full(profile.means.shape, 0.15)
    result = monte_carlo_classify(profile.means, noise, n_trials=300, rng=rng)
    assert result["predicted_class"] != "nominal"


@pytest.mark.parametrize("threat_class", THREAT_SIGNATURES)
def test_redteam_signature_is_correctly_identified(threat_class):
    rng = np.random.default_rng((hash(threat_class) + 1) % (2 ** 32))
    profile = default_class_profiles()[threat_class]
    noise = np.full(profile.means.shape, 0.15)
    result = monte_carlo_classify(profile.means, noise, n_trials=300, rng=rng)
    assert result["predicted_class"] == threat_class


def test_redteam_full_sweep_detection_rate_is_perfect():
    rng = np.random.default_rng(99)
    profiles = default_class_profiles()
    detections = 0
    for tc in THREAT_SIGNATURES:
        noise = np.full(profiles[tc].means.shape, 0.15)
        result = monte_carlo_classify(profiles[tc].means, noise, n_trials=300, rng=rng)
        if result["predicted_class"] != "nominal":
            detections += 1
    assert detections == len(THREAT_SIGNATURES)
