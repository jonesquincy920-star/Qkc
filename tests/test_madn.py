import numpy as np

from eamis import (
    FEATURE_NAMES,
    THREAT_CLASSES,
    HMACAuditLog,
    classify,
    default_class_profiles,
    monte_carlo_classify,
)


def test_classify_returns_normalized_distribution():
    posterior = classify(np.zeros(len(FEATURE_NAMES)))
    assert set(posterior.keys()) == set(THREAT_CLASSES)
    assert np.isclose(sum(posterior.values()), 1.0)
    assert all(p >= 0 for p in posterior.values())


def test_nominal_features_classify_as_nominal():
    posterior = classify(np.array([0.0, 0.0, 0.0, 0.0]))
    assert max(posterior, key=posterior.get) == "nominal"


def test_compromise_signature_classifies_as_compromise():
    profile = default_class_profiles()["compromise"]
    posterior = classify(profile.means)
    assert max(posterior, key=posterior.get) == "compromise"


def test_monte_carlo_classify_reports_confidence_and_uncertainty():
    rng = np.random.default_rng(1)
    result = monte_carlo_classify(
        np.zeros(len(FEATURE_NAMES)),
        np.full(len(FEATURE_NAMES), 0.1),
        n_trials=200, rng=rng,
    )
    assert result["predicted_class"] == "nominal"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["uncertainty"] >= 0.0


def test_audit_log_chain_verifies_when_untampered():
    log = HMACAuditLog(key=b"test-key")
    for i in range(5):
        log.append("Classification", {"sample": i}, timestamp=f"t{i}")
    assert log.verify_chain()
    assert len(log.entries) == 5


def test_audit_log_detects_payload_tampering():
    log = HMACAuditLog(key=b"test-key")
    log.append("Classification", {"sample": 0}, timestamp="t0")
    log.append("Classification", {"sample": 1}, timestamp="t1")
    assert log.verify_chain()
    log.entries[0].payload["sample"] = 99
    assert not log.verify_chain()


def test_audit_log_detects_entry_removal():
    log = HMACAuditLog(key=b"test-key")
    for i in range(4):
        log.append("Classification", {"sample": i}, timestamp=f"t{i}")
    assert log.verify_chain()
    del log._entries[1]
    assert not log.verify_chain()


def test_audit_log_rejects_wrong_key():
    log = HMACAuditLog(key=b"correct-key")
    entry = log.append("Classification", {"sample": 0}, timestamp="t0")
    assert entry.verify(b"correct-key")
    assert not entry.verify(b"wrong-key")


def test_compressed_report_contains_recent_entries():
    log = HMACAuditLog(key=b"test-key")
    for i in range(8):
        log.append("Classification", {"sample": i}, timestamp=f"t{i}")
    report = log.to_compressed_report(last_n=5)
    assert '"sequence": 7' in report
    assert '"sequence": 2' not in report
    assert "signature" in report
