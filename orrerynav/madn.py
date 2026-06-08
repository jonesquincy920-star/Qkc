"""MADN: Multi-Agent Defense Network.

A spacecraft-local Bayesian behavioral classifier that monitors onboard AI
agents for drift, deception, prompt injection, or covert-channel activity
without requiring ground contact.

- A naive-Bayes classifier scores agent telemetry against a 7-class threat
  taxonomy.
- Monte Carlo sampling over the feature posteriors quantifies classification
  uncertainty (mean + spread across repeated draws).
- Every classification is appended to an HMAC-chained audit log: each entry's
  signature covers its own content plus the previous entry's signature, so any
  tampering with or removal of a past entry invalidates the chain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

THREAT_CLASSES = (
    "nominal",
    "behavioral_drift",
    "deception",
    "prompt_injection",
    "covert_channel",
    "data_exfiltration",
    "compromise",
)

FEATURE_NAMES = (
    "action_rate",       # actions per minute, z-scored against baseline
    "lexical_anomaly",   # fraction of output tokens outside expected lexicon
    "goal_divergence",   # cosine distance between stated and inferred goal embeddings
    "resource_spike",    # z-scored deviation in compute/bandwidth usage
)


@dataclass(frozen=True)
class ClassProfile:
    """Gaussian feature profile (mean, std) for one threat class, used as the
    class-conditional likelihood model in the naive-Bayes classifier."""

    prior: float
    means: np.ndarray
    stds: np.ndarray


def default_class_profiles() -> dict:
    """A hand-specified Gaussian profile per threat class.

    `nominal` telemetry clusters near zero on every feature; each threat
    class is characterized by an elevated signature on the features it most
    plausibly perturbs. These are deliberately simple, interpretable priors —
    Phase I would replace them with profiles fit to red-team telemetry.
    """
    specs = {
        "nominal":            (0.55, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        "behavioral_drift":   (0.10, [1.5, 0.5, 1.0, 0.3], [1.0, 1.0, 1.0, 1.0]),
        "deception":          (0.08, [0.3, 1.0, 2.0, 0.2], [1.0, 1.0, 1.0, 1.0]),
        "prompt_injection":   (0.08, [0.5, 2.5, 0.8, 0.2], [1.0, 1.0, 1.0, 1.0]),
        "covert_channel":     (0.07, [2.0, 0.5, 0.3, 2.0], [1.0, 1.0, 1.0, 1.0]),
        "data_exfiltration":  (0.06, [1.0, 0.3, 0.5, 3.0], [1.0, 1.0, 1.0, 1.0]),
        "compromise":         (0.06, [3.0, 2.0, 2.5, 2.5], [1.0, 1.0, 1.0, 1.0]),
    }
    return {
        name: ClassProfile(prior=prior, means=np.array(means), stds=np.array(stds))
        for name, (prior, means, stds) in specs.items()
    }


def _gaussian_log_likelihood(features: np.ndarray, profile: ClassProfile) -> float:
    variance = profile.stds ** 2
    return float(np.sum(
        -0.5 * np.log(2 * np.pi * variance)
        - 0.5 * ((features - profile.means) ** 2) / variance
    ))


def classify(features: np.ndarray, profiles: dict | None = None) -> dict:
    """Naive-Bayes posterior over threat classes for one telemetry sample.

    Returns a dict mapping each class name to its posterior probability.
    """
    profiles = profiles or default_class_profiles()
    log_posteriors = {}
    for name in THREAT_CLASSES:
        profile = profiles[name]
        log_posteriors[name] = np.log(profile.prior) + _gaussian_log_likelihood(features, profile)

    # log-sum-exp normalization for numerical stability
    log_values = np.array(list(log_posteriors.values()))
    normalizer = np.max(log_values) + np.log(np.sum(np.exp(log_values - np.max(log_values))))
    return {name: float(np.exp(lp - normalizer)) for name, lp in log_posteriors.items()}


def monte_carlo_classify(
    features: np.ndarray,
    feature_noise_std: np.ndarray,
    n_trials: int = 500,
    profiles: dict | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Quantifies classification uncertainty by re-running `classify` over
    `n_trials` noisy redraws of the input features (Monte Carlo dropout-style
    sampling), returning the mean posterior and its trial-to-trial spread.
    """
    rng = rng or np.random.default_rng()
    profiles = profiles or default_class_profiles()

    samples = {name: np.zeros(n_trials) for name in THREAT_CLASSES}
    for trial in range(n_trials):
        noisy_features = features + rng.normal(0.0, feature_noise_std, size=features.shape)
        posterior = classify(noisy_features, profiles)
        for name in THREAT_CLASSES:
            samples[name][trial] = posterior[name]

    mean = {name: float(np.mean(values)) for name, values in samples.items()}
    std = {name: float(np.std(values)) for name, values in samples.items()}
    predicted_class = max(mean, key=mean.get)
    return {
        "predicted_class": predicted_class,
        "mean_posterior": mean,
        "posterior_std": std,
        "confidence": mean[predicted_class],
        "uncertainty": std[predicted_class],
    }


@dataclass
class AuditLogEntry:
    sequence: int
    timestamp: str
    event: str
    payload: dict
    previous_signature: str
    signature: str = field(init=False)

    def _signed_content(self) -> bytes:
        body = {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event": self.event,
            "payload": self.payload,
            "previous_signature": self.previous_signature,
        }
        return json.dumps(body, sort_keys=True).encode("utf-8")

    def sign(self, key: bytes) -> None:
        self.signature = hmac.new(key, self._signed_content(), hashlib.sha256).hexdigest()

    def verify(self, key: bytes) -> bool:
        expected = hmac.new(key, self._signed_content(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature)


GENESIS_SIGNATURE = "0" * 64


class HMACAuditLog:
    """An append-only, HMAC-chained audit log.

    Each entry's signature is computed over its own content *and* the
    previous entry's signature, forming a hash chain: altering or deleting
    any historical entry breaks every signature after it, making tampering
    detectable even without ground contact to compare against a reference.
    """

    def __init__(self, key: bytes):
        self._key = key
        self._entries: list[AuditLogEntry] = []

    def append(self, event: str, payload: dict, timestamp: str | None = None) -> AuditLogEntry:
        previous_signature = self._entries[-1].signature if self._entries else GENESIS_SIGNATURE
        entry = AuditLogEntry(
            sequence=len(self._entries),
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event=event,
            payload=payload,
            previous_signature=previous_signature,
        )
        entry.sign(self._key)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[AuditLogEntry]:
        return list(self._entries)

    def verify_chain(self) -> bool:
        """Returns True iff every entry's signature is valid and correctly
        chains to its predecessor (tamper-evidence check)."""
        previous_signature = GENESIS_SIGNATURE
        for entry in self._entries:
            if entry.previous_signature != previous_signature:
                return False
            if not entry.verify(self._key):
                return False
            previous_signature = entry.signature
        return True

    def to_compressed_report(self, last_n: int = 5) -> str:
        """Serializes the most recent `last_n` entries for ground transmission."""
        recent = self._entries[-last_n:]
        return json.dumps([
            {
                "sequence": e.sequence,
                "timestamp": e.timestamp,
                "event": e.event,
                "payload": e.payload,
                "signature": e.signature,
            }
            for e in recent
        ], sort_keys=True)
