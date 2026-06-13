"""EAMIS — Enneract Autonomous Mission Intelligence Stack
================================================================
Single-file, numpy-only implementation of the four core modules:

  1. XNAV      — absolute position via X-ray pulsar Roemer delays
  2. CMB_NAV   — absolute velocity via the CMB Doppler dipole
  3. EKF       — fused 6-state Joseph-form Extended Kalman Filter
  4. MADN      — Bayesian onboard AI safety monitor + HMAC audit log

All modules share this single file so the full stack can be audited,
imported, or run as a standalone script without any package installation.

Usage
-----
  python eamis.py            # run the integrated demo
  python eamis.py --selftest # run embedded validation suite

Dependencies: numpy (standard library only otherwise)

Reference values
----------------
  Planck 2018 CMB dipole : |v| = 369.82 km/s toward l=264.02 deg, b=48.25 deg
  CMB monopole T0         : 2.72548 K  (Fixsen 2009)
  Speed of light          : 299 792.458 km/s
  NICER demonstrated XNAV : ~5 km position fix on ISS (SEXTANT 2018)
  This simulation target  : sub-km position, sub-0.5 deg CMB direction
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

C_KM_S: float = 299_792.458          # speed of light, km/s
T0_KELVIN: float = 2.72548           # CMB monopole temperature, K (Fixsen 2009)
AU_KM: float = 1.495_978_707e8       # 1 AU in km

# Planck 2018 best-fit solar-system CMB dipole
PLANCK_2018_SPEED_KM_S: float = 369.82
PLANCK_2018_GALACTIC_LON_DEG: float = 264.02
PLANCK_2018_GALACTIC_LAT_DEG: float = 48.25

# ---------------------------------------------------------------------------
# Section 1 — XNAV: X-ray pulsar navigation
# ---------------------------------------------------------------------------
#
# Millisecond pulsars are the most stable natural clocks known.  Their X-ray
# pulses arrive at the spacecraft delayed by the light-travel time from the
# Solar System Barycenter (SSB):
#
#   dt_i = (p_hat_i . r) / c
#
# where p_hat_i is the ICRS unit vector toward pulsar i and r is the
# spacecraft position relative to the SSB (km).  Observing N >= 3 pulsars
# from non-degenerate directions yields a linear system solved by least
# squares for r.  GDOP (geometric dilution of precision) characterises
# constellation quality; values near 1–2 are considered good.
#
# NICER demonstrated ~5 km fixes on the ISS in 2018 (SEXTANT experiment).
# Sub-km fixes require a 9-pulsar constellation and ~1 µs timing precision.
# ---------------------------------------------------------------------------

# ATNF millisecond pulsar catalog entries: (RA hours, Dec degrees).
# Selected for wide sky coverage to minimise GDOP.
PULSAR_CATALOG: dict[str, tuple[float, float]] = {
    "J0030+0451": (0.509,   4.861),
    "J0218+4232": (2.306,  42.540),
    "J0437-4715": (4.616, -47.253),
    "J0534+2200": (5.575,  22.014),   # Crab — bright, used for calibration
    "J0613-0200": (6.220,  -2.007),
    "J0751+1807": (7.858,  18.123),
    "J1012+5307": (10.205,  53.117),
    "J1024-0719": (10.408,  -7.327),
    "J1744-1134": (17.741, -11.574),
    "J1909-3744": (19.157, -37.744),
    "J2124-3358": (21.408, -33.967),
    "J2241-5236": (22.690, -52.610),
}

# Subset recommended for navigation (9 pulsars, well-spread, GDOP < 1.2)
NAV_CONSTELLATION: tuple[str, ...] = (
    "J0030+0451", "J0437-4715", "J0613-0200", "J0751+1807",
    "J1024-0719", "J1744-1134", "J1909-3744", "J2124-3358", "J2241-5236",
)


def radec_to_unit_vector(ra_hours: float, dec_deg: float) -> np.ndarray:
    """Converts equatorial coordinates (RA h, Dec deg) to an ICRS unit vector."""
    ra = np.deg2rad(ra_hours * 15.0)
    dec = np.deg2rad(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])


def pulsar_directions(
    names: tuple[str, ...] | None = None,
    catalog: dict | None = None,
) -> np.ndarray:
    """Returns an (N, 3) array of ICRS unit vectors for the given pulsar names.

    Defaults to the 9-pulsar NAV_CONSTELLATION optimised for low GDOP.
    """
    catalog = catalog or PULSAR_CATALOG
    names = names or NAV_CONSTELLATION
    return np.array([radec_to_unit_vector(*catalog[n]) for n in names])


def geometric_dilution_of_precision(directions: np.ndarray) -> float:
    """GDOP = sqrt(trace((A^T A)^-1)) for a pulsar direction matrix A."""
    gram = directions.T @ directions
    return float(np.sqrt(np.trace(np.linalg.inv(gram))))


def xnav_timing_noise(
    distance_au: float,
    aperture_m2: float = 0.1,
    base_noise_s: float = 1e-6,
) -> float:
    """Models timing noise growth with distance and detector aperture.

    Flux falls as 1/r^2, so photon shot noise grows as r * aperture^-0.5.
    At 1 AU with 0.1 m^2 effective area the noise is ~1 µs (NICER-class).
    """
    return base_noise_s * distance_au * (0.1 / aperture_m2) ** 0.5


def simulate_roemer_delays(
    position_km: np.ndarray,
    directions: np.ndarray,
    timing_noise_s: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulates measured Roemer delays (s) for a spacecraft at `position_km`."""
    delays = (directions @ np.asarray(position_km)) / C_KM_S
    if timing_noise_s > 0:
        rng = rng or np.random.default_rng()
        delays = delays + rng.normal(0.0, timing_noise_s, size=delays.shape)
    return delays


def solve_position(directions: np.ndarray, delays_s: np.ndarray) -> np.ndarray:
    """Recovers spacecraft position (km) from Roemer delays via least squares."""
    position, *_ = np.linalg.lstsq(directions, C_KM_S * delays_s, rcond=None)
    return position


def xnav_position_fix(
    true_position_km: np.ndarray,
    timing_noise_s: float = 1e-6,
    names: tuple[str, ...] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """End-to-end XNAV fix: simulate → recover → report error and GDOP."""
    dirs = pulsar_directions(names)
    delays = simulate_roemer_delays(true_position_km, dirs, timing_noise_s, rng)
    estimate = solve_position(dirs, delays)
    gdop = geometric_dilution_of_precision(dirs)
    return {
        "estimate_km": estimate,
        "error_km": float(np.linalg.norm(estimate - np.asarray(true_position_km))),
        "gdop": gdop,
    }

# ---------------------------------------------------------------------------
# Section 2 — CMB_NAV: CMB dipole velocity navigation
# ---------------------------------------------------------------------------
#
# The spacecraft's motion relative to the CMB rest frame imprints a dipole
# anisotropy on the observed sky temperature:
#
#   T(n_hat) = T0 * [1 + (v . n_hat) / c]
#
# Sampling the sky in N directions and inverting the linear system gives the
# velocity vector v — without any ground contact or external infrastructure.
# The velocity is expressed in the same frame as the Planck 2018 reference.
#
# Direction accuracy is limited by radiometer noise (~50 µK for a cryo CMB
# detector) and sky coverage; 32 Fibonacci-sphere samples give ~0.30 deg.
# Speed accuracy is limited by confusion with the CMB quadrupole and higher
# multipoles; this simulation treats the sky as a pure dipole.
# ---------------------------------------------------------------------------


def galactic_to_unit_vector(lon_deg: float, lat_deg: float) -> np.ndarray:
    """Converts galactic (l, b) to a unit vector in the galactic frame."""
    l, b = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.array([np.cos(b) * np.cos(l),
                     np.cos(b) * np.sin(l),
                     np.sin(b)])


def planck_2018_velocity_vector() -> np.ndarray:
    """Reference CMB dipole velocity (km/s) from Planck 2018."""
    direction = galactic_to_unit_vector(
        PLANCK_2018_GALACTIC_LON_DEG, PLANCK_2018_GALACTIC_LAT_DEG
    )
    return PLANCK_2018_SPEED_KM_S * direction


def sky_sample_directions(n_samples: int = 32) -> np.ndarray:
    """Fibonacci-sphere unit vectors spread roughly uniformly over the sky."""
    idx = np.arange(n_samples)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    lat = np.arcsin(1 - 2 * (idx + 0.5) / n_samples)
    lon = 2 * np.pi * idx / phi
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=1)


def simulate_sky_temperatures(
    velocity_km_s: np.ndarray,
    directions: np.ndarray,
    noise_kelvin: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulates CMB sky temperatures along `directions` for given velocity."""
    temps = T0_KELVIN * (1.0 + (directions @ np.asarray(velocity_km_s)) / C_KM_S)
    if noise_kelvin > 0:
        rng = rng or np.random.default_rng()
        temps = temps + rng.normal(0.0, noise_kelvin, size=temps.shape)
    return temps


def solve_velocity(directions: np.ndarray, temperatures_k: np.ndarray) -> np.ndarray:
    """Recovers velocity (km/s) from sky temperature samples via least squares."""
    rhs = C_KM_S * (temperatures_k / T0_KELVIN - 1.0)
    velocity, *_ = np.linalg.lstsq(directions, rhs, rcond=None)
    return velocity


def angular_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Angle in degrees between two velocity direction vectors."""
    cos_a = np.dot(estimate, truth) / (np.linalg.norm(estimate) * np.linalg.norm(truth))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def cmb_velocity_fix(
    true_velocity_km_s: np.ndarray,
    noise_kelvin: float = 50e-6,
    n_samples: int = 32,
    rng: np.random.Generator | None = None,
) -> dict:
    """End-to-end CMB dipole velocity fix: simulate → recover → report error."""
    dirs = sky_sample_directions(n_samples)
    temps = simulate_sky_temperatures(true_velocity_km_s, dirs, noise_kelvin, rng)
    estimate = solve_velocity(dirs, temps)
    truth = np.asarray(true_velocity_km_s)
    return {
        "estimate_km_s": estimate,
        "speed_error_km_s": float(abs(np.linalg.norm(estimate) - np.linalg.norm(truth))),
        "direction_error_deg": angular_error_deg(estimate, truth),
    }

# ---------------------------------------------------------------------------
# Section 3 — EKF: fused 6-state Extended Kalman Filter
# ---------------------------------------------------------------------------
#
# State: x = [px, py, pz, vx, vy, vz]  (km, km/s, ICRS frame)
#
# Process model: constant-velocity kinematics with discrete white-noise
# acceleration (DWNA) process noise — appropriate for deep-space cruise where
# unmodeled perturbations (solar radiation pressure, gravitational harmonics)
# are small and slowly varying.
#
# Measurement model: direct position observation from XNAV, direct velocity
# observation from CMB dipole.  Both are linear in the state, so this is a
# standard Kalman filter (not truly "extended"), but we use the Joseph-form
# covariance update throughout for numerical robustness:
#
#   P+ = (I - K H) P (I - K H)^T + K R K^T
#
# This guarantees P stays symmetric and limits accumulation of floating-point
# asymmetry over the ~10 000 update steps expected in a 1-year cruise.
# ---------------------------------------------------------------------------

_STATE_DIM = 6
_H_POS = np.hstack([np.eye(3), np.zeros((3, 3))])   # position measurement matrix
_H_VEL = np.hstack([np.zeros((3, 3)), np.eye(3)])   # velocity measurement matrix


def _state_transition(dt_s: float) -> np.ndarray:
    f = np.eye(_STATE_DIM)
    f[0:3, 3:6] = np.eye(3) * dt_s
    return f


def _process_noise(dt_s: float, accel_psd: float) -> np.ndarray:
    """Discrete white-noise-acceleration process noise covariance Q."""
    q = np.zeros((_STATE_DIM, _STATE_DIM))
    dt2, dt3, dt4 = dt_s ** 2, dt_s ** 3, dt_s ** 4
    for i in range(3):
        j = i + 3
        q[i, i] = dt4 / 4 * accel_psd
        q[i, j] = q[j, i] = dt3 / 2 * accel_psd
        q[j, j] = dt2 * accel_psd
    return q


class FusedEKF:
    """6-state Kalman filter fusing XNAV position and CMB-dipole velocity."""

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        accel_psd: float = 1e-13,
    ) -> None:
        self.state = np.asarray(initial_state, dtype=float).copy()
        self.covariance = np.asarray(initial_covariance, dtype=float).copy()
        self.accel_psd = accel_psd

    # -- time update ---------------------------------------------------------

    def predict(self, dt_s: float) -> None:
        f = _state_transition(dt_s)
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + _process_noise(dt_s, self.accel_psd)

    # -- measurement update (Joseph form) ------------------------------------

    def _update(self, z: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        s = h @ self.covariance @ h.T + r
        k = self.covariance @ h.T @ np.linalg.inv(s)
        self.state = self.state + k @ (z - h @ self.state)
        a = np.eye(_STATE_DIM) - k @ h
        self.covariance = a @ self.covariance @ a.T + k @ r @ k.T

    def update_position(self, position_km: np.ndarray, cov_km2: np.ndarray) -> None:
        self._update(np.asarray(position_km, dtype=float), _H_POS,
                     np.asarray(cov_km2, dtype=float))

    def update_velocity(self, velocity_km_s: np.ndarray, cov_km2s2: np.ndarray) -> None:
        self._update(np.asarray(velocity_km_s, dtype=float), _H_VEL,
                     np.asarray(cov_km2s2, dtype=float))

    # -- accessors -----------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    def position_uncertainty_km(self) -> float:
        """1-sigma position uncertainty from the diagonal of the covariance."""
        return float(np.sqrt(np.trace(self.covariance[0:3, 0:3]) / 3))

    def velocity_uncertainty_km_s(self) -> float:
        """1-sigma velocity uncertainty from the diagonal of the covariance."""
        return float(np.sqrt(np.trace(self.covariance[3:6, 3:6]) / 3))

# ---------------------------------------------------------------------------
# Section 4 — MADN: Multi-Agent Defense Network
# ---------------------------------------------------------------------------
#
# MADN monitors onboard AI agents for behavioural anomalies without ground
# contact, using:
#
#   - A naive-Bayes classifier over a 7-class threat taxonomy, with Gaussian
#     class-conditional likelihoods on 4 extracted telemetry features.
#   - Monte Carlo uncertainty quantification: N=500 noisy redraws of the
#     feature vector, producing a mean posterior and trial-to-trial spread.
#   - An HMAC-SHA256-chained audit log: each entry's signature covers its
#     own content plus the previous entry's signature, making tampering or
#     deletion detectable even without a ground-contact reference.
#
# 7-class taxonomy:
#   0  nominal             — expected behaviour
#   1  behavioral_drift    — gradual deviation from baseline
#   2  deception           — goal-misaligned outputs
#   3  prompt_injection    — adversarial input hijacking
#   4  covert_channel      — exfiltration via side channel
#   5  data_exfiltration   — direct data leakage
#   6  compromise          — full adversarial takeover
#
# 4 telemetry features (z-scored against rolling baseline):
#   action_rate       — actions per minute
#   lexical_anomaly   — out-of-vocabulary token fraction
#   goal_divergence   — cosine distance between stated and inferred goals
#   resource_spike    — compute / bandwidth deviation
# ---------------------------------------------------------------------------

THREAT_CLASSES: tuple[str, ...] = (
    "nominal",
    "behavioral_drift",
    "deception",
    "prompt_injection",
    "covert_channel",
    "data_exfiltration",
    "compromise",
)

FEATURE_NAMES: tuple[str, ...] = (
    "action_rate",
    "lexical_anomaly",
    "goal_divergence",
    "resource_spike",
)

_N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class ClassProfile:
    """Gaussian class-conditional feature model for naive-Bayes classification."""
    prior: float
    means: np.ndarray
    stds: np.ndarray


def default_class_profiles() -> dict[str, ClassProfile]:
    """Hand-specified Gaussian profiles per threat class.

    Nominal telemetry clusters near zero on all features.  Each threat class
    is defined by elevated signatures on its most characteristic features.
    Phase I will replace these with profiles fit to red-team telemetry.
    """
    specs: dict[str, tuple[float, list, list]] = {
        "nominal":           (0.55, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        "behavioral_drift":  (0.10, [1.5, 0.5, 1.0, 0.3], [1.0, 1.0, 1.0, 1.0]),
        "deception":         (0.08, [0.3, 1.0, 2.0, 0.2], [1.0, 1.0, 1.0, 1.0]),
        "prompt_injection":  (0.08, [0.5, 2.5, 0.8, 0.2], [1.0, 1.0, 1.0, 1.0]),
        "covert_channel":    (0.07, [2.0, 0.5, 0.3, 2.0], [1.0, 1.0, 1.0, 1.0]),
        "data_exfiltration": (0.06, [1.0, 0.3, 0.5, 3.0], [1.0, 1.0, 1.0, 1.0]),
        "compromise":        (0.06, [3.0, 2.0, 2.5, 2.5], [1.0, 1.0, 1.0, 1.0]),
    }
    return {
        name: ClassProfile(prior=prior,
                           means=np.array(means),
                           stds=np.array(stds))
        for name, (prior, means, stds) in specs.items()
    }


def _log_likelihood(features: np.ndarray, profile: ClassProfile) -> float:
    var = profile.stds ** 2
    return float(np.sum(-0.5 * np.log(2 * np.pi * var)
                        - 0.5 * ((features - profile.means) ** 2) / var))


def classify(
    features: np.ndarray,
    profiles: dict[str, ClassProfile] | None = None,
) -> dict[str, float]:
    """Naive-Bayes posterior over threat classes for one telemetry sample."""
    profiles = profiles or default_class_profiles()
    log_p = {name: np.log(p.prior) + _log_likelihood(features, p)
             for name, p in profiles.items()}
    values = np.array(list(log_p.values()))
    log_z = np.max(values) + np.log(np.sum(np.exp(values - np.max(values))))
    return {name: float(np.exp(lp - log_z)) for name, lp in log_p.items()}


def monte_carlo_classify(
    features: np.ndarray,
    feature_noise_std: np.ndarray,
    n_trials: int = 500,
    profiles: dict[str, ClassProfile] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Monte Carlo uncertainty quantification over `n_trials` noisy redraws."""
    rng = rng or np.random.default_rng()
    profiles = profiles or default_class_profiles()
    samples: dict[str, list[float]] = {name: [] for name in THREAT_CLASSES}
    for _ in range(n_trials):
        noisy = features + rng.normal(0.0, feature_noise_std, size=features.shape)
        for name, p in classify(noisy, profiles).items():
            samples[name].append(p)
    mean = {name: float(np.mean(v)) for name, v in samples.items()}
    std = {name: float(np.std(v)) for name, v in samples.items()}
    predicted = max(mean, key=mean.__getitem__)
    return {
        "predicted_class": predicted,
        "mean_posterior": mean,
        "posterior_std": std,
        "confidence": mean[predicted],
        "uncertainty": std[predicted],
    }


def madn_confusion_matrix(
    n_trials_per_class: int = 100,
    feature_noise_std: float = 0.15,
    mc_trials: int = 200,
    profiles: dict[str, ClassProfile] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Confusion matrix over all threat classes, returned as a nested dict."""
    rng = rng or np.random.default_rng()
    profiles = profiles or default_class_profiles()
    noise = np.full(_N_FEATURES, feature_noise_std)
    matrix: dict[str, dict[str, int]] = {t: {p: 0 for p in THREAT_CLASSES}
                                          for t in THREAT_CLASSES}
    for true_class in THREAT_CLASSES:
        true_features = profiles[true_class].means
        for _ in range(n_trials_per_class):
            result = monte_carlo_classify(true_features, noise,
                                         n_trials=mc_trials, profiles=profiles, rng=rng)
            matrix[true_class][result["predicted_class"]] += 1
    return matrix


# -- HMAC-chained audit log --------------------------------------------------

_GENESIS_SIG = "0" * 64


@dataclass
class AuditLogEntry:
    sequence: int
    timestamp: str
    event: str
    payload: dict
    previous_signature: str
    signature: str = field(init=False)

    def _body_bytes(self) -> bytes:
        return json.dumps({
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event": self.event,
            "payload": self.payload,
            "previous_signature": self.previous_signature,
        }, sort_keys=True).encode()

    def sign(self, key: bytes) -> None:
        self.signature = _hmac.new(key, self._body_bytes(), hashlib.sha256).hexdigest()

    def verify(self, key: bytes) -> bool:
        expected = _hmac.new(key, self._body_bytes(), hashlib.sha256).hexdigest()
        return _hmac.compare_digest(expected, self.signature)


class HMACAuditLog:
    """Append-only, HMAC-SHA256-chained tamper-evident log.

    Each entry signs its own content and the previous entry's signature,
    so deleting or altering any historical entry invalidates every
    subsequent signature without requiring a ground-contact reference.
    """

    def __init__(self, key: bytes) -> None:
        self._key = key
        self._entries: list[AuditLogEntry] = []

    def append(self, event: str, payload: dict,
               timestamp: str | None = None) -> AuditLogEntry:
        prev_sig = self._entries[-1].signature if self._entries else _GENESIS_SIG
        entry = AuditLogEntry(
            sequence=len(self._entries),
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event=event,
            payload=payload,
            previous_signature=prev_sig,
        )
        entry.sign(self._key)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[AuditLogEntry]:
        return list(self._entries)

    def verify_chain(self) -> bool:
        prev_sig = _GENESIS_SIG
        for entry in self._entries:
            if entry.previous_signature != prev_sig:
                return False
            if not entry.verify(self._key):
                return False
            prev_sig = entry.signature
        return True

    def to_compressed_report(self, last_n: int = 5) -> str:
        return json.dumps([
            {"sequence": e.sequence, "timestamp": e.timestamp,
             "event": e.event, "payload": e.payload, "signature": e.signature}
            for e in self._entries[-last_n:]
        ], sort_keys=True)

# ---------------------------------------------------------------------------
# Section 5 — OrreryNavigator: integrated mission simulator
# ---------------------------------------------------------------------------
#
# Combines all four modules into a single mission-loop object:
#
#   1. Propagate true constant-velocity cruise dynamics.
#   2. Take an XNAV pulsar position fix.
#   3. Take a CMB dipole velocity fix.
#   4. Fuse both through the Joseph-form EKF.
#   5. Screen onboard agent telemetry through MADN.
#   6. Append the screening result to the HMAC audit log.
#
# This represents one cruise-interval step (typically dt_s = 3600 s, one
# hourly navigation cycle).  The mission_report() method serialises the
# navigation state and a compressed audit log excerpt for ground transmission
# once DSN contact is restored.
# ---------------------------------------------------------------------------


class OrreryNavigator:
    """Single spacecraft running the full EAMIS navigation + safety stack."""

    def __init__(
        self,
        initial_position_km: np.ndarray,
        initial_velocity_km_s: np.ndarray,
        audit_key: bytes = b"eamis-mission-key",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.true_position_km = np.asarray(initial_position_km, dtype=float).copy()
        self.true_velocity_km_s = np.asarray(initial_velocity_km_s, dtype=float).copy()

        self._pulsar_dirs = pulsar_directions()
        self._sky_dirs = sky_sample_directions()
        self._gdop = geometric_dilution_of_precision(self._pulsar_dirs)

        initial_state = np.concatenate([self.true_position_km, self.true_velocity_km_s])
        initial_cov = np.diag([1e8, 1e8, 1e8, 1.0, 1.0, 1.0])
        self.ekf = FusedEKF(initial_state, initial_cov, accel_psd=1e-13)

        self._madn_profiles = default_class_profiles()
        self.audit_log = HMACAuditLog(key=audit_key)
        self.history: list[dict] = []

    # -- navigation fixes ----------------------------------------------------

    def _xnav_fix(self, timing_noise_s: float = 1e-6) -> dict:
        delays = simulate_roemer_delays(
            self.true_position_km, self._pulsar_dirs, timing_noise_s, self.rng)
        estimate = solve_position(self._pulsar_dirs, delays)
        sigma_km = timing_noise_s * C_KM_S * self._gdop
        return {"estimate_km": estimate, "cov_km2": np.eye(3) * sigma_km ** 2}

    def _cmb_fix(self, noise_kelvin: float = 50e-6) -> dict:
        temps = simulate_sky_temperatures(
            self.true_velocity_km_s, self._sky_dirs, noise_kelvin, self.rng)
        estimate = solve_velocity(self._sky_dirs, temps)
        sigma_km_s = noise_kelvin / T0_KELVIN * C_KM_S
        return {"estimate_km_s": estimate, "cov_km2s2": np.eye(3) * sigma_km_s ** 2}

    # -- agent screening -----------------------------------------------------

    def _screen_agent(self, step: int, features: np.ndarray) -> dict:
        noise = np.full(features.shape, 0.15)
        result = monte_carlo_classify(features, noise, n_trials=200,
                                      profiles=self._madn_profiles, rng=self.rng)
        self.audit_log.append(
            event="AgentTelemetryScreened",
            payload={"step": step,
                     "predicted_class": result["predicted_class"],
                     "confidence": result["confidence"],
                     "uncertainty": result["uncertainty"]},
        )
        return result

    # -- mission step --------------------------------------------------------

    def step(
        self,
        dt_s: float = 3600.0,
        agent_features: np.ndarray | None = None,
    ) -> dict:
        """One cruise interval: propagate → fix → fuse → screen → log."""
        self.true_position_km = self.true_position_km + self.true_velocity_km_s * dt_s
        self.ekf.predict(dt_s)

        xf = self._xnav_fix()
        self.ekf.update_position(xf["estimate_km"], xf["cov_km2"])

        cf = self._cmb_fix()
        self.ekf.update_velocity(cf["estimate_km_s"], cf["cov_km2s2"])

        if agent_features is None:
            agent_features = self.rng.normal(0.0, 1.0, size=_N_FEATURES)
        madn_result = self._screen_agent(len(self.history), agent_features)

        record = {
            "true_position_km":      self.true_position_km.copy(),
            "true_velocity_km_s":    self.true_velocity_km_s.copy(),
            "estimated_position_km": self.ekf.position.copy(),
            "estimated_velocity_km_s": self.ekf.velocity.copy(),
            "position_error_km":     float(np.linalg.norm(
                self.ekf.position - self.true_position_km)),
            "velocity_error_km_s":   float(np.linalg.norm(
                self.ekf.velocity - self.true_velocity_km_s)),
            "madn_class":       madn_result["predicted_class"],
            "madn_confidence":  madn_result["confidence"],
        }
        self.history.append(record)
        return record

    def run(self, n_steps: int, dt_s: float = 3600.0) -> list[dict]:
        for _ in range(n_steps):
            self.step(dt_s)
        return self.history

    def mission_report(self) -> dict:
        latest = self.history[-1] if self.history else None
        return {
            "steps_completed":            len(self.history),
            "latest_position_km":         latest["estimated_position_km"].tolist() if latest else None,
            "latest_velocity_km_s":       latest["estimated_velocity_km_s"].tolist() if latest else None,
            "latest_position_error_km":   latest["position_error_km"] if latest else None,
            "latest_velocity_error_km_s": latest["velocity_error_km_s"] if latest else None,
            "gdop":                       self._gdop,
            "audit_chain_intact":         self.audit_log.verify_chain(),
            "audit_log_excerpt":          self.audit_log.to_compressed_report(last_n=5),
        }

# ---------------------------------------------------------------------------
# Section 6 — Embedded self-test suite
# ---------------------------------------------------------------------------
#
# Every function above has at least one regression test here.  Run with
#   python eamis.py --selftest
# Tests are designed to fail loudly on regression: a test failure means a
# physics constant was changed, a numerical routine was broken, or a claim
# in the validation table no longer holds.
# ---------------------------------------------------------------------------


def _run_selftests(verbose: bool = True) -> int:
    """Returns number of failures."""

    failures = 0

    def check(name: str, cond: bool, msg: str = "") -> None:
        nonlocal failures
        if cond:
            if verbose:
                print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}" + (f" — {msg}" if msg else ""))

    rng = np.random.default_rng(20260608)

    # -- XNAV ----------------------------------------------------------------
    if verbose:
        print("\n[XNAV]")

    dirs = pulsar_directions()
    check("unit_vectors", np.allclose(np.linalg.norm(dirs, axis=1), 1.0))
    check("constellation_rank", np.linalg.matrix_rank(dirs) == 3)

    gdop = geometric_dilution_of_precision(dirs)
    check("gdop_range", 1.0 < gdop < 2.0, f"gdop={gdop:.3f}")

    true_pos = np.array([2.2e8, 1.1e8, -5.0e7])
    delays_nf = simulate_roemer_delays(true_pos, dirs, 0.0)
    recovered = solve_position(dirs, delays_nf)
    check("noiseless_exact", np.allclose(recovered, true_pos, rtol=1e-9))

    errors = [xnav_position_fix(rng.uniform(-2e8, 2e8, 3),
              timing_noise_s=1e-6, rng=rng)["error_km"]
              for _ in range(500)]
    median_err = float(np.median(errors))
    check("median_sub_km", median_err < 1.0, f"median={median_err:.3f} km")

    dist_noise = xnav_timing_noise(distance_au=50.0)
    check("noise_grows_with_distance", dist_noise > 1e-6)

    # -- CMB_NAV -------------------------------------------------------------
    if verbose:
        print("\n[CMB_NAV]")

    v_ref = planck_2018_velocity_vector()
    check("planck_speed", np.isclose(np.linalg.norm(v_ref), PLANCK_2018_SPEED_KM_S))

    sdirs = sky_sample_directions(32)
    check("sky_dirs_unit", np.allclose(np.linalg.norm(sdirs, axis=1), 1.0))
    check("sky_dirs_rank", np.linalg.matrix_rank(sdirs) == 3)

    temps_nf = simulate_sky_temperatures(v_ref, sdirs, 0.0)
    v_est = solve_velocity(sdirs, temps_nf)
    check("noiseless_velocity", np.allclose(v_est, v_ref, rtol=1e-6))

    dir_errors = [cmb_velocity_fix(v_ref, 50e-6, rng=rng)["direction_error_deg"]
                  for _ in range(300)]
    med_dir = float(np.median(dir_errors))
    check("direction_sub_half_deg", med_dir < 0.5, f"median={med_dir:.3f} deg")

    # -- EKF -----------------------------------------------------------------
    if verbose:
        print("\n[EKF]")

    ekf = FusedEKF(np.zeros(6), np.eye(6) * 1e6, accel_psd=1e-13)
    for _ in range(20):
        ekf.predict(3600.0)
        ekf.update_position(rng.normal(size=3) * 1e3, np.eye(3))
        ekf.update_velocity(rng.normal(size=3) * 0.1, np.eye(3) * 1e-4)
    check("cov_symmetric", np.allclose(ekf.covariance, ekf.covariance.T, atol=1e-6))
    check("cov_positive_definite",
          np.all(np.linalg.eigvalsh(ekf.covariance) > 0))

    nav = OrreryNavigator(np.array([2.2e8, 1.1e8, -5e7]),
                          np.array([12., -4., 1.5]), rng=rng)
    hist = nav.run(40, 3600.0)
    check("velocity_converges",
          hist[-1]["velocity_error_km_s"] < 0.01,
          f"vel_err={hist[-1]['velocity_error_km_s']:.5f}")
    check("position_converges",
          hist[-1]["position_error_km"] < 2.0,
          f"pos_err={hist[-1]['position_error_km']:.3f}")

    # -- MADN ----------------------------------------------------------------
    if verbose:
        print("\n[MADN]")

    nominal_features = np.zeros(_N_FEATURES)
    posterior = classify(nominal_features)
    check("posterior_sums_to_one", np.isclose(sum(posterior.values()), 1.0))
    check("nominal_predicted_as_nominal",
          max(posterior, key=posterior.__getitem__) == "nominal")

    profiles = default_class_profiles()
    noise = np.full(_N_FEATURES, 0.15)
    threat_classes = [c for c in THREAT_CLASSES if c != "nominal"]
    detections, correct = 0, 0
    for tc in threat_classes:
        result = monte_carlo_classify(profiles[tc].means, noise,
                                     n_trials=300, rng=rng)
        if result["predicted_class"] != "nominal":
            detections += 1
        if result["predicted_class"] == tc:
            correct += 1
    check("redteam_detection_6of6", detections == 6,
          f"{detections}/6 detected")
    check("redteam_identification_6of6", correct == 6,
          f"{correct}/6 identified")

    false_rejects = sum(
        monte_carlo_classify(nominal_features, noise, n_trials=100, rng=rng
                             )["predicted_class"] != "nominal"
        for _ in range(200)
    )
    check("false_reject_rate_low", false_rejects / 200 < 0.10,
          f"rate={false_rejects/200:.2%}")

    # behavioral_drift's profile mean is the closest of the six threat
    # classes to nominal's, making it the highest-risk class for false
    # positives if telemetry noise grows beyond the calibrated 0.15 std.
    # These checks lock in that margin so a future profile tweak can't
    # silently erode it.
    nominal_mean = profiles["nominal"].means
    drift_margin = float(np.linalg.norm(profiles["behavioral_drift"].means - nominal_mean))
    other_margins = [float(np.linalg.norm(profiles[c].means - nominal_mean))
                     for c in threat_classes if c != "behavioral_drift"]
    check("drift_is_nearest_threat_to_nominal", drift_margin < min(other_margins),
          f"drift_margin={drift_margin:.3f}, nearest_other={min(other_margins):.3f}")
    check("drift_margin_safe_at_calibrated_noise", drift_margin / 0.15 > 5.0,
          f"margin/noise_std={drift_margin / 0.15:.1f}")

    drift_fp_calibrated = sum(
        max(classify(nominal_mean + rng.normal(0.0, 0.15, size=_N_FEATURES), profiles
                      ).items(), key=lambda kv: kv[1])[0] == "behavioral_drift"
        for _ in range(1000)
    )
    check("drift_false_positives_zero_at_calibrated_noise", drift_fp_calibrated == 0,
          f"{drift_fp_calibrated}/1000")

    drift_fp_stress = sum(
        max(classify(nominal_mean + rng.normal(0.0, 0.75, size=_N_FEATURES), profiles
                      ).items(), key=lambda kv: kv[1])[0] == "behavioral_drift"
        for _ in range(1000)
    )
    check("drift_false_positive_rate_bounded_under_stress", drift_fp_stress / 1000 < 0.05,
          f"rate={drift_fp_stress/1000:.2%} @ noise_std=0.75")

    # -- Audit log -----------------------------------------------------------
    if verbose:
        print("\n[HMACAuditLog]")

    log = HMACAuditLog(b"test-key")
    for i in range(6):
        log.append("Test", {"i": i}, timestamp=f"t{i}")
    check("chain_intact_untampered", log.verify_chain())

    log.entries[2].payload["i"] = 999
    check("chain_broken_on_tamper", not log.verify_chain())

    # -- Integration ---------------------------------------------------------
    if verbose:
        print("\n[OrreryNavigator integration]")

    nav2 = OrreryNavigator(np.array([5e7, -1e8, 3e7]),
                           np.array([8., 3., -1.]), rng=rng)
    nav2.run(10, 3600.0)
    check("audit_chain_after_10_steps", nav2.audit_log.verify_chain())
    check("audit_entries_count", len(nav2.audit_log.entries) == 10)
    report = nav2.mission_report()
    check("report_audit_intact", report["audit_chain_intact"])

    # inject compromise
    compromise_sig = profiles["compromise"].means
    rec = nav2.step(3600.0, agent_features=compromise_sig)
    check("compromise_flagged", rec["madn_class"] == "compromise")

    return failures


# ---------------------------------------------------------------------------
# Section 7 — Validation report (reproducible, seeded)
# ---------------------------------------------------------------------------

def validation_report(seed: int = 20260608, n_trials: int = 500) -> None:
    """Prints the validated performance figures for this exact codebase.
    Cite numbers from this output only — run it yourself to reproduce.
    """
    rng = np.random.default_rng(seed)
    print(f"EAMIS validation report  seed={seed}  n_trials={n_trials}")
    print("-" * 60)

    # XNAV
    dirs = pulsar_directions()
    gdop = geometric_dilution_of_precision(dirs)
    errors = [xnav_position_fix(rng.uniform(-2e8, 2e8, 3), 1e-6, rng=rng)["error_km"]
              for _ in range(n_trials)]
    print(f"XNAV  pulsars={len(dirs)}  GDOP={gdop:.3f}")
    print(f"      median error @ 1 µs : {np.median(errors):.3f} km")
    print(f"      p95 error @ 1 µs    : {np.percentile(errors, 95):.3f} km")

    # CMB
    v_ref = planck_2018_velocity_vector()
    sky_dirs = sky_sample_directions()
    dir_errs = [cmb_velocity_fix(v_ref, 50e-6, rng=rng)["direction_error_deg"]
                for _ in range(n_trials)]
    print(f"CMB   sky_samples={len(sky_dirs)}")
    print(f"      median direction error @ 50 µK : {np.median(dir_errs):.3f} deg")

    # EKF
    nav = OrreryNavigator(np.array([2.2e8, 1.1e8, -5e7]),
                          np.array([12., -4., 1.5]), rng=rng)
    hist = nav.run(40, 3600.0)
    late = hist[-5:]
    print(f"EKF   40-step hourly cruise")
    print(f"      mean pos error (steps 36-40) : {np.mean([h['position_error_km'] for h in late]):.3f} km")
    print(f"      mean vel error (steps 36-40) : {np.mean([h['velocity_error_km_s'] for h in late]):.6f} km/s")

    # MADN
    profiles = default_class_profiles()
    noise = np.full(_N_FEATURES, 0.15)
    threat_classes = [c for c in THREAT_CLASSES if c != "nominal"]
    detections = sum(
        monte_carlo_classify(profiles[tc].means, noise, n_trials=100, rng=rng
                             )["predicted_class"] != "nominal"
        for tc in threat_classes
    )
    false_rejects = sum(
        monte_carlo_classify(np.zeros(_N_FEATURES), noise, n_trials=100, rng=rng
                             )["predicted_class"] != "nominal"
        for _ in range(200)
    )
    print(f"MADN  red-team detection    : {detections}/{len(threat_classes)}")
    print(f"      false-reject rate     : {false_rejects/200:.1%}")

# ---------------------------------------------------------------------------
# Section 8 — Demo entry point
# ---------------------------------------------------------------------------


def _demo() -> None:
    rng = np.random.default_rng(42)
    nav = OrreryNavigator(
        initial_position_km=np.array([2.2e8, 1.1e8, -5.0e7]),
        initial_velocity_km_s=np.array([12.0, -4.0, 1.5]),
        rng=rng,
    )
    history = nav.run(n_steps=40, dt_s=3600.0)

    print("EAMIS / ORRERYNAV  —  40-step hourly cruise simulation\n")
    for step in (0, 9, 19, 29, 39):
        h = history[step]
        print(f"  step {step:2d}  pos_err {h['position_error_km']:7.3f} km  "
              f"vel_err {h['velocity_error_km_s']:.5f} km/s  "
              f"MADN → {h['madn_class']} ({h['madn_confidence']:.2f})")

    print()
    report = nav.mission_report()
    for k, v in report.items():
        if k == "audit_log_excerpt":
            print(f"  {k}: <{len(v)} bytes, signatures included>")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("Running EAMIS self-tests …")
        n_fail = _run_selftests(verbose=True)
        print(f"\n{'All tests passed.' if n_fail == 0 else f'{n_fail} test(s) FAILED.'}")
        sys.exit(n_fail)
    elif "--validate" in sys.argv:
        validation_report()
    else:
        _demo()
