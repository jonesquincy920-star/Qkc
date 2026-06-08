"""ORRERYNAV: the integrated deep-space navigation + AI-safety simulator.

Combines the four EAMIS capabilities into a single mission loop:

  1. XNAV       -- absolute position from pulsar Roemer delays
  2. CMB dipole -- absolute velocity from the cosmic microwave background
  3. Fused EKF  -- 6-state position+velocity estimate (Joseph-form update)
  4. MADN       -- onboard AI agent monitoring with a tamper-evident audit log

Each step produces a measurement that feeds the EKF, and each step's agent
telemetry is screened by MADN and recorded to the HMAC-chained audit log.
"""

from __future__ import annotations

import numpy as np

from . import cmb_nav, madn, xnav
from .ekf import FusedEKF


class OrreryNavigator:
    """A single spacecraft running the fused navigation + AI-safety stack."""

    def __init__(
        self,
        initial_position_km: np.ndarray,
        initial_velocity_km_s: np.ndarray,
        audit_key: bytes = b"orrerynav-mission-key",
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.true_position_km = np.asarray(initial_position_km, dtype=float).copy()
        self.true_velocity_km_s = np.asarray(initial_velocity_km_s, dtype=float).copy()

        self.pulsar_directions = xnav.pulsar_directions()
        self.sky_directions = cmb_nav.sky_sample_directions()

        initial_state = np.concatenate([self.true_position_km, self.true_velocity_km_s])
        initial_covariance = np.diag([1e8, 1e8, 1e8, 1.0, 1.0, 1.0])
        self.ekf = FusedEKF(initial_state, initial_covariance, accel_psd=1e-13)

        self.madn_profiles = madn.default_class_profiles()
        self.audit_log = madn.HMACAuditLog(key=audit_key)
        self.history: list[dict] = []

    def _take_xnav_fix(self, timing_noise_s: float = 1e-6) -> dict:
        delays = xnav.simulate_roemer_delays(
            self.true_position_km, self.pulsar_directions, timing_noise_s, self.rng
        )
        estimate = xnav.solve_position(self.pulsar_directions, delays)
        gdop = xnav.geometric_dilution_of_precision(self.pulsar_directions)
        # Measurement covariance scales with timing noise and constellation geometry.
        sigma_km = timing_noise_s * xnav.C_KM_S * gdop
        covariance = np.eye(3) * sigma_km ** 2
        return {"estimate_km": estimate, "covariance_km2": covariance, "gdop": gdop}

    def _take_cmb_fix(self, noise_kelvin: float = 50e-6) -> dict:
        temperatures = cmb_nav.simulate_sky_temperatures(
            self.true_velocity_km_s, self.sky_directions, noise_kelvin, self.rng
        )
        estimate = cmb_nav.solve_velocity(self.sky_directions, temperatures)
        sigma_km_s = noise_kelvin / cmb_nav.T0_KELVIN * cmb_nav.C_KM_S
        covariance = np.eye(3) * sigma_km_s ** 2
        return {"estimate_km_s": estimate, "covariance_km2_s2": covariance}

    def _screen_agent_telemetry(self, step: int, features: np.ndarray) -> dict:
        noise = np.full(features.shape, 0.15)
        result = madn.monte_carlo_classify(features, noise, n_trials=200, rng=self.rng)
        self.audit_log.append(
            event="AgentTelemetryScreened",
            payload={
                "step": step,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "uncertainty": result["uncertainty"],
            },
        )
        return result

    def step(self, dt_s: float, agent_features: np.ndarray | None = None) -> dict:
        """Advances the mission by one cruise interval `dt_s` (seconds):
        propagates true dynamics, takes XNAV/CMB fixes, fuses them through
        the EKF, and screens onboard agent telemetry through MADN.
        """
        # True dynamics (constant velocity cruise, for simulation purposes).
        self.true_position_km = self.true_position_km + self.true_velocity_km_s * dt_s

        self.ekf.predict(dt_s)

        xnav_fix = self._take_xnav_fix()
        self.ekf.update_position(xnav_fix["estimate_km"], xnav_fix["covariance_km2"])

        cmb_fix = self._take_cmb_fix()
        self.ekf.update_velocity(cmb_fix["estimate_km_s"], cmb_fix["covariance_km2_s2"])

        if agent_features is None:
            agent_features = self.rng.normal(0.0, 1.0, size=len(madn.FEATURE_NAMES))
        madn_result = self._screen_agent_telemetry(len(self.history), agent_features)

        record = {
            "true_position_km": self.true_position_km.copy(),
            "true_velocity_km_s": self.true_velocity_km_s.copy(),
            "estimated_position_km": self.ekf.position.copy(),
            "estimated_velocity_km_s": self.ekf.velocity.copy(),
            "position_error_km": float(np.linalg.norm(self.ekf.position - self.true_position_km)),
            "velocity_error_km_s": float(np.linalg.norm(self.ekf.velocity - self.true_velocity_km_s)),
            "madn_class": madn_result["predicted_class"],
            "madn_confidence": madn_result["confidence"],
        }
        self.history.append(record)
        return record

    def run(self, n_steps: int, dt_s: float = 3600.0) -> list[dict]:
        for _ in range(n_steps):
            self.step(dt_s)
        return self.history

    def mission_report(self) -> dict:
        """A compact summary suitable for ground transmission once contact
        is restored: latest navigation state, error trend, and the
        tamper-evident MADN audit trail."""
        latest = self.history[-1] if self.history else None
        return {
            "steps_completed": len(self.history),
            "latest_position_km": latest["estimated_position_km"].tolist() if latest else None,
            "latest_velocity_km_s": latest["estimated_velocity_km_s"].tolist() if latest else None,
            "latest_position_error_km": latest["position_error_km"] if latest else None,
            "latest_velocity_error_km_s": latest["velocity_error_km_s"] if latest else None,
            "audit_chain_intact": self.audit_log.verify_chain(),
            "audit_log_excerpt": self.audit_log.to_compressed_report(last_n=5),
        }
