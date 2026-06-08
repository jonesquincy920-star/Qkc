"""Fused 6-state Extended Kalman Filter combining XNAV position fixes and
CMB dipole velocity fixes into a single absolute position+velocity estimate
in the ICRS frame.

State vector: x = [px, py, pz, vx, vy, vz] (km, km/s).
Process model: constant-velocity kinematics with additive process noise.
Measurement model: direct observation of position (from XNAV) or velocity
(from CMB dipole), so H is a fixed selection matrix and the filter is
linear-Gaussian — but we use the Joseph-form covariance update for the
numerical robustness an EKF needs when models become nonlinear.
"""

from __future__ import annotations

import numpy as np

STATE_DIM = 6

# Measurement model selects either the position block or the velocity block.
H_POSITION = np.hstack([np.eye(3), np.zeros((3, 3))])
H_VELOCITY = np.hstack([np.zeros((3, 3)), np.eye(3)])


def state_transition(dt_s: float) -> np.ndarray:
    """Constant-velocity state transition matrix F for time step `dt_s`."""
    f = np.eye(STATE_DIM)
    f[0:3, 3:6] = np.eye(3) * dt_s
    return f


def process_noise(dt_s: float, accel_psd: float) -> np.ndarray:
    """Discrete white-noise-acceleration process noise covariance Q.

    `accel_psd` is the power spectral density of unmodeled acceleration
    (km^2/s^3); larger values mean less trust in the constant-velocity model.
    """
    q = np.zeros((STATE_DIM, STATE_DIM))
    dt2 = dt_s ** 2
    dt3 = dt_s ** 3
    dt4 = dt_s ** 4
    for i in range(3):
        j = i + 3
        q[i, i] = dt4 / 4 * accel_psd
        q[i, j] = q[j, i] = dt3 / 2 * accel_psd
        q[j, j] = dt2 * accel_psd
    return q


class FusedEKF:
    """6-state EKF fusing XNAV position and CMB-dipole velocity measurements."""

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, accel_psd: float = 1e-13):
        self.state = np.asarray(initial_state, dtype=float).copy()
        self.covariance = np.asarray(initial_covariance, dtype=float).copy()
        self.accel_psd = accel_psd

    def predict(self, dt_s: float) -> None:
        f = state_transition(dt_s)
        q = process_noise(dt_s, self.accel_psd)
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q

    def _update(self, measurement: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        """Joseph-form measurement update, numerically robust against the
        covariance losing positive-definiteness from rounding error:

            P+ = (I - K H) P (I - K H)^T + K R K^T
        """
        innovation = measurement - h @ self.state
        innovation_cov = h @ self.covariance @ h.T + r
        kalman_gain = self.covariance @ h.T @ np.linalg.inv(innovation_cov)

        self.state = self.state + kalman_gain @ innovation

        identity = np.eye(STATE_DIM)
        a = identity - kalman_gain @ h
        self.covariance = a @ self.covariance @ a.T + kalman_gain @ r @ kalman_gain.T

    def update_position(self, position_km: np.ndarray, covariance_km2: np.ndarray) -> None:
        self._update(np.asarray(position_km, dtype=float), H_POSITION, np.asarray(covariance_km2, dtype=float))

    def update_velocity(self, velocity_km_s: np.ndarray, covariance_km2_s2: np.ndarray) -> None:
        self._update(np.asarray(velocity_km_s, dtype=float), H_VELOCITY, np.asarray(covariance_km2_s2, dtype=float))

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]
