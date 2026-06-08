"""CMB dipole navigation: absolute velocity from the cosmic microwave
background's Doppler dipole.

Motion relative to the CMB rest frame imprints a dipole anisotropy on the
observed sky temperature:

    T(n_hat) = T0 * (1 + (v . n_hat) / c)

where n_hat is the observation direction, v is the spacecraft's velocity
relative to the CMB frame, and T0 is the isotropic CMB temperature. Sampling
the sky in several directions yields a linear system for v, recoverable by
least squares without any ground contact.

Reference value (Planck 2018): |v_sun| = 369.82 km/s toward galactic
longitude l = 264.02 deg, latitude b = 48.25 deg.
"""

from __future__ import annotations

import numpy as np

C_KM_S = 299_792.458  # speed of light, km/s
T0_KELVIN = 2.72548  # CMB monopole temperature (Fixsen 2009)

# Planck 2018 best-fit solar-system dipole velocity.
PLANCK_2018_SPEED_KM_S = 369.82
PLANCK_2018_GALACTIC_LON_DEG = 264.02
PLANCK_2018_GALACTIC_LAT_DEG = 48.25


def galactic_to_unit_vector(lon_deg: float, lat_deg: float) -> np.ndarray:
    """Converts galactic coordinates (longitude l, latitude b) to a unit
    vector in the galactic frame."""
    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)
    return np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])


def planck_2018_velocity_vector() -> np.ndarray:
    """The reference CMB dipole velocity vector (km/s, galactic frame)."""
    direction = galactic_to_unit_vector(
        PLANCK_2018_GALACTIC_LON_DEG, PLANCK_2018_GALACTIC_LAT_DEG
    )
    return PLANCK_2018_SPEED_KM_S * direction


def sky_sample_directions(n_samples: int = 12) -> np.ndarray:
    """Returns unit vectors roughly evenly spread over the sky (Fibonacci
    sphere) at which to sample the CMB temperature."""
    indices = np.arange(n_samples)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    lat = np.arcsin(1 - 2 * (indices + 0.5) / n_samples)
    lon = 2 * np.pi * indices / phi
    return np.array([
        [np.cos(lat[i]) * np.cos(lon[i]), np.cos(lat[i]) * np.sin(lon[i]), np.sin(lat[i])]
        for i in range(n_samples)
    ])


def simulate_sky_temperatures(
    velocity_km_s: np.ndarray,
    directions: np.ndarray,
    noise_kelvin: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulates measured sky temperatures along `directions` for a
    spacecraft moving at `velocity_km_s` relative to the CMB frame.

    `noise_kelvin` is the 1-sigma radiometer noise (e.g. 50e-6 for 50 uK).
    """
    temperatures = T0_KELVIN * (1.0 + (directions @ velocity_km_s) / C_KM_S)
    if noise_kelvin > 0:
        rng = rng or np.random.default_rng()
        temperatures = temperatures + rng.normal(0.0, noise_kelvin, size=temperatures.shape)
    return temperatures


def solve_velocity(directions: np.ndarray, temperatures_kelvin: np.ndarray) -> np.ndarray:
    """Recovers the velocity vector (km/s) from sky temperature samples via
    least squares on T/T0 - 1 = (n_hat . v) / c."""
    rhs = C_KM_S * (temperatures_kelvin / T0_KELVIN - 1.0)
    velocity, *_ = np.linalg.lstsq(directions, rhs, rcond=None)
    return velocity


def angular_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Angle (degrees) between the estimated and true velocity directions."""
    cos_angle = np.dot(estimate, truth) / (np.linalg.norm(estimate) * np.linalg.norm(truth))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def velocity_fix(
    true_velocity_km_s: np.ndarray,
    noise_kelvin: float = 50e-6,
    n_samples: int = 12,
    rng: np.random.Generator | None = None,
) -> dict:
    """End-to-end CMB dipole velocity fix: simulate sky samples for a known
    true velocity, then recover velocity and report direction/speed error."""
    directions = sky_sample_directions(n_samples)
    temperatures = simulate_sky_temperatures(true_velocity_km_s, directions, noise_kelvin, rng)
    estimate = solve_velocity(directions, temperatures)
    return {
        "estimate_km_s": estimate,
        "speed_error_km_s": float(abs(np.linalg.norm(estimate) - np.linalg.norm(true_velocity_km_s))),
        "direction_error_deg": angular_error_deg(estimate, true_velocity_km_s),
    }
