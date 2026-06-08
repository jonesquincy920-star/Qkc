"""XNAV: X-ray pulsar navigation via Roemer delay.

Millisecond pulsars emit X-ray pulses with ~microsecond timing precision.
The Roemer delay between a pulse's arrival at the spacecraft and at the
Solar System Barycenter (SSB) is

    dt = (p_hat . r) / c

where p_hat is the unit vector toward the pulsar, r is the spacecraft's
position relative to the SSB, and c is the speed of light. Observing N >= 3
pulsars from non-degenerate directions yields a linear system that is solved
for r by least squares.
"""

from __future__ import annotations

import numpy as np

C_KM_S = 299_792.458  # speed of light, km/s


def radec_to_unit_vector(ra_hours: float, dec_deg: float) -> np.ndarray:
    """Converts equatorial coordinates (RA in hours, Dec in degrees) to an
    ICRS-frame unit vector."""
    ra_rad = np.deg2rad(ra_hours * 15.0)
    dec_rad = np.deg2rad(dec_deg)
    return np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])


# A small constellation of well-known millisecond pulsars (RA hours, Dec deg),
# representative of ATNF catalog entries used for XNAV studies.
PULSAR_CATALOG = {
    "J0437-4715": (4.616, -47.253),
    "J0613-0200": (6.220, -2.007),
    "J1024-0719": (10.408, -7.327),
    "J1744-1134": (17.741, -11.574),
    "J2124-3358": (21.408, -33.967),
}


def pulsar_directions(catalog: dict | None = None) -> np.ndarray:
    """Returns an (N, 3) array of unit vectors toward each cataloged pulsar."""
    catalog = catalog or PULSAR_CATALOG
    return np.array([radec_to_unit_vector(ra, dec) for ra, dec in catalog.values()])


def simulate_roemer_delays(
    position_km: np.ndarray,
    directions: np.ndarray,
    timing_noise_s: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulates measured Roemer delays (seconds) for a spacecraft at
    `position_km` relative to the SSB, observing pulsars along `directions`.

    `timing_noise_s` is the 1-sigma timing measurement noise (e.g. 1e-6 for
    the ~1 microsecond precision quoted for millisecond pulsars).
    """
    delays = (directions @ position_km) / C_KM_S
    if timing_noise_s > 0:
        rng = rng or np.random.default_rng()
        delays = delays + rng.normal(0.0, timing_noise_s, size=delays.shape)
    return delays


def solve_position(directions: np.ndarray, delays_s: np.ndarray) -> np.ndarray:
    """Recovers spacecraft position (km, relative to SSB) from Roemer delay
    measurements via least squares: directions @ r = c * delays."""
    b = C_KM_S * delays_s
    position, *_ = np.linalg.lstsq(directions, b, rcond=None)
    return position


def geometric_dilution_of_precision(directions: np.ndarray) -> float:
    """GDOP for a constellation of pulsar directions: sqrt(trace((A^T A)^-1)).

    Lower GDOP means the pulsar geometry is better conditioned for a position
    fix; values near 1-2 are considered good.
    """
    gram = directions.T @ directions
    return float(np.sqrt(np.trace(np.linalg.inv(gram))))


def position_fix(
    true_position_km: np.ndarray,
    timing_noise_s: float = 1e-6,
    catalog: dict | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """End-to-end XNAV position fix: simulate measurements from a known true
    position, then recover position and report the residual error and GDOP."""
    directions = pulsar_directions(catalog)
    delays = simulate_roemer_delays(true_position_km, directions, timing_noise_s, rng)
    estimate = solve_position(directions, delays)
    error_km = float(np.linalg.norm(estimate - true_position_km))
    return {
        "estimate_km": estimate,
        "error_km": error_km,
        "gdop": geometric_dilution_of_precision(directions),
    }
