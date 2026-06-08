import numpy as np

from orrerynav.xnav import (
    C_KM_S,
    geometric_dilution_of_precision,
    position_fix,
    pulsar_directions,
    radec_to_unit_vector,
    simulate_roemer_delays,
    solve_position,
)


def test_radec_to_unit_vector_is_normalized():
    v = radec_to_unit_vector(ra_hours=6.0, dec_deg=30.0)
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_pulsar_directions_are_unit_vectors():
    dirs = pulsar_directions()
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0)
    assert dirs.shape[0] >= 3


def test_noiseless_position_recovery_is_exact():
    true_position = np.array([1.5e8, -2.3e7, 6.1e6])  # km, ~1 AU scale
    dirs = pulsar_directions()
    delays = simulate_roemer_delays(true_position, dirs, timing_noise_s=0.0)
    estimate = solve_position(dirs, delays)
    assert np.allclose(estimate, true_position, rtol=1e-9)


def test_roemer_delay_matches_definition():
    position = np.array([1.0, 0.0, 0.0]) * C_KM_S  # 1 light-second away along x
    dirs = np.array([[1.0, 0.0, 0.0]])
    delays = simulate_roemer_delays(position, dirs)
    assert np.isclose(delays[0], 1.0)


def test_gdop_is_positive_and_finite():
    gdop = geometric_dilution_of_precision(pulsar_directions())
    assert gdop > 0
    assert np.isfinite(gdop)


def test_position_fix_at_microsecond_noise_is_sub_kilometer():
    rng = np.random.default_rng(42)
    true_position = np.array([2.2e8, 1.1e8, -5.0e7])  # km
    result = position_fix(true_position, timing_noise_s=1e-6, rng=rng)
    assert result["error_km"] < 1.0
    assert result["gdop"] > 0
