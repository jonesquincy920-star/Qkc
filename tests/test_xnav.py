import numpy as np

from eamis import (
    C_KM_S,
    NAV_CONSTELLATION,
    geometric_dilution_of_precision,
    pulsar_directions,
    radec_to_unit_vector,
    simulate_roemer_delays,
    solve_position,
    xnav_position_fix,
    xnav_timing_noise,
)


def test_radec_to_unit_vector_is_normalized():
    v = radec_to_unit_vector(ra_hours=6.0, dec_deg=30.0)
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_pulsar_directions_are_unit_vectors():
    dirs = pulsar_directions()
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert dirs.shape[0] == len(NAV_CONSTELLATION)


def test_noiseless_position_recovery_is_exact():
    true_position = np.array([1.5e8, -2.3e7, 6.1e6])
    dirs = pulsar_directions()
    delays = simulate_roemer_delays(true_position, dirs, timing_noise_s=0.0)
    estimate = solve_position(dirs, delays)
    assert np.allclose(estimate, true_position, rtol=1e-9)


def test_roemer_delay_matches_definition():
    position = np.array([1.0, 0.0, 0.0]) * C_KM_S  # 1 light-second along x
    dirs = np.array([[1.0, 0.0, 0.0]])
    delays = simulate_roemer_delays(position, dirs)
    assert np.isclose(delays[0], 1.0)


def test_gdop_is_positive_and_in_range():
    gdop = geometric_dilution_of_precision(pulsar_directions())
    assert 1.0 < gdop < 2.0


def test_position_fix_at_microsecond_noise_is_sub_kilometer():
    rng = np.random.default_rng(42)
    true_position = np.array([2.2e8, 1.1e8, -5.0e7])
    result = xnav_position_fix(true_position, timing_noise_s=1e-6, rng=rng)
    assert result["error_km"] < 1.0
    assert result["gdop"] > 0


def test_timing_noise_grows_with_distance():
    assert xnav_timing_noise(50.0) > xnav_timing_noise(1.0)
