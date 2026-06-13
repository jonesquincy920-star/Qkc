import numpy as np

from eamis import (
    C_KM_S,
    T0_KELVIN,
    angular_error_deg,
    cmb_velocity_fix,
    galactic_to_unit_vector,
    planck_2018_velocity_vector,
    simulate_sky_temperatures,
    sky_sample_directions,
    solve_velocity,
    PLANCK_2018_SPEED_KM_S,
)


def test_galactic_unit_vector_is_normalized():
    v = galactic_to_unit_vector(lon_deg=264.02, lat_deg=48.25)
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_planck_2018_velocity_matches_quoted_speed():
    v = planck_2018_velocity_vector()
    assert np.isclose(np.linalg.norm(v), PLANCK_2018_SPEED_KM_S)


def test_sky_sample_directions_are_unit_vectors_and_well_spread():
    dirs = sky_sample_directions(32)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert np.linalg.matrix_rank(dirs) == 3


def test_dipole_temperature_matches_definition():
    velocity = np.array([1.0, 0.0, 0.0]) * C_KM_S * 0.01
    dirs = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    temps = simulate_sky_temperatures(velocity, dirs)
    assert np.isclose(temps[0], T0_KELVIN * 1.01)
    assert np.isclose(temps[1], T0_KELVIN * 0.99)


def test_noiseless_velocity_recovery_is_exact():
    true_velocity = planck_2018_velocity_vector()
    dirs = sky_sample_directions(32)
    temps = simulate_sky_temperatures(true_velocity, dirs, noise_kelvin=0.0)
    estimate = solve_velocity(dirs, temps)
    assert np.allclose(estimate, true_velocity, rtol=1e-6)


def test_angular_error_zero_for_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert np.isclose(angular_error_deg(v, v), 0.0)


def test_velocity_fix_at_50uK_noise_is_sub_half_degree():
    rng = np.random.default_rng(7)
    result = cmb_velocity_fix(planck_2018_velocity_vector(), noise_kelvin=50e-6, rng=rng)
    assert result["direction_error_deg"] < 0.5
