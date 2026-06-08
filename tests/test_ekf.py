import numpy as np

from eamis import FusedEKF, _H_POS, _H_VEL, _process_noise, _state_transition


def test_state_transition_propagates_constant_velocity():
    f = _state_transition(dt_s=10.0)
    state = np.array([0, 0, 0, 1.0, 2.0, 3.0])
    assert np.allclose(f @ state, [10.0, 20.0, 30.0, 1.0, 2.0, 3.0])


def test_process_noise_is_symmetric_positive_semidefinite():
    q = _process_noise(dt_s=5.0, accel_psd=1e-9)
    assert np.allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) >= -1e-15)


def test_covariance_stays_symmetric_positive_definite():
    rng = np.random.default_rng(0)
    ekf = FusedEKF(np.zeros(6), np.eye(6) * 1e6)
    for _ in range(20):
        ekf.predict(60.0)
        ekf.update_position(rng.normal(size=3) * 1e3, np.eye(3))
        ekf.update_velocity(rng.normal(size=3) * 0.1, np.eye(3) * 1e-4)
        assert np.allclose(ekf.covariance, ekf.covariance.T, atol=1e-6)
        assert np.all(np.linalg.eigvalsh(ekf.covariance) > 0)


def test_filter_converges_to_constant_truth():
    rng = np.random.default_rng(123)
    true_pos = np.array([1.0e8, 2.0e8, -5.0e7])
    true_vel = np.array([10.0, -5.0, 2.0])
    ekf = FusedEKF(np.zeros(6), np.eye(6) * 1e10)
    dt = 60.0
    for _ in range(40):
        true_pos = true_pos + true_vel * dt
        ekf.predict(dt)
        ekf.update_position(true_pos + rng.normal(scale=1.0, size=3), np.eye(3))
        ekf.update_velocity(true_vel + rng.normal(scale=0.05, size=3), np.eye(3) * 0.0025)
    assert np.linalg.norm(ekf.velocity - true_vel) < 0.1
    assert np.linalg.norm(ekf.position - true_pos) < 5.0


def test_measurement_selection_matrices():
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(_H_POS @ state, [1.0, 2.0, 3.0])
    assert np.allclose(_H_VEL @ state, [4.0, 5.0, 6.0])


def test_uncertainty_accessors_are_positive():
    ekf = FusedEKF(np.zeros(6), np.eye(6) * 1e4)
    assert ekf.position_uncertainty_km() > 0
    assert ekf.velocity_uncertainty_km_s() > 0
