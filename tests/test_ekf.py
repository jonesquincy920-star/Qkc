import numpy as np

from orrerynav.ekf import FusedEKF, H_POSITION, H_VELOCITY, process_noise, state_transition


def test_state_transition_propagates_constant_velocity():
    f = state_transition(dt_s=10.0)
    state = np.array([0, 0, 0, 1.0, 2.0, 3.0])
    propagated = f @ state
    assert np.allclose(propagated, [10.0, 20.0, 30.0, 1.0, 2.0, 3.0])


def test_process_noise_is_symmetric_positive_semidefinite():
    q = process_noise(dt_s=5.0, accel_psd=1e-9)
    assert np.allclose(q, q.T)
    eigenvalues = np.linalg.eigvalsh(q)
    assert np.all(eigenvalues >= -1e-15)


def test_covariance_stays_symmetric_positive_definite_through_updates():
    rng = np.random.default_rng(0)
    state = np.zeros(6)
    covariance = np.eye(6) * 1e6
    ekf = FusedEKF(state, covariance, accel_psd=1e-12)

    for _ in range(20):
        ekf.predict(dt_s=60.0)
        ekf.update_position(rng.normal(size=3) * 1e3, np.eye(3) * 1.0)
        ekf.update_velocity(rng.normal(size=3) * 0.1, np.eye(3) * 1e-4)

        assert np.allclose(ekf.covariance, ekf.covariance.T, atol=1e-6)
        eigenvalues = np.linalg.eigvalsh(ekf.covariance)
        assert np.all(eigenvalues > 0)


def test_filter_converges_to_constant_truth():
    rng = np.random.default_rng(123)
    true_position = np.array([1.0e8, 2.0e8, -5.0e7])
    true_velocity = np.array([10.0, -5.0, 2.0])

    state = np.zeros(6)
    covariance = np.eye(6) * 1e10
    ekf = FusedEKF(state, covariance, accel_psd=1e-12)

    position_noise = np.eye(3) * (1.0 ** 2)   # 1 km position noise
    velocity_noise = np.eye(3) * (0.05 ** 2)  # 0.05 km/s velocity noise

    dt = 60.0
    for step in range(40):
        true_position = true_position + true_velocity * dt
        ekf.predict(dt)

        position_meas = true_position + rng.normal(scale=1.0, size=3)
        velocity_meas = true_velocity + rng.normal(scale=0.05, size=3)
        ekf.update_position(position_meas, position_noise)
        ekf.update_velocity(velocity_meas, velocity_noise)

    velocity_error = np.linalg.norm(ekf.velocity - true_velocity)
    position_error = np.linalg.norm(ekf.position - true_position)

    assert velocity_error < 0.1
    assert position_error < 5.0


def test_measurement_selection_matrices_select_expected_blocks():
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(H_POSITION @ state, [1.0, 2.0, 3.0])
    assert np.allclose(H_VELOCITY @ state, [4.0, 5.0, 6.0])
