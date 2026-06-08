import numpy as np

from orrerynav.simulator import OrreryNavigator


def test_navigator_runs_and_converges_position_and_velocity():
    rng = np.random.default_rng(2024)
    navigator = OrreryNavigator(
        initial_position_km=np.array([1.0e8, 2.0e8, -5.0e7]),
        initial_velocity_km_s=np.array([12.0, -4.0, 1.5]),
        rng=rng,
    )

    history = navigator.run(n_steps=40, dt_s=3600.0)
    assert len(history) == 40

    early_velocity_error = np.mean([h["velocity_error_km_s"] for h in history[:5]])
    late_velocity_error = np.mean([h["velocity_error_km_s"] for h in history[-5:]])
    assert late_velocity_error < early_velocity_error
    assert late_velocity_error < 0.2


def test_navigator_screens_agents_and_produces_intact_audit_chain():
    rng = np.random.default_rng(7)
    navigator = OrreryNavigator(
        initial_position_km=np.array([5.0e7, -1.0e8, 3.0e7]),
        initial_velocity_km_s=np.array([8.0, 3.0, -1.0]),
        rng=rng,
    )

    navigator.run(n_steps=10, dt_s=3600.0)

    assert len(navigator.audit_log.entries) == 10
    assert navigator.audit_log.verify_chain()
    for entry in navigator.audit_log.entries:
        assert entry.event == "AgentTelemetryScreened"


def test_mission_report_summarizes_state_and_audit_integrity():
    rng = np.random.default_rng(11)
    navigator = OrreryNavigator(
        initial_position_km=np.array([2.0e8, 0.0, 0.0]),
        initial_velocity_km_s=np.array([0.0, 15.0, 0.0]),
        rng=rng,
    )
    navigator.run(n_steps=5, dt_s=1800.0)

    report = navigator.mission_report()
    assert report["steps_completed"] == 5
    assert report["audit_chain_intact"] is True
    assert len(report["latest_position_km"]) == 3
    assert len(report["latest_velocity_km_s"]) == 3


def test_compromised_agent_telemetry_is_flagged_and_logged():
    from orrerynav import madn

    rng = np.random.default_rng(5)
    navigator = OrreryNavigator(
        initial_position_km=np.array([1.0e8, 1.0e8, 1.0e8]),
        initial_velocity_km_s=np.array([5.0, 5.0, 5.0]),
        rng=rng,
    )

    compromise_signature = madn.default_class_profiles()["compromise"].means
    record = navigator.step(dt_s=3600.0, agent_features=compromise_signature)

    assert record["madn_class"] == "compromise"
    assert navigator.audit_log.entries[-1].payload["predicted_class"] == "compromise"
    assert navigator.audit_log.verify_chain()
