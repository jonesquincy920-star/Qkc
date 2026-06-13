import numpy as np

from eamis import OrreryNavigator, default_class_profiles


def test_navigator_runs_and_velocity_converges():
    rng = np.random.default_rng(2024)
    nav = OrreryNavigator(np.array([1.0e8, 2.0e8, -5.0e7]),
                          np.array([12.0, -4.0, 1.5]), rng=rng)
    history = nav.run(40, 3600.0)
    assert len(history) == 40
    early = np.mean([h["velocity_error_km_s"] for h in history[:5]])
    late = np.mean([h["velocity_error_km_s"] for h in history[-5:]])
    assert late < early
    assert late < 0.01


def test_navigator_audit_chain_intact():
    rng = np.random.default_rng(7)
    nav = OrreryNavigator(np.array([5.0e7, -1.0e8, 3.0e7]),
                          np.array([8.0, 3.0, -1.0]), rng=rng)
    nav.run(10, 3600.0)
    assert len(nav.audit_log.entries) == 10
    assert nav.audit_log.verify_chain()


def test_mission_report_is_complete_and_intact():
    rng = np.random.default_rng(11)
    nav = OrreryNavigator(np.array([2.0e8, 0.0, 0.0]),
                          np.array([0.0, 15.0, 0.0]), rng=rng)
    nav.run(5, 1800.0)
    report = nav.mission_report()
    assert report["steps_completed"] == 5
    assert report["audit_chain_intact"] is True
    assert len(report["latest_position_km"]) == 3
    assert report["gdop"] > 0


def test_compromised_agent_telemetry_is_flagged():
    rng = np.random.default_rng(5)
    nav = OrreryNavigator(np.array([1.0e8, 1.0e8, 1.0e8]),
                          np.array([5.0, 5.0, 5.0]), rng=rng)
    compromise_sig = default_class_profiles()["compromise"].means
    record = nav.step(3600.0, agent_features=compromise_sig)
    assert record["madn_class"] == "compromise"
    assert nav.audit_log.entries[-1].payload["predicted_class"] == "compromise"
    assert nav.audit_log.verify_chain()
