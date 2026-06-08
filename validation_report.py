"""Generates the validation numbers this codebase actually produces, with a
fixed seed for reproducibility. This is the authoritative source for any
"validated result" claims about ORRERYNAV: run this script and cite its
output, rather than citing figures that didn't come from a run of this code.

Usage: python validation_report.py
"""

import numpy as np

from orrerynav import cmb_nav, madn, xnav
from orrerynav.simulator import OrreryNavigator

SEED = 20260608
N_TRIALS = 500


def xnav_validation(rng):
    directions = xnav.pulsar_directions()
    gdop = xnav.geometric_dilution_of_precision(directions)
    errors = []
    for _ in range(N_TRIALS):
        true_position = rng.uniform(-2e8, 2e8, size=3)
        delays = xnav.simulate_roemer_delays(true_position, directions, timing_noise_s=1e-6, rng=rng)
        estimate = xnav.solve_position(directions, delays)
        errors.append(np.linalg.norm(estimate - true_position))
    return {
        "n_pulsars": len(directions),
        "gdop": gdop,
        "median_error_km": float(np.median(errors)),
        "p95_error_km": float(np.percentile(errors, 95)),
    }


def cmb_validation(rng):
    directions = cmb_nav.sky_sample_directions()
    truth = cmb_nav.planck_2018_velocity_vector()
    direction_errors = []
    speed_errors = []
    for _ in range(N_TRIALS):
        temperatures = cmb_nav.simulate_sky_temperatures(truth, directions, noise_kelvin=50e-6, rng=rng)
        estimate = cmb_nav.solve_velocity(directions, temperatures)
        direction_errors.append(cmb_nav.angular_error_deg(estimate, truth))
        speed_errors.append(abs(np.linalg.norm(estimate) - np.linalg.norm(truth)))
    return {
        "n_sky_samples": len(directions),
        "median_direction_error_deg": float(np.median(direction_errors)),
        "median_speed_error_km_s": float(np.median(speed_errors)),
    }


def ekf_validation(rng):
    navigator = OrreryNavigator(
        initial_position_km=np.array([2.2e8, 1.1e8, -5.0e7]),
        initial_velocity_km_s=np.array([12.0, -4.0, 1.5]),
        rng=rng,
    )
    history = navigator.run(n_steps=40, dt_s=3600.0)
    return {
        "n_steps": 40,
        "final_position_error_km": history[-1]["position_error_km"],
        "final_velocity_error_km_s": history[-1]["velocity_error_km_s"],
        "mean_last5_position_error_km": float(np.mean([h["position_error_km"] for h in history[-5:]])),
        "mean_last5_velocity_error_km_s": float(np.mean([h["velocity_error_km_s"] for h in history[-5:]])),
    }


def madn_validation(rng):
    profiles = madn.default_class_profiles()
    noise = np.full(len(madn.FEATURE_NAMES), 0.15)

    # False-reject rate: how often nominal telemetry is misclassified as a threat.
    nominal_trials = 200
    false_rejects = 0
    for _ in range(nominal_trials):
        result = madn.monte_carlo_classify(np.zeros(len(madn.FEATURE_NAMES)), noise, n_trials=100, rng=rng)
        if result["predicted_class"] != "nominal":
            false_rejects += 1

    # Red-team detection: every non-nominal signature should be caught and identified.
    threat_classes = [c for c in madn.THREAT_CLASSES if c != "nominal"]
    detections = 0
    correct_identifications = 0
    for threat_class in threat_classes:
        result = madn.monte_carlo_classify(profiles[threat_class].means, noise, n_trials=100, rng=rng)
        if result["predicted_class"] != "nominal":
            detections += 1
        if result["predicted_class"] == threat_class:
            correct_identifications += 1

    return {
        "false_reject_rate": false_rejects / nominal_trials,
        "redteam_detection": f"{detections}/{len(threat_classes)}",
        "redteam_correct_identification": f"{correct_identifications}/{len(threat_classes)}",
    }


def main():
    rng = np.random.default_rng(SEED)
    print(f"ORRERYNAV validation report (seed={SEED}, n_trials={N_TRIALS})")
    print("Run this script directly to reproduce; do not cite figures that")
    print("did not come from an actual run of this codebase.\n")

    print("XNAV (pulsar Roemer-delay positioning):")
    for k, v in xnav_validation(rng).items():
        print(f"  {k}: {v}")

    print("\nCMB dipole velocity navigation:")
    for k, v in cmb_validation(rng).items():
        print(f"  {k}: {v}")

    print("\nFused 6-state EKF (40-step cruise):")
    for k, v in ekf_validation(rng).items():
        print(f"  {k}: {v}")

    print("\nMADN behavioral classifier:")
    for k, v in madn_validation(rng).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
