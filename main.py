import numpy as np

from orrerynav import OrreryNavigator


def main():
    rng = np.random.default_rng(42)
    navigator = OrreryNavigator(
        initial_position_km=np.array([2.2e8, 1.1e8, -5.0e7]),  # ~ 2 AU from SSB
        initial_velocity_km_s=np.array([12.0, -4.0, 1.5]),
        rng=rng,
    )

    history = navigator.run(n_steps=40, dt_s=3600.0)

    print("EAMIS / ORRERYNAV cruise simulation — 40 hourly steps\n")
    for step in (0, 9, 19, 29, 39):
        record = history[step]
        print(
            f"step {step:2d}: "
            f"pos error {record['position_error_km']:8.3f} km | "
            f"vel error {record['velocity_error_km_s']:.4f} km/s | "
            f"agent screen -> {record['madn_class']} "
            f"(confidence {record['madn_confidence']:.2f})"
        )

    print("\n--- MISSION REPORT (ground transmission excerpt) ---")
    report = navigator.mission_report()
    for key, value in report.items():
        if key == "audit_log_excerpt":
            print(f"{key}: <{len(value)} bytes, signatures included>")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
