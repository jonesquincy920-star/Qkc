# ORRERYNAV

ORRERYNAV is a numpy-only simulation of the **Enneract Autonomous Mission
Intelligence Stack (EAMIS)**: an infrastructure-free deep-space navigation
system fused with onboard AI safety monitoring, designed for spacecraft
operating beyond useful Deep Space Network (DSN) range (>10 AU).

It combines four capabilities, each independently testable and composable:

## Capabilities

### 1. XNAV — X-ray pulsar positioning (`orrerynav/xnav.py`)
Recovers absolute spacecraft position from the Roemer delay between a
millisecond pulsar's pulse arrival at the spacecraft and at the Solar System
Barycenter: `dt = (p_hat . r) / c`. Observing a constellation of cataloged
pulsars yields a linear least-squares position fix, with GDOP reported as a
geometry-quality metric.

### 2. CMB dipole navigation (`orrerynav/cmb_nav.py`)
Recovers absolute velocity from the Doppler dipole the CMB exhibits along the
spacecraft's direction of motion: `T(n_hat) = T0 (1 + v.n_hat / c)`. Sampling
sky temperature in several directions yields a velocity fix referenced
against the Planck 2018 solar dipole (369.82 km/s toward l=264.02°,
b=48.25°) — no ground contact required.

### 3. Fused 6-state EKF (`orrerynav/ekf.py`)
Fuses XNAV position fixes and CMB-dipole velocity fixes into a single
position+velocity state estimate (ICRS frame) via an Extended Kalman Filter
with a Joseph-form covariance update, chosen for its numerical robustness
against covariance loss of positive-definiteness.

### 4. MADN — Multi-Agent Defense Network (`orrerynav/madn.py`)
A spacecraft-local Bayesian behavioral classifier that screens onboard AI
agent telemetry against a 7-class threat taxonomy (nominal, behavioral
drift, deception, prompt injection, covert channel, data exfiltration,
compromise), reports Monte Carlo uncertainty over repeated noisy redraws,
and records every classification to an HMAC-chained, tamper-evident audit
log suitable for compressed transmission once ground contact is restored.

## Integration (`orrerynav/simulator.py`)
`OrreryNavigator` runs the full mission loop: propagate true dynamics, take
an XNAV fix and a CMB-dipole fix, fuse them through the EKF, screen agent
telemetry through MADN, and append to the audit log — producing a
`mission_report()` summary suitable for ground transmission.

## Usage

```bash
pip install -r requirements.txt
python main.py              # runs a 40-step cruise simulation and prints a mission report
pytest tests/ -q            # runs the full test suite, including a MADN red-team sweep
python validation_report.py # regenerates the validated numbers below from this exact code
```

## Validated results (reproduce with `python validation_report.py`, seed 20260608)

These figures come directly from running this codebase — they are the only
numbers that should be cited as "validated" for this implementation. Re-run
the script to reproduce them exactly (it is seeded and deterministic).

| Module | Metric | Result |
|---|---|---|
| XNAV (9-pulsar constellation) | GDOP | 1.17 |
| XNAV | median position error @ 1 µs timing noise | 0.29 km (p95: 0.57 km) |
| CMB dipole (32 sky samples) | median direction error @ 50 µK noise | 0.30° |
| CMB dipole | median speed error @ 50 µK noise | 1.15 km/s |
| Fused 6-state EKF | position error after 40 hourly steps | 0.45 km |
| Fused 6-state EKF | velocity error after 40 hourly steps | 0.00028 km/s |
| MADN | false-reject rate (nominal telemetry) | 0% |
| MADN | red-team detection (non-nominal flagged) | 6/6 |
| MADN | red-team correct identification | 6/6 |

(MADN's 7-class taxonomy includes `nominal`; the red-team sweep tests the
six non-nominal threat signatures, hence 6/6.)

## Status

This is a simulation-stage implementation (TRL 2): all four modules are
validated independently and in combination by the test suite in `tests/`,
including a red-team sweep that confirms every non-nominal threat signature
is correctly flagged and classified under noisy observation. It mirrors the
architecture described in the EAMIS NIAC Phase I concept (XNAV + CMB fusion
via a 6-state EKF, MADN behavioral monitoring), with simplified, documented
models standing in for flight-validated noise/threat profiles — the gap
Phase I work would close with real ATNF timing residuals, hardware-in-loop
EKF prototyping, and red-team-derived MADN class profiles.
