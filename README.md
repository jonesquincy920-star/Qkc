# ORRERYNAV / EAMIS

**Enneract Autonomous Mission Intelligence Stack** — a single-file,
numpy-only simulation of infrastructure-free deep-space navigation fused
with onboard AI safety monitoring.

The entire implementation lives in **`eamis.py`** (~980 lines, no
dependencies beyond numpy).  Tests live in `tests/`.

---

## The problem

Every NASA deep-space mission depends on the Deep Space Network (DSN) for
navigation.  Beyond ~10 AU the round-trip light time breaks any meaningful
real-time navigation loop; beyond 50 AU it exceeds 14 hours.  A spacecraft
must navigate itself — using only what the universe provides everywhere.

---

## The solution: EAMIS

Four integrated modules, all implemented in `eamis.py`:

### 1. XNAV — X-ray pulsar positioning
Recovers absolute spacecraft position from the Roemer delay between a
millisecond pulsar's pulse and the Solar System Barycenter:
`dt = (p_hat · r) / c`.  A 9-pulsar ATNF constellation with GDOP < 1.2
gives sub-km position fixes at 1 µs timing precision.

### 2. CMB dipole navigation
Recovers absolute velocity from the Doppler dipole the spacecraft's motion
imprints on the CMB sky: `T(n) = T₀(1 + v·n/c)`.  32 Fibonacci-sphere
sky samples at 50 µK radiometer noise give ~0.30° direction accuracy —
referenced against Planck 2018, no ground contact required.

### 3. Fused 6-state EKF
Combines XNAV position fixes and CMB-dipole velocity fixes in a
constant-velocity Extended Kalman Filter with a Joseph-form covariance
update (numerical robustness over long cruise phases).

### 4. MADN — Multi-Agent Defense Network
Monitors onboard AI agent telemetry against a 7-class threat taxonomy
(nominal / behavioral drift / deception / prompt injection / covert
channel / data exfiltration / compromise) using:
- Naive-Bayes classification on 4 z-scored telemetry features
- Monte Carlo uncertainty quantification (N=500 noisy redraws)
- HMAC-SHA256-chained tamper-evident audit log

---

## Usage

```bash
pip install numpy pytest
python eamis.py                  # integrated 40-step cruise demo
python eamis.py --selftest       # 30 embedded regression tests
python eamis.py --validate       # reproducible validated-results report
pytest tests/ -q                 # 50 pytest tests (red-team sweep included)
```

---

## Validated results (`python eamis.py --validate`, seed 20260608, N=500)

These figures are generated directly from `eamis.py` with a fixed seed.
Run `python eamis.py --validate` to reproduce them exactly.

| Module | Metric | Result |
|---|---|---|
| XNAV (9-pulsar ATNF constellation) | GDOP | 1.17 |
| XNAV | median position error @ 1 µs timing noise | 0.29 km |
| XNAV | p95 position error @ 1 µs timing noise | 0.57 km |
| CMB dipole (32 sky samples) | median direction error @ 50 µK | 0.30° |
| Fused EKF | mean position error, steps 36–40 | 0.30 km |
| Fused EKF | mean velocity error, steps 36–40 | < 0.001 km/s |
| MADN | red-team detection (non-nominal flagged) | 6/6 |
| MADN | false-reject rate (nominal telemetry) | 0% |

---

## Status: TRL 2 — simulation validated

All four modules are independently tested and their combined behaviour is
validated in the `OrreryNavigator` integration tests.  The embedded
`--selftest` suite is designed to fail loudly on regression.

Phase I work will close the gap to TRL 4:
- Replace synthetic pulsar timing noise with real ATNF timing residuals
- Characterise CMB noise model against WMAP 9-year and Planck 2018 data
- Prototype FPGA EKF running at 10 Hz on a navigation-class processor
- Extend MADN threat profiles from red-team telemetry rather than hand specs
