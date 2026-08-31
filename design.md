# PitMind — Design

## Decisions & Defaults

| Decision | Choice | Notes |
|---|---|---|
| Simulator | ACC (PC) | Shared memory via `pyaccsharedmemory` |
| MVP scope | Offline CSV analysis | Live telemetry (v0.6) later |
| Tracks | All F1 circuits | Track-agnostic detection; no per-track config |
| Data acquisition | Own recorder → CSV | Avoid fragile MoTeC `.ld` export |
| Dev/test data | Synthetic generator | Realistic laps with *injected, known* mistakes for pytest |
| Reference lap | Best lap (+ best corner, best sector for potential lap) | Configurable |
| Mistake detection | Threshold rules (doc §14) | Thresholds in config; ML later (doc §26) |
| Time loss | Kinematic heuristic | ML later once labeled data exists |
| Coaching | Templates | Optional LLM hook isolated (doc §21) |
| UI | Streamlit + Plotly | Quick MVP dashboards |

## CSV Telemetry Schema (the contract)

Emitter: recorder AND synthetic generator. Column names fixed.

```text
timestamp                 float     seconds since start of recording
lap_number                int       completed lap counter
sector                    int       1..3
track_position            float     0..1 around the lap (ACC normalized_car_position)
speed_kmh                 float
throttle                  0..1
brake                     0..1
steering                  float     -1..1 (ACC convention)
gear                      int
rpm                       float
x, y, z                   float     world coordinates (meters)
```

Extra columns allowed (optional) but the pipeline requires the ones above.

## Derived / Corner Dataset Schema

Produced by `features.py` per lap per corner (doc §25):

```text
lap
corner_index                int
brake_point_m               distance marker (or track_position) where braking starts
entry_speed_kmh
apex_speed_kmh
exit_speed_kmh
throttle_on_s               seconds after apex when throttle resumed (>threshold)
steering_max                max |steering| in corner
corner_time_s
time_loss_s                 estimated lost seconds vs reference
```

## Mistake Classes

```text
EARLY_BRAKING
LATE_BRAKING
LOW_ENTRY_SPEED
LOW_APEX_SPEED
LATE_THROTTLE
POOR_CORNER_EXIT
EXCESSIVE_STEERING
POOR_LINE
INCONSISTENT_INPUT
```

## Threshold Defaults (config.yaml)

```yaml
ranges:
  brake_point_delta_m:      # early/late braking severity
    siginificant: 3.0
    potential:    10.0
    strong:       15.0
  apex_speed_delta_kmh: 5.0
  throttle_delay_s: 0.1
  exit_speed_delta_kmh: 3.0
  steering_excess: 0.15

detection:
  min_brake_pressure: 0.2      # brake event >= this = event start
  throttle_resume: 0.3         # throttle above this counts as "on"
  sample_rate_hz: 60           # resample target

timeloss:
  mode: kinematic              # heuristic until ML model exists
```

Thresholds are placeholders until validated against real ACC laps — never magic numbers
scattered in code.

## Coaching Style (doc §19, §38)

Bad: "You seem to have braked earlier than usual at Turn 3..."
Better: "Brake 10 meters later into Turn 3."
Best: "Carry 9 km/h more speed into Turn 3."

- Precision over conversation.
- Short, actionable, non-repetitive.
- Not every mistake triggers feedback — priority system (doc §18).

## Success Criteria (doc §35)

1. Ingest ACC telemetry (recorder + CSV).
2. Identify corners and driving events automatically.
3. Detect meaningful deviations vs reference.
4. Identify likely cause of lost time.
5. Estimate lost time per corner/lap + potential lap.
6. Generate concise coaching feedback.
7. Offline end-to-end latency acceptable (fast enough to later go real-time).
8. **The real test:** following the advice measurably improves lap time.

## MVP Acceptance (doc §29)

System can say reliably:

> **Turn 3: Brake 12m later. Estimated loss 0.18s.**

backed by a Streamlit dashboard showing the evidence (plots, corner table, mistake cards).