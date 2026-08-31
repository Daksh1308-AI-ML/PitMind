# 🏎️ AI Driver Coach

> **An AI-powered real-time race engineer that analyzes driving telemetry, detects performance issues, estimates lost lap time, and provides actionable coaching feedback.**

---

## 1. Project Overview

**AI Driver Coach** is an intelligent driving-performance assistant designed for racing simulators.

The system continuously analyzes telemetry generated while a driver is racing and uses machine learning, time-series analysis, and performance modeling to understand how the driver is driving the car.

Instead of waiting until the end of a session to inspect telemetry graphs, the AI Driver Coach can identify mistakes as they happen and provide short, actionable feedback such as:

> 🔊 "Brake later into Turn 3."

> 🔊 "You're carrying too little speed into the apex."

> 🔊 "Earlier throttle on corner exit."

> 🔊 "You lost approximately 0.18 seconds through Turn 7."

The long-term goal is to make the system behave like a **virtual race engineer** sitting beside the driver.

---

# 2. Problem Statement

Racing simulators provide enormous amounts of telemetry data, but raw telemetry is difficult for most drivers to interpret.

A driver may know that they are slower through a particular corner, but determining **why** they are slower requires analyzing several signals simultaneously:

- Speed
- Brake pressure
- Throttle
- Steering
- Gear
- RPM
- Position
- Racing line
- Acceleration
- Corner entry
- Apex speed
- Corner exit

Traditional telemetry tools generally show this information through graphs and overlays.

The driver still has to interpret the data themselves.

### The problem

> **How can we automatically analyze a driver's telemetry, identify exactly where performance is being lost, determine why it is happening, and communicate a useful correction in real time?**

---

# 3. Core Idea

The project follows a simple pipeline:

```text
                    Racing Simulator
                           │
                           ▼
                    Telemetry Data
                           │
                           ▼
                  Telemetry Processor
                           │
                           ▼
                 Driving Event Detection
                           │
                           ▼
                  Driver Behavior Model
                           │
                           ▼
                  Reference Comparison
                           │
                           ▼
                    Mistake Detection
                           │
                           ▼
                  Time-Loss Estimation
                           │
                           ▼
                     Coaching Engine
                       /          \
                      /            \
                     ▼              ▼
               Voice Coach      Dashboard
```

The important concept is:

> **Telemetry → Understanding → Diagnosis → Coaching**

The system should not simply visualize telemetry.

It should **understand what the driver is doing and tell them what to improve.**

---

# 4. Example Scenario

Imagine the driver approaches **Turn 3**.

The telemetry shows:

```text
Current Lap

Braking Point:     123 m
Entry Speed:       135 km/h
Apex Speed:        109 km/h
Throttle On:       51%
Corner Exit:       118 km/h
```

The driver's best lap shows:

```text
Best Lap

Braking Point:     110 m
Entry Speed:       142 km/h
Apex Speed:        118 km/h
Throttle On:       72%
Corner Exit:       125 km/h
```

The AI detects:

```text
Brake Point Difference = +13 m
Apex Speed Difference  = -9 km/h
Throttle Delay         = +0.18 sec
```

The system estimates:

```text
Potential Time Loss = 0.21 sec
```

The coach then generates:

> **"Turn 3: you're braking around 13 meters too early. Try carrying more speed into the corner and get back on throttle earlier."**

This is the core experience the project is trying to achieve.

---

# 5. Project Goals

## Primary Goals

### 1. Analyze telemetry

Process real-time or recorded racing telemetry.

### 2. Detect driving events

Automatically identify:

- Braking zones
- Corner entry
- Apex
- Corner exit
- Throttle application
- Gear changes
- Acceleration zones

### 3. Detect performance mistakes

Identify problems such as:

- Braking too early
- Braking too late
- Low corner-entry speed
- Low apex speed
- Late throttle
- Excessive steering
- Poor racing line
- Inconsistent driving
- Poor corner exits

### 4. Estimate time loss

Estimate how much lap time the driver loses because of a particular mistake.

### 5. Provide actionable feedback

Convert telemetry analysis into concise coaching instructions.

### 6. Support real-time coaching

Eventually provide feedback while the driver is racing.

---

# 6. Non-Goals

The first version should **not** attempt to solve everything.

The MVP should avoid:

- Building a complete racing simulator
- Controlling the car automatically
- Autonomous racing
- Perfect racing-line optimization
- Supporting every simulator
- Building a massive LLM system
- Training extremely large deep-learning models
- Creating a complex UI before the core ML works

The focus should remain:

> **Understand the driver → find lost time → explain how to improve.**

---

# 7. Target Users

### Sim Racing Beginners

Users who don't understand telemetry and want simple coaching.

### Intermediate Drivers

Drivers trying to improve lap times and consistency.

### Advanced Drivers

Drivers who want detailed analysis of braking points, corner speeds, and throttle application.

### Sim Racing Enthusiasts

Users interested in data-driven performance analysis.

---

# 8. Key Features

## Feature 1 — Live Telemetry

The system receives telemetry continuously.

Possible signals:

```text
timestamp
speed
throttle
brake
steering
gear
rpm
x_position
y_position
z_position
lap
sector
fuel
engine_temperature
```

The exact telemetry fields will depend on the simulator.

---

# 9. Feature 2 — Telemetry Processing

Raw telemetry will first pass through a preprocessing pipeline.

### Processing steps

```text
Raw telemetry
      ↓
Missing-value handling
      ↓
Noise filtering
      ↓
Synchronization
      ↓
Normalization
      ↓
Feature extraction
      ↓
ML / rule-based analysis
```

Potential processing techniques:

- Rolling averages
- Signal smoothing
- Outlier detection
- Interpolation
- Normalization
- Resampling

---

# 10. Feature 3 — Driving Event Detection

The system needs to understand what is happening during a lap.

For example:

```text
Straight
   ↓
Braking
   ↓
Corner Entry
   ↓
Apex
   ↓
Corner Exit
   ↓
Acceleration
   ↓
Straight
```

These events can initially be detected using rules.

Example:

```text
if brake > threshold:
    braking_event = True
```

Later, machine-learning models can replace or improve these rules.

---

# 11. Feature 4 — Corner Detection

Each corner should be treated as an individual performance unit.

Example:

```text
Turn 1
Turn 2
Turn 3
Turn 4
...
Turn 15
```

For every corner, the system can calculate:

- Brake point
- Brake pressure
- Entry speed
- Minimum speed
- Apex speed
- Throttle application point
- Exit speed
- Steering behavior
- Corner duration
- Time loss

This allows the AI to say:

> "Turn 11 is currently your biggest source of lost time."

---

# 12. Feature 5 — Driver Baseline

The system should learn what the driver considers their normal or good performance.

Possible baselines:

### Best Lap

Compare the current lap against the driver's fastest lap.

### Best Sector

Use the driver's best performance in each sector.

### Best Corner

Use the driver's best execution of each corner.

### Session Average

Measure consistency against the driver's average performance.

This enables personalized coaching.

---

# 13. Feature 6 — Reference Lap Comparison

The current lap is aligned against a reference lap.

Example:

```text
                 Best Lap
                    │
                    ▼
              Reference Data
                    │
                    │
Current Lap ────────┤
                    │
                    ▼
              Difference Analysis
```

The system compares:

```text
Brake Point
Entry Speed
Apex Speed
Throttle Point
Exit Speed
Steering
Corner Duration
```

---

# 14. Feature 7 — Mistake Detection

The AI should classify driving behavior.

Example classes:

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

A mistake should only be reported when the evidence is strong enough.

For example:

```text
Brake point difference < 3m
        ↓
Probably insignificant

Brake point difference > 10m
        ↓
Potential issue

Brake point difference > 15m
        ↓
Strong issue
```

The exact thresholds should eventually be learned from data rather than hard-coded.

---

# 15. Feature 8 — Time-Loss Estimation

One of the most valuable features is estimating how much time is being lost.

Example:

```text
Turn 3

Early braking       -0.08s
Low apex speed      -0.07s
Late throttle       -0.06s

Estimated loss      -0.21s
```

The system can then calculate:

```text
Current Lap       1:42.381
Potential Lap     1:41.720

Potential Gain    0.661s
```

This gives the driver a clear target.

---

# 16. Feature 9 — Potential Lap

Instead of only showing the driver's current lap time, the system estimates the theoretical performance achievable from their best corner/sector executions.

For example:

```text
Current Lap
1:42.381

Best Sector 1
+0.000

Best Sector 2
-0.231

Best Sector 3
-0.412

Potential Lap
1:41.738
```

The system can tell the driver:

> **"You currently have a 1:41.74 in your driving."**

This is highly motivating because it shows the driver where their pace is hiding.

---

# 17. Feature 10 — Real-Time Coaching

The coach should provide short feedback during the lap.

Examples:

```text
"Brake later."

"Hold the apex."

"More speed into Turn 5."

"Earlier throttle."

"Good exit."

"You're losing time on entry."

"Stay wider on corner entry."
```

The feedback must be:

- Short
- Relevant
- Timely
- Actionable
- Non-repetitive

The system should avoid overwhelming the driver.

---

# 18. Coaching Priority System

Not every mistake deserves immediate feedback.

A priority system can determine whether the AI should speak.

Example:

```text
Minor mistake
     ↓
No voice feedback

Moderate mistake
     ↓
Optional feedback

Major mistake
     ↓
Immediate feedback

Repeated major mistake
     ↓
High-priority coaching
```

Priority can depend on:

```text
Magnitude of mistake
        +
Time lost
        +
Frequency
        +
Confidence
```

---

# 19. AI Coach Personality

The AI should behave more like a professional race engineer than a chatbot.

### Bad

> "You seem to have braked earlier than usual at Turn 3. Perhaps you could consider trying to brake slightly later."

### Better

> **"Brake 10 meters later into Turn 3."**

### After the lap

> **"Turn 3 cost you 0.21 seconds. You braked early and got back on throttle late."**

The AI should prioritize **precision over conversation**.

---

# 20. Machine Learning Architecture

The project can use multiple models rather than one giant model.

## Layer 1 — Signal Processing

Purpose:

> Clean and transform telemetry.

Possible techniques:

- Filtering
- Smoothing
- Feature engineering

---

## Layer 2 — Event Detection

Purpose:

> Understand what part of the corner/lap the driver is currently in.

Possible approaches:

- Rule-based logic
- Classification models
- Time-series models

---

## Layer 3 — Driver Modeling

Purpose:

> Learn how the driver normally drives.

Possible models:

- Random Forest
- XGBoost
- Gradient Boosting
- Clustering

---

## Layer 4 — Performance Prediction

Purpose:

> Predict expected speed/time for a given situation.

Possible models:

- Regression
- Gradient Boosting
- Neural networks

---

## Layer 5 — Sequential Modeling

For advanced versions:

- LSTM
- GRU
- Temporal CNN
- Transformer

These models can analyze sequences of telemetry rather than individual data points.

---

# 21. Where the LLM Fits

The LLM should **not be responsible for detecting driving mistakes directly**.

Instead:

```text
Telemetry
   ↓
ML Models
   ↓
Structured Insight
   ↓
LLM
   ↓
Natural Language Coaching
```

Example ML output:

```json
{
  "corner": "Turn 3",
  "issue": "early_braking",
  "brake_delta_m": 13,
  "apex_speed_delta_kmh": -9,
  "estimated_time_loss": 0.21,
  "confidence": 0.94
}
```

The LLM converts that into:

> **"You're braking about 13 meters too early into Turn 3. Carry more speed into the corner and get back on throttle earlier."**

This makes the architecture much more reliable.

---

# 22. Dashboard

The dashboard should provide deeper analysis than the voice coach.

Possible sections:

### Live Telemetry

```text
Speed       ███████████
Throttle    ████████
Brake       ███
Steering    █████
```

### Track Map

Show:

- Current position
- Racing line
- Braking points
- Problem corners

### Corner Analysis

```text
Turn 3

Brake Point       +13m
Entry Speed       -7 km/h
Apex Speed        -9 km/h
Throttle          +0.18s late
Time Loss         0.21s
```

### Lap Comparison

Overlay:

```text
Current Lap
Best Lap
Reference Lap
```

---

# 23. Technology Stack

The stack can evolve as the project grows.

## Programming

**Python**

Primary language for:

- Data processing
- ML
- Telemetry analysis
- Backend

## Data Processing

Potential tools:

- NumPy
- Pandas
- Polars
- SciPy

## Machine Learning

Potential tools:

- Scikit-learn
- XGBoost
- PyTorch

## Backend

Possible:

- FastAPI
- WebSockets

## Database

Possible:

- PostgreSQL
- SQLite for MVP
- TimescaleDB for large telemetry datasets

## Visualization

Possible:

- Plotly
- Streamlit
- React
- Dash

## Real-Time Communication

Possible:

- WebSockets
- UDP

## Voice

Potential:

- Text-to-speech API
- Local TTS engine

---

# 24. Data Strategy

Data is one of the biggest challenges in this project.

The system needs telemetry from racing sessions.

Possible sources:

### Option 1 — Simulator Telemetry

Use telemetry APIs/SDKs provided by supported racing simulators.

### Option 2 — Recorded Telemetry

Use CSV/JSON telemetry recordings.

### Option 3 — Self-generated Dataset

Collect many laps and label:

```text
Corner
Brake point
Entry speed
Apex speed
Throttle point
Exit speed
Lap time
Driver
Track
Car
```

### Option 4 — Synthetic Data

Generate controlled examples to test models.

---

# 25. Dataset Structure

A possible telemetry dataset:

```text
timestamp
lap_number
sector
track_position
speed
throttle
brake
steering
gear
rpm
x
y
z
acceleration
corner_id
event
```

A processed corner dataset could look like:

```text
lap
corner
brake_point
entry_speed
apex_speed
exit_speed
throttle_point
steering_angle
corner_time
time_loss
driver
```

---

# 26. Training Strategy

The project should start simple.

## Stage 1

Use deterministic rules.

Example:

```text
Brake point > reference + threshold
        ↓
Early braking
```

## Stage 2

Train classical ML models.

Example:

```text
Input:
speed
brake
throttle
steering
position

Output:
driving mistake
```

## Stage 3

Train regression models.

Predict:

```text
expected_corner_time
expected_exit_speed
potential_time_loss
```

## Stage 4

Introduce sequential models.

Use telemetry windows:

```text
Previous 5 seconds
       ↓
LSTM / Transformer
       ↓
Driving state
```

---

# 27. Evaluation Metrics

The project should measure more than model accuracy.

## Mistake Detection

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Time Prediction

- MAE
- RMSE
- R²

## Coaching Quality

Measure:

- Detection latency
- False coaching rate
- Correct coaching rate
- Time saved after following advice

The most important real-world metric is:

> **Does following the AI's advice actually improve lap time?**

---

# 28. Example Evaluation

Suppose the driver completes:

```text
Before coaching:

Average Lap = 1:42.80
```

After following AI feedback:

```text
Average Lap = 1:42.12
```

Improvement:

```text
0.68 seconds
```

This becomes strong evidence that the AI isn't merely analyzing telemetry — it is actually improving driver performance.

---

# 29. MVP Definition

The first working version should be intentionally small.

### MVP Scope

Support:

- One racing simulator
- One track
- One car
- Recorded telemetry
- One reference lap
- 3–5 corners
- Basic mistake detection
- Basic time-loss estimation
- Simple dashboard

### MVP Feedback

The system should be able to say:

> **Turn 3: Brake 12m later. Estimated loss: 0.18s.**

If this works reliably, the project has achieved its core objective.

---

# 30. Version Roadmap

## Version 0.1 — Telemetry Analyzer

Input:

```text
CSV telemetry
```

Output:

```text
Graphs
Lap statistics
Corner statistics
```

---

## Version 0.2 — Corner Intelligence

Add:

- Corner detection
- Braking point detection
- Apex detection
- Throttle analysis
- Reference comparison

---

## Version 0.3 — Mistake Detection

Add:

- Early braking
- Late braking
- Low apex speed
- Late throttle
- Poor corner exit

---

## Version 0.4 — Time-Loss Model

Add:

- Corner time prediction
- Time-loss estimation
- Potential lap calculation

---

## Version 0.5 — AI Coach

Add:

- Coaching engine
- Feedback prioritization
- Natural-language generation

---

## Version 0.6 — Real-Time Telemetry

Add:

```text
Simulator
    ↓
Live telemetry
    ↓
Streaming processor
    ↓
AI
```

---

## Version 1.0 — Real-Time AI Race Engineer

Final MVP product experience:

```text
Driver races
      ↓
AI analyzes telemetry
      ↓
Mistake detected
      ↓
Time loss estimated
      ↓
AI decides whether feedback is necessary
      ↓
Voice instruction
      ↓
Driver improves
```

---

# 31. Advanced Features

Once the core system works, several advanced features can be added.

## Driver Fingerprint

Build a personalized model of the driver's habits.

Example:

> "You consistently brake early in medium-speed corners."

---

## Adaptive Coaching

The AI learns whether the driver responds better to:

- Aggressive coaching
- Conservative coaching
- Detailed feedback
- Short commands

---

## Automatic Racing Line Analysis

Analyze:

```text
Entry line
Apex position
Exit line
```

and identify inefficient racing lines.

---

## Tire Management

Analyze:

- Tire temperature
- Tire degradation
- Slip
- Cornering behavior

Then coach:

> "You're overheating the front-left through Turns 5–7."

---

## Fuel Strategy

Estimate:

- Fuel consumption
- Remaining laps
- Optimal fuel strategy

---

## Setup Recommendations

Eventually correlate driving behavior with:

- Brake bias
- Downforce
- Suspension
- Differential
- Tire pressure

The system could recommend setup changes based on observed behavior.

---

# 32. Long-Term Vision

The ultimate product is not just an AI dashboard.

It is a **virtual race engineer**.

The driver should be able to start a session and forget about the technical details.

The AI handles:

```text
Telemetry
    ↓
Driver Analysis
    ↓
Performance Diagnosis
    ↓
Strategy
    ↓
Coaching
    ↓
Progress Tracking
```

After every session, it should understand:

- What the driver did well
- Where they lost time
- Why they lost time
- What they should practice
- Whether they improved
- What to focus on next

---

# 33. Example Final User Experience

### Driver starts a session

> **AI Driver Coach:**  
> "Session started. I'll focus on braking consistency and corner exits."

### During the lap

> **"Brake later into Turn 3."**

> **"Good apex."**

> **"Earlier throttle."**

### After the lap

> **Lap: 1:42.381**

> "You lost most of your time in Turns 3, 7 and 11."

```text
Turn 3     -0.21s
Turn 7     -0.14s
Turn 11    -0.24s
```

> "Your biggest opportunity is Turn 11. You're braking 14 meters early and getting back on throttle 0.2 seconds late."

### After several laps

> "Your average lap improved by 0.63 seconds. Your Turn 3 braking is now consistent with your best lap."

This creates a complete **AI-powered learning loop**.

---

# 34. Core Technical Challenge

The hardest part of this project is **not the LLM**.

The real challenge is correctly understanding racing telemetry.

The project should therefore prioritize:

```text
Telemetry Quality
       ↓
Event Detection
       ↓
Reference Alignment
       ↓
Performance Modeling
       ↓
Mistake Detection
       ↓
Time-Loss Estimation
       ↓
Coaching
```

If the underlying telemetry analysis is wrong, even the best LLM will produce bad coaching.

---

# 35. Project Success Criteria

The project can be considered successful when it can reliably:

1. Receive racing telemetry.
2. Identify corners and driving events.
3. Detect meaningful deviations from a reference.
4. Identify the likely cause of lost time.
5. Estimate the approximate time lost.
6. Generate concise coaching feedback.
7. Deliver feedback with low enough latency for real-time use.
8. Demonstrate that following the advice can improve lap performance.

The ultimate test:

> **Can a driver become measurably faster by using the AI Driver Coach?**

---

# 36. One-Line Project Description

> **AI Driver Coach is a real-time AI race engineer that analyzes simulator telemetry, detects driving mistakes, estimates lost lap time, and provides personalized coaching to help drivers become faster.**

---

# 37. Portfolio / Resume Positioning

A strong portfolio description could eventually be:

> **Built an AI-powered real-time racing coach that analyzes high-frequency simulator telemetry, detects corner-level driving errors, estimates time loss against reference laps, and generates actionable voice feedback for performance improvement.**

The project demonstrates skills in:

```text
Python
Data Engineering
Time-Series Analysis
Machine Learning
Feature Engineering
Real-Time Systems
Streaming Data
Model Evaluation
AI/LLMs
APIs
Data Visualization
```

---

# 38. Guiding Principle

The entire project should follow one rule:

> ### Don't just tell the driver what happened. Tell them what to do next.

Bad:

> "Your apex speed was 109 km/h."

Better:

> "You're 9 km/h slower at the apex."

Best:

> **"Carry 9 km/h more speed into Turn 3."**

That difference is what turns a **telemetry analyzer** into an **AI Driver Coach**.

---

# Development Notes (addendum — does not change the roadmap above)

## Synthetic test data uses real F1 circuit geometry

- Dev/test telemetry is no longer generated on a made-up track. `synthetic/generator.py` now drives
  **real circuit centerlines** vendored from [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits)
  (MIT; OSM-derived, attribution in `data/circuits/README.md`). Default: **Monza**; also Spa,
  Silverstone (+ optional Imola).
- Circuit GeoJSON (lat/lon) is projected to meters, resampled to a ~1 m arc grid, and its corner
  regions are auto-derived with the same chord-curvature logic the analysis pipeline uses on
  telemetry. The discrete-corner kinematic model still brakes/apexes/exits per corner and still
  injects known, labeled mistakes for pytest.
- **Analysis remains track-agnostic** (architect.md rule 1): circuit files are a *fixture* for
  generating test telemetry only; runtime corner detection never reads them. This keeps the roadmap
  goal "any F1 circuit, no config" intact while making dev/test data realistic and validation
  comparable to real ACC recordings (Monza, Spa, Silverstone, Imola are all in ACC).