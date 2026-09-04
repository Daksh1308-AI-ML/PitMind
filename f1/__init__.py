"""PitMind F1 data bridge.

Converts real Formula 1 telemetry (FastF1 / public timing data) into the same
13-column CSV contract the pipeline consumes, so the full analysis core
(corners -> mistakes -> timeloss -> coaching) runs on real F1 drivers with zero
analysis changes (architect.md "F1 data bridge", design.md "Reaching real F1").

FastF1/OpenF1 are educational / non-commercial (CC BY-NC-SA). "F1 official"
here means professional-grade analysis of public F1 data.
"""
