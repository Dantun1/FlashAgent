# Hephaestus
Project for AI Expo 2026: Optimising Agentic Caching.

## Visualisation pipeline
### Step 1 — baseline (every query generates from scratch)
`python3 benchmarks.py --baseline --output data/baseline.csv --n {NUMBER_OF_QUERIES}`

### Step 2 — cached run (normal APC behaviour)
`python3 benchmarks.py --output data/cache_telemetry.csv --n {NUMBER_OF_QUERIES}`

### Step 3 — generate all plots
`python3 visualise.py --baseline data/baseline.csv --cached data/cache_telemetry.csv`
