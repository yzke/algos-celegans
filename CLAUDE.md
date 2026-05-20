# ALGOS-Celegans

Autopoietic digital organism — C. elegans connectome × ALGOS framework.

## Project Structure
- docs/design.md — master design doc v0.2 (theoretical anchors, architecture, interface specs)
- docs/phase0.md — Phase 0 taskbook (what to implement now)
- src/algos/ — Python package
- tests/ — pytest suite
- data/connectome/ — connectome data
- output/ — generated figures and results
- logs/ — session logs

## Phase 0 Scope
Neural skeleton ONLY — no body, no environment, no modulators, no plasticity.
- Load Cook 2019 connectome
- Build W_chem and W_gap matrices
- Implement CTRNN dynamics
- Stability tests (10^5 ticks no NaN)
- Activity visualization

## Tech Stack
Python 3.11+, NumPy (primary), pandas, matplotlib, pytest.
uv or poetry for deps.
Single tick < 1ms on CPU. No GPU needed.
