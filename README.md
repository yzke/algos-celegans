# ALGOS-Celegans

Phase 0 implementation of the ALGOS-Celegans project: a 302-neuron C. elegans
connectome simulated as a continuous-time recurrent neural network (CTRNN).

This phase contains **only** the neural skeleton — no body, no environment, no
modulators, no plasticity. See `docs/design.md` and `docs/phase0.md` for the
master design and Phase 0 acceptance criteria.

## Quick start

```bash
pip install -e .[dev]

# 1. Verify connectome loads and statistics match Cook 2019
pytest tests/

# 2. Run a 5000-tick simulation and write output/basic_simulation.png
python scripts/run_basic_simulation.py
```

## Data

Cook et al. 2019 connectome data must live in `data/connectome/`. See
`data/connectome/README.md` for the exact files and where to download them.

## Layout

- `src/algos/connectome.py` — load `SI 5 Connectome adjacency matrices,
  corrected July 2020.xlsx` into a 302×302 `ConnectomeData` object.
- `src/algos/neural/` — CTRNN state and dynamics.
- `src/algos/viz/` — minimal activity visualization.
- `tests/` — pytest suite covering AC1–AC3.
- `scripts/run_basic_simulation.py` — end-to-end demo producing the AC4 figure.

Project context, decisions, and the Phase 0 report live in `DECISIONS.md`,
`QUESTIONS.md`, and `PHASE0_REPORT.md` at the repo root.
