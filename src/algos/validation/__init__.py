"""Phase 0.5 validation harness.

Modules:
  - ``neuron_specificity``: AC0.5.3 — drive single sensory/command neurons in
    the bare CTRNN and verify activity propagates to anatomically-expected
    downstream targets, with the literature-reported sign.
  - ``comparison``: AC0.5.2 — three quantitative similarity metrics for
    digital-vs-real activity matrices.
  - ``reference_data``: AC0.5.1 — uniform interface to real C. elegans
    whole-brain calcium-imaging datasets (Atanas 2023, Kato 2015, ...).

Phase 0 code under ``src/algos/connectome.py`` and ``src/algos/neural/`` is
intentionally not modified by this validation pass — the validation runs the
existing dynamics on the existing connectome.
"""
