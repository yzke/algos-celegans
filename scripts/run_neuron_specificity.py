"""Run the AC0.5.3 specificity battery and emit a human + machine report.

Usage:
    python3 scripts/run_neuron_specificity.py

Outputs:
    output/neuron_specificity_report.txt   — human-readable summary
    output/neuron_specificity_results.json — per-test results, including
                                              per-neuron ΔV
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from algos.config import OUTPUT_DIR
from algos.connectome import ConnectomeData
from algos.validation.neuron_specificity import (
    MIN_MAGNITUDE,
    default_test_battery,
    measure_specificity,
)


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    connectome = ConnectomeData.load()

    results = []
    t0 = time.time()
    for spec in default_test_battery():
        r = measure_specificity(connectome, **spec)
        results.append(r)
    elapsed = time.time() - t0

    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)

    # ---- Human-readable text report ----------------------------------------
    report_path = OUTPUT_DIR / "neuron_specificity_report.txt"
    with report_path.open("w") as f:
        f.write("AC0.5.3 — Key-neuron functional specificity report\n")
        f.write("=" * 60 + "\n")
        f.write(f"connectome:       Cook 2019 corrected, N={connectome.n_neurons}\n")
        f.write(f"dynamics:         tanh(β=1·V), τ=10, noise=0\n")
        f.write(f"pre-eq baseline:  max|V| ≈ {results[0].baseline_max_abs_v:.2e}\n")
        f.write(f"runtime:          {elapsed:.2f}s for {n_total} tests\n")
        f.write(f"min magnitude:    {MIN_MAGNITUDE} (responses below this → FAIL)\n")
        f.write("\n")
        f.write(f"Overall: {n_pass}/{n_total} tests pass\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(r.summary() + "\n")
            f.write(f"     notes: {r.notes}\n")
            # Sorted per-target table.
            items = sorted(r.per_target_dv.items(), key=lambda x: -abs(x[1]))
            for n, v in items:
                f.write(f"     {n:6s}  ΔV={v:+.4f}\n")
            f.write(
                f"     driven max|V| across whole network = "
                f"{r.driven_max_abs_v:.4f}\n\n"
            )

    # ---- Machine-readable JSON ---------------------------------------------
    json_path = OUTPUT_DIR / "neuron_specificity_results.json"
    payload = [
        {
            "name": r.name,
            "driver_neurons": r.driver_neurons,
            "target_neurons": r.target_neurons,
            "expected_sign": int(r.expected_sign),
            "drive_strength": float(r.drive_strength),
            "passed": bool(r.passed),
            "mean_dv": float(r.mean_dv),
            "max_abs_dv": float(r.max_abs_dv),
            "per_target_dv": {k: float(v) for k, v in r.per_target_dv.items()},
            "baseline_max_abs_v": float(r.baseline_max_abs_v),
            "driven_max_abs_v": float(r.driven_max_abs_v),
            "notes": r.notes,
        }
        for r in results
    ]
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    print(f"\n=== {n_pass}/{n_total} specificity tests pass "
          f"({elapsed:.2f}s) ===\n")
    for r in results:
        print(r.summary())

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
