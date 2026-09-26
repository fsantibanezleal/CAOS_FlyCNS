#!/usr/bin/env python3
"""Write the whole-CNS scenario the browser engines are measured on, and the NumPy reference's output for it.

The published LIF on the whole MaleCNS v1.0 (weights exact, float64), under the moderate drive (the 57 gustatory
neurons of types LB1a to LB1e at 150 Hz), 2,000 steps (200 ms), seed 0. ``scripts/measure_browser.mjs`` then runs the
TypeScript CPU engine and the WebGPU engine on it and compares both with the reference (SDD section 7).

``strong/`` holds the second regime: the 1,428 neurons of the gustatory class at 150 Hz for 5,000 steps (500 ms),
where the network may switch into its high state at a random moment, so single runs are compared as samples of one
process: the reference's per-neuron rates for seeds 0 to 9, and the 5th percentile of the correlation between the
mean rates of five of those seeds and the other five, over the 126 splits that put seed 0 in the first half.

Usage: python scripts/write_whole_cns_scenario.py COMPILED OUT_DIR
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

from flycns.bundle import Scenario, lif_arrays, lif_constants, run_reference, write_expected, write_scenario
from flycns.compiled import read_compiled, write_compiled
from flycns.dynamics import Drive, LIFParams, LIFReference, synaptic_weights

LB1 = ["LB1a", "LB1b", "LB1c", "LB1d", "LB1e"]
STEPS = 2000
SEED = 0
STRONG_STEPS = 5000
STRONG_SEEDS = 10


def mean_rate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """As tests/test_lif.py: the correlation of mean rates over the neurons active in either set of runs."""
    mx, my = x.mean(0), y.mean(0)
    either = (mx > 0) | (my > 0)
    return float(np.corrcoef(mx[either], my[either])[0, 1])


def main(compiled: Path, out: Path) -> None:
    c = read_compiled(compiled, verify=False)
    weights = synaptic_weights(c["csr_indptr"], c["csr_indices"], c["csr_count"], c["neuron_sign"], 0.275)
    types = np.array(c.strings["type"], dtype=object)[c["neuron_type"]]
    lb1 = np.flatnonzero(np.isin(types, LB1))
    if len(lb1) != 57:
        raise SystemExit(f"expected the 57 LB1a-LB1e neurons, found {len(lb1)}")
    params = LIFParams()
    n = len(c["csr_indptr"]) - 1
    scenario = Scenario(
        arrays=lif_arrays(c["csr_indptr"], c["csr_indices"], weights, exact=True),
        constants={"engine": "lif", "lif": lif_constants(params), "n_neurons": n,
                   "release_name": "MaleCNS v1.0", "drive_name": "moderate: LB1a-LB1e at 150 Hz"},
        steps=STEPS, seed=SEED, drive=Drive(activate={int(i): 150.0 for i in lb1}))
    write_scenario(out, scenario)
    t0 = time.time()
    outputs = run_reference(scenario)
    seconds = time.time() - t0
    write_expected(out / "expected", outputs, {"engine": "NumPy reference", "seconds": round(seconds, 1)})
    spikes = len(outputs["neuron_index"])
    active = len(np.unique(outputs["neuron_index"]))
    print(f"{n} neurons, {len(c['csr_indices'])} connections; reference: {spikes} spikes from {active} neurons "
          f"in {seconds:.1f} s")

    classes = np.array(c.strings["class"], dtype=object)[c["neuron_class"]]
    gustatory = np.flatnonzero(classes == "gustatory")
    reference = LIFReference(c["csr_indptr"], c["csr_indices"], weights, params)
    strong = Drive(activate={int(i): 150.0 for i in gustatory})
    t0 = time.time()
    rates = np.stack([reference.run(STRONG_STEPS, strong, seed=s).rates_hz() for s in range(STRONG_SEEDS)])
    splits = []
    for half in itertools.combinations(range(STRONG_SEEDS), STRONG_SEEDS // 2):
        if 0 in half:
            other = [s for s in range(STRONG_SEEDS) if s not in half]
            splits.append(mean_rate_correlation(rates[list(half)], rates[other]))
    write_compiled(out / "strong", {"gustatory": np.asarray(gustatory, dtype=np.int64),
                                    "reference_rates_hz": rates.astype(np.float32)},
                   {"release": {"steps": STRONG_STEPS, "rate_hz": 150.0, "seeds": STRONG_SEEDS,
                                "splits": len(splits), "split_p5": float(np.percentile(splits, 5)),
                                "split_median": float(np.median(splits)),
                                "reference_spikes": [int(round(r.sum() * STRONG_STEPS * params.dt_ms / 1000))
                                                     for r in rates],
                                "seconds": round(time.time() - t0, 1)}},
                   schema="flycns.expected/1")
    print(f"strong drive: {len(gustatory)} gustatory neurons, {STRONG_SEEDS} reference runs in "
          f"{time.time() - t0:.0f} s; split 5th percentile {np.percentile(splits, 5):.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
