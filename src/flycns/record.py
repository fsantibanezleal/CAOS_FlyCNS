"""Recordings of a run, in the compiled-directory style: a manifest with hashes and little-endian arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .compiled import read_compiled, write_compiled
from .dynamics.lif import Run

SCHEMA = "flycns.recording/1"


def write_recording(directory: Path, run: Run, meta: dict | None = None) -> dict:
    """Write a run's spikes (a CSR over steps) and traces; ``meta`` lands in the manifest's ``release`` field."""
    arrays = {
        "tick_indptr": np.asarray(run.tick_indptr, dtype=np.int64),
        "neuron_index": np.asarray(run.neuron_index, dtype=np.int32),
        "trace_neurons": np.asarray(run.trace_neurons, dtype=np.int32),
        "traces_mv": np.asarray(run.traces_mv, dtype=np.float32),
    }
    counts = {"neurons": run.n_neurons, "steps": run.steps, "dt_ms": run.dt_ms, "spikes": int(len(run.neuron_index))}
    return write_compiled(directory, arrays, {"release": meta or {}, "counts": counts}, schema=SCHEMA)


def read_recording(directory: Path) -> Run:
    rec = read_compiled(Path(directory), schema=SCHEMA)
    c = rec.counts
    return Run(n_neurons=int(c["neurons"]), steps=int(c["steps"]), dt_ms=float(c["dt_ms"]),
               tick_indptr=rec["tick_indptr"], neuron_index=rec["neuron_index"], trace_neurons=rec["trace_neurons"],
               traces_mv=rec["traces_mv"])
