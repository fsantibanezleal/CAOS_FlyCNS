"""A recording reads back as the run it came from, and a changed byte is caught."""

from __future__ import annotations

import numpy as np
import pytest

from flycns.compiled import CompiledError, read_compiled
from flycns.dynamics import Drive, LIFReference, synaptic_weights
from flycns.record import SCHEMA, read_recording, write_recording


def test_recording_round_trips(tmp_path):
    indptr = np.array([0, 1, 2, 2], dtype=np.int64)
    indices = np.array([1, 2], dtype=np.int64)
    weights = synaptic_weights(indptr, indices, np.array([100, 100]), np.array([1, 1, 1]), 0.275)
    run = LIFReference(indptr, indices, weights).run(3000, Drive(activate={0: 200.0}), seed=3,
                                                     trace_neurons=np.array([1, 2]))
    assert len(run.neuron_index) > 50
    manifest = write_recording(tmp_path / "rec", run, {"engine": "reference", "seed": 3})
    assert manifest["schema"] == SCHEMA
    assert manifest["release"] == {"engine": "reference", "seed": 3}
    assert manifest["counts"]["spikes"] == len(run.neuron_index)
    back = read_recording(tmp_path / "rec")
    assert (back.n_neurons, back.steps, back.dt_ms) == (run.n_neurons, run.steps, run.dt_ms)
    for name in ("tick_indptr", "neuron_index", "trace_neurons", "traces_mv"):
        assert np.array_equal(getattr(back, name), getattr(run, name)), name
    assert np.array_equal(back.spike_counts(), run.spike_counts())
    # a recording is not a compiled graph
    with pytest.raises(CompiledError, match="schema"):
        read_compiled(tmp_path / "rec")
    # a changed byte is caught
    traces = tmp_path / "rec" / "traces_mv.bin"
    data = bytearray(traces.read_bytes())
    data[0] ^= 1
    traces.write_bytes(bytes(data))
    with pytest.raises(CompiledError):
        read_recording(tmp_path / "rec")
