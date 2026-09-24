"""The spiking engines: activation, counter-based input, silencing, and the GPU engine against the reference.

The small-circuit tests run both engines when PyTorch is installed (the PyTorch one on the CPU; CI has no PyTorch and
runs the reference only, and ``-rs`` lists the skips). The whole-graph comparison needs the compiled MaleCNS
(``FLYCNS_MALECNS_COMPILED``, default ``E:/_Datos/destello/compiled/malecns-v1.0``) and a CUDA device.
"""

from __future__ import annotations

import itertools
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from flycns.dynamics import Drive, LIFParams, LIFReference, synaptic_weights
from flycns.rng import event_threshold, fmix32, hash3, poisson_events

COMPILED = Path(os.environ.get("FLYCNS_MALECNS_COMPILED", "E:/_Datos/destello/compiled/malecns-v1.0"))
ENGINES = ["reference", "torch"]

# (seed, neuron, step) -> hash, pinned; the TypeScript and WGSL implementations must reproduce them
VECTORS = [
    ((0, 0, 0), 1669671676),
    ((1, 2, 3), 2920678231),
    ((0xDEADBEEF, 166699, 1000000), 2110861114),
    ((0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), 3338479333),
    ((42, 12345, 67890), 80117027),
]


def engine(kind, indptr, indices, weights, params=None):
    if kind == "reference":
        return LIFReference(indptr, indices, weights, params)
    pytest.importorskip("torch")
    from flycns.dynamics import LIFTorch

    return LIFTorch(indptr, indices, weights, params, device="cpu")


def chain():
    """0 -> 1 -> 2, 200 synapses per connection, all excitatory."""
    indptr = np.array([0, 1, 2, 2], dtype=np.int64)
    indices = np.array([1, 2], dtype=np.int64)
    counts = np.array([200, 200], dtype=np.uint16)
    signs = np.array([1, 1, 1], dtype=np.int8)
    return indptr, indices, counts, signs


@pytest.mark.parametrize("kind", ENGINES)
def test_activation_forces_spikes_without_refractoriness(kind):
    params = LIFParams()
    steps, rate, seed = 5000, 800.0, 7
    isolated = (np.zeros(2, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
    run = engine(kind, *isolated, params).run(steps, Drive(activate={0: rate}), seed=seed,
                                             trace_neurons=np.array([0]))
    event = hash3(seed, np.zeros(steps, dtype=np.int64), np.arange(steps)) < event_threshold(rate, params.dt_ms / 1e3)
    # each event lifts the neuron from rest by w_syn f_poi, and it spikes on the next step; an event on the step it
    # spikes is lost to the reset
    spiked = np.zeros(steps, dtype=bool)
    for k in range(steps - 1):
        spiked[k + 1] = event[k] and not spiked[k]
    spike_steps, _ = run.spike_times()
    assert np.array_equal(spike_steps, np.flatnonzero(spiked))
    lifted = params.v0_mv + params.w_syn_mv * params.f_poi
    assert lifted == 16.75
    assert np.array_equal(run.traces_mv[:, 0], np.where(event & ~spiked, lifted, params.v0_mv).astype(np.float32))
    # no refractory period: spikes two steps apart, far inside the 22 steps (2.2 ms) other neurons must wait
    assert np.diff(spike_steps).min() == 2


def test_poisson_events_are_counter_based_and_have_the_requested_rate():
    assert int(fmix32(np.uint32(1))) == 0x514E28B7              # MurmurHash3's finaliser, its published value
    assert [int(hash3(*key)) for key, _ in VECTORS] == [value for _, value in VECTORS]
    mmh3 = pytest.importorskip("mmh3")                           # an independent MurmurHash3 implementation
    rng = np.random.default_rng(0)
    for seed, neuron, step in rng.integers(0, 2**32, size=(200, 3), dtype=np.uint64).tolist():
        assert int(hash3(seed, neuron, step)) == mmh3.hash(struct.pack("<II", neuron, step), seed, signed=False)
    # a pure function of (seed, neuron, step): batching and order do not matter, and seeds give distinct streams
    neurons = np.arange(1000)
    whole = hash3(3, neurons, 17)
    assert np.array_equal(whole, np.concatenate([hash3(3, neurons[i:i + 7], 17) for i in range(0, 1000, 7)]))
    assert np.array_equal(whole[::-1], np.array([int(hash3(3, int(i), 17)) for i in neurons[::-1]]))
    assert np.mean(hash3(4, neurons, 17) == whole) < 0.01
    # the long-run rate is the requested one, and consecutive steps are uncorrelated
    rate, dt = 50.0, 1e-4
    grid = np.stack([poisson_events(11, np.arange(10_000), k, event_threshold(rate, dt)) for k in range(1_000)])
    expected = grid.size * rate * dt
    assert abs(grid.sum() - expected) < 5 * np.sqrt(expected)
    lagged = np.corrcoef(grid[:-1].ravel().astype(float), grid[1:].ravel().astype(float))[0, 1]
    assert abs(lagged) < 5e-3
    # neurons are independent of each other within a step, and steps within a neuron: both dispersion indices are 1
    p = grid.mean()
    assert abs(grid.sum(1).var() / (grid.shape[1] * p * (1 - p)) - 1) < 0.2
    assert abs(grid.sum(0).var() / (grid.shape[0] * p * (1 - p)) - 1) < 0.1
    assert event_threshold(0.0, dt) == 0 and event_threshold(1e6, dt) == 2**32 - 1


@pytest.mark.parametrize("kind", ENGINES)
def test_a_silenced_neuron_still_listens_but_reaches_no_one(kind):
    indptr, indices, counts, signs = chain()
    weights = synaptic_weights(indptr, indices, counts, signs, 0.275)
    traced = np.array([1, 2])
    drive = Drive(activate={0: 200.0})
    intact = engine(kind, indptr, indices, weights).run(5000, drive, seed=1, trace_neurons=traced)
    assert intact.spike_counts()[1] > 25 and intact.spike_counts()[2] > 25
    muted = engine(kind, indptr, indices, weights).run(5000, Drive(activate={0: 200.0}, silenced=np.array([1])),
                                                       seed=1, trace_neurons=traced)
    # neuron 1 still hears neuron 0 and fires exactly as before; neuron 2 hears nothing and never leaves rest
    steps_i, neurons_i = intact.spike_times()
    steps_m, neurons_m = muted.spike_times()
    assert np.array_equal(steps_m[neurons_m == 1], steps_i[neurons_i == 1])
    assert muted.spike_counts()[2] == 0
    assert np.all(muted.traces_mv[:, 1] == -52.0)
    # silencing through the weights zeroes the outgoing connection only, and gives the same run
    muted_weights = synaptic_weights(indptr, indices, counts, signs, 0.275, silenced=np.array([1]))
    assert muted_weights.tolist() == [pytest.approx(55.0), 0.0]
    same = engine(kind, indptr, indices, muted_weights).run(5000, drive, seed=1, trace_neurons=traced)
    assert np.array_equal(same.tick_indptr, muted.tick_indptr)
    assert np.array_equal(same.neuron_index, muted.neuron_index)
    assert np.array_equal(same.traces_mv, muted.traces_mv)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def _mean_rate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(0), y.mean(0)
    either = (mx > 0) | (my > 0)
    return float(np.corrcoef(mx[either], my[either])[0, 1])


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.skipif(not (COMPILED / "manifest.json").is_file(), reason=f"no compiled MaleCNS at {COMPILED}")
def test_torch_engine_matches_the_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from flycns.compiled import read_compiled
    from flycns.dynamics import LIFTorch

    c = read_compiled(COMPILED, verify=False)
    weights = synaptic_weights(c["csr_indptr"], c["csr_indices"], c["csr_count"], c["neuron_sign"], 0.275)
    reference = LIFReference(c["csr_indptr"], c["csr_indices"], weights)
    gpu = LIFTorch(c["csr_indptr"], c["csr_indices"], weights, device="cuda")
    types = np.array(c.strings["type"], dtype=object)[c["neuron_type"]]
    classes = np.array(c.strings["class"], dtype=object)[c["neuron_class"]]

    # moderate drive, 200 ms: the float32 engine gives the reference's activity (the same neurons, the same counts)
    lb1 = np.flatnonzero(np.isin(types, ["LB1a", "LB1b", "LB1c", "LB1d", "LB1e"]))
    assert len(lb1) == 57
    moderate = Drive(activate={int(i): 150.0 for i in lb1})
    a, b = reference.run(2000, moderate, seed=0), gpu.run(2000, moderate, seed=0)
    assert a.spike_counts().sum() > 5000
    assert _jaccard(a.spike_counts() > 0, b.spike_counts() > 0) >= 0.99
    assert _mean_rate_correlation(a.rates_hz()[None], b.rates_hz()[None]) >= 0.99

    # strong drive, 500 ms: the network has two states and single runs diverge, so the engines are compared as two
    # samples of one process. Five GPU trials against five independent reference trials must correlate no worse than
    # the 5th percentile of the reference against itself, over the 126 ways of splitting ten reference seeds in two
    gustatory = np.flatnonzero(classes == "gustatory")
    strong = Drive(activate={int(i): 150.0 for i in gustatory})
    ref_rates = np.stack([reference.run(5000, strong, seed=s).rates_hz() for s in range(10)])
    gpu_rates = np.stack([gpu.run(5000, strong, seed=s).rates_hz() for s in range(5)])
    splits = []
    for half in itertools.combinations(range(10), 5):
        if 0 in half:
            other = [s for s in range(10) if s not in half]
            splits.append(_mean_rate_correlation(ref_rates[list(half)], ref_rates[other]))
    assert len(splits) == 126
    assert _mean_rate_correlation(ref_rates[5:], gpu_rates) >= np.percentile(splits, 5)
