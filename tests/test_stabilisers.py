"""E4's stabilisers and the modulated (light-driven) Poisson input, on small circuits with exact or bounded answers."""

from __future__ import annotations

import numpy as np
import pytest

from flycns.dynamics import Drive, LIFParams, LIFReference, synaptic_weights
from flycns.dynamics.lif import stabilised_weights

ENGINES = ["reference", "torch"]


def engine(kind, indptr, indices, weights, params=None):
    if kind == "reference":
        return LIFReference(indptr, indices, weights, params)
    pytest.importorskip("torch")
    from flycns.dynamics import LIFTorch

    return LIFTorch(indptr, indices, weights, params, device="cpu")


def chain(n_syn=200):
    indptr = np.array([0, 1, 2, 2], dtype=np.int64)
    indices = np.array([1, 2], dtype=np.int64)
    return indptr, indices, synaptic_weights(indptr, indices, np.array([n_syn, n_syn]), np.array([1, 1, 1]), 0.275)


@pytest.mark.parametrize("kind", ENGINES)
def test_stabilisers_do_what_they_state(kind):
    indptr, indices, weights = chain()
    drive = Drive(activate={0: 300.0})
    plain = engine(kind, indptr, indices, weights).run(5000, drive, seed=2)
    # adaptation off is the published model exactly
    zero = engine(kind, indptr, indices, weights, LIFParams(adaptation_mv=0.0)).run(5000, drive, seed=2)
    assert np.array_equal(plain.neuron_index, zero.neuron_index)
    # adaptation on: the driven neurons fire less, the activated one (driven past any adaptation) does not change
    adapted = engine(kind, indptr, indices, weights, LIFParams(adaptation_mv=1.5)).run(5000, drive, seed=2)
    assert adapted.spike_counts()[0] == plain.spike_counts()[0]
    assert adapted.spike_counts()[1] < plain.spike_counts()[1]

    # modulated Poisson input at a constant rate is the published activation, event for event
    mod = Drive(modulated=(np.array([0]), np.full((10, 1), 300.0), 500))
    assert np.array_equal(engine(kind, indptr, indices, weights).run(5000, mod, seed=2).neuron_index,
                          plain.neuron_index)
    # and a rate that is zero in a frame gives no events in it
    rates = np.full((10, 1), 300.0)
    rates[3:6] = 0.0
    off = engine(kind, indptr, indices, weights).run(5000, Drive(modulated=(np.array([0]), rates, 500)), seed=2)
    steps, neurons = off.spike_times()
    first = steps[neurons == 0]
    assert not np.any((first >= 3 * 500 + 1) & (first < 6 * 500 + 1))
    assert np.any(first < 3 * 500) and np.any(first > 6 * 500)


def test_stabilised_weights_cap_damp_and_normalise():
    # 0 -> 2 (80 synapses, capped at 60), 1 -> 2 (30), 2 -> 3 (10, same type as 3), 4 -> 3 (9000: normalised)
    indptr = np.array([0, 1, 2, 3, 3, 4], dtype=np.int64)
    indices = np.array([2, 2, 3, 3], dtype=np.int64)
    counts = np.array([80, 30, 10, 9000])
    signs = np.array([1, -1, 1, 1, 1])
    types = np.array([0, 1, 2, 2, 3])
    w = stabilised_weights(indptr, indices, counts, signs, types, 1.0, cap=60, same_type_factor=0.1,
                           fan_in_limit=5000)
    assert w[0] == pytest.approx(60.0) and w[1] == pytest.approx(-30.0)
    # neuron 3 receives 10 x 0.1 + min(9000, 60) = 61, under the limit: nothing to normalise
    assert w[2] == pytest.approx(1.0) and w[3] == pytest.approx(60.0)
    big = stabilised_weights(indptr, indices, counts, signs, types, 1.0, cap=1e9, same_type_factor=0.1,
                             fan_in_limit=5000)
    total = 10 * 0.1 + 9000
    assert big[3] == pytest.approx(9000 * 5000 / total) and big[2] == pytest.approx(1.0 * 5000 / total)
