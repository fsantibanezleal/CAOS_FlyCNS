"""One step of the graded engines against the published formula, written out neuron by neuron."""

from __future__ import annotations

import numpy as np
import pytest

from flycns.dynamics.graded import GradedNetwork, GradedReference

DT = 1 / 200


def small_network(seed: int = 0) -> GradedNetwork:
    rng = np.random.default_rng(seed)
    n, e, columns = 12, 40, 3
    source = rng.integers(n, size=e)
    target = rng.integers(n, size=e)
    return GradedNetwork.from_input_index(
        bias=rng.normal(0.5, 0.5, size=n),
        # some time constants below the step, where tau_eff = dt
        time_const_s=np.where(rng.random(n) < 0.3, 0.002, rng.uniform(0.01, 0.2, size=n)),
        source=source,
        target=target,
        weight=rng.normal(0, 0.5, size=e),
        input_index=np.arange(2 * columns).reshape(2, columns),     # two input types, three columns
    )


def formula(net: GradedNetwork, v: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """flyvis's PPNeuronIGRSynapses step, one neuron at a time."""
    out = np.empty_like(v)
    for i in range(net.n):
        synaptic = sum(net.weight[k] * max(v[net.source[k]], 0.0) for k in range(len(net.weight))
                       if net.target[k] == i)
        x = sum(intensity[c] for j, c in zip(net.input_neuron, net.input_column, strict=True) if j == i)
        tau_eff = max(net.time_const_s[i], DT)
        out[i] = v[i] + DT / tau_eff * (-v[i] + net.bias[i] + synaptic + x)
    return out


@pytest.mark.parametrize("kind", ["reference", "torch"])
def test_one_step_is_the_published_formula(kind):
    net = small_network()
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, size=net.n)                      # negative values exercise the rectification
    intensity = np.array([0.2, 0.9, 0.5])
    expected = formula(net, v, intensity)
    if kind == "reference":
        engine = GradedReference(net, DT)
        got = engine.step(v, net.column_current(intensity))
        assert np.allclose(got, expected, rtol=0, atol=1e-12)
    else:
        torch = pytest.importorskip("torch")
        from flycns.dynamics.graded import GradedTorch

        engine = GradedTorch(net, DT, device="cpu")
        got = engine.step(torch.as_tensor(v, dtype=torch.float32), engine.column_current(intensity)).numpy()
        assert np.allclose(got, expected, rtol=0, atol=1e-5)
    # input reaches the input neurons only
    current = net.column_current(intensity)
    assert np.all(current[6:] == 0) and np.allclose(current[:6], np.tile(intensity, 2))


def test_run_steps_from_the_resting_potentials_and_records_after_each_step():
    net = small_network(2)
    engine = GradedReference(net, DT)
    frames = np.random.default_rng(3).uniform(0, 1, size=(5, 3))
    final, activity = engine.run(frames)
    v = net.bias.astype(np.float64).copy()
    for k, frame in enumerate(frames):
        v = formula(net, v, frame)
        assert np.allclose(activity[k], v.astype(np.float32), atol=1e-6)
    assert np.allclose(final, v, atol=1e-12)
    # a steady state under constant input is a fixed point of the step
    steady = engine.steady_state(5.0, grey=0.5)
    assert np.allclose(engine.step(steady, net.column_current(np.full(3, 0.5))), steady, atol=1e-9)
