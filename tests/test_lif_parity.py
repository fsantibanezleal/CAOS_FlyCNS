"""The reference engine against a literal Brian2 transcription of the published model (``model.py``).

The transcription keeps the published equations, method, threshold, reset, refractoriness and synapse exactly. Only the
drive differs from the paper's PoissonInput: fixed input events (a SpikeGeneratorGroup onto ``v``) replace the random
ones, because Brian2's random stream cannot be reproduced outside Brian2; the events land in the same slot of the
schedule as PoissonInput's do.
"""

from __future__ import annotations

import numpy as np
import pytest

from flycns.dynamics import Drive, LIFParams, LIFReference
from flycns.parity import circuit

brian2 = pytest.importorskip("brian2")


def brian2_run(n, pre, post, weights_mv, events, steps):
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
        prefs,
        start_scope,
    )

    prefs.codegen.target = "numpy"
    start_scope()
    defaultclock.dt = 0.1 * ms
    namespace = {"v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV, "t_mbr": 20 * ms, "tau": 5 * ms}
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neu = NeuronGroup(n, model=eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                      refractory="rfc", namespace=namespace)
    neu.v = -52 * mV
    neu.g = 0 * mV
    neu.rfc = 2.2 * ms
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms)
    syn.connect(i=pre, j=post)
    syn.w = weights_mv * mV
    gen_index, gen_time, targets, amounts = [], [], [], []
    k = 0
    for step, (neurons, dvs) in sorted(events.items()):
        for neuron, dv in zip(neurons, dvs, strict=True):
            gen_index.append(k)
            gen_time.append(step * 0.1)
            targets.append(neuron)
            amounts.append(dv)
            k += 1
    gen = SpikeGeneratorGroup(k, np.array(gen_index), np.array(gen_time) * ms)
    inp = Synapses(gen, neu, "wp : volt", on_pre="v += wp")
    inp.connect(i=np.arange(k), j=np.array(targets))
    inp.wp = np.array(amounts) * mV
    monitor = SpikeMonitor(neu)
    Network(neu, syn, gen, inp, monitor).run(steps * 0.1 * ms)
    step_of = np.rint(np.asarray(monitor.t / ms) / 0.1).astype(np.int64)
    return step_of, np.asarray(monitor.i, dtype=np.int64)


@pytest.mark.parity
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_reference_matches_a_literal_brian2_transcription(seed):
    n, pre, post, weights_mv, indptr, events = circuit(seed)
    steps = 2000
    order = np.lexsort((post, pre))
    engine = LIFReference(indptr, post[order], weights_mv[order], LIFParams())
    drive = Drive(events={k: (np.array(v[0]), np.array(v[1])) for k, v in events.items()})
    run = engine.run(steps, drive)
    ours_step, ours_neuron = run.spike_times()
    theirs_step, theirs_neuron = brian2_run(n, pre, post, weights_mv, events, steps)
    theirs = sorted(zip(theirs_step.tolist(), theirs_neuron.tolist(), strict=True))
    ours = sorted(zip(ours_step.tolist(), ours_neuron.tolist(), strict=True))
    assert len(ours) > 50                       # the circuit is active, so the comparison means something
    assert ours == theirs
