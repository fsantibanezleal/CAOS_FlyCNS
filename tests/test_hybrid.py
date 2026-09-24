"""The coupling of the graded optic lobe to the spiking CNS, on a toy CNS whose answers are closed forms.

Six R1-R6 terminals, an L1 and an Mi1 in one column are graded; an LC4, a DNp01 and a centrifugal MeVC1 spike. One
R1-R6 terminal drives LC4 through one synapse (the bridge); MeVC1 synapses onto L1 (feedback).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flycns.compiled import Compiled
from flycns.dynamics import Drive
from flycns.dynamics.hybrid import HybridReference, build_hybrid
from flycns.flyvis import load_ensemble
from flycns.optic_lobe import build_optic_lobe

ENSEMBLE = load_ensemble()
BETA = 100.0


def toy():
    types = ["R1-R6", "L1", "Mi1", "LC4", "DNp01", "MeVC1"]
    superclasses = ["ol_sensory", "ol_intrinsic", "visual_projection", "descending_neuron", "visual_centrifugal"]
    neurons = [("R1-R6", 0, -1, 0)] * 6 + [("L1", 0, -1, 1), ("Mi1", 0, 1, 1), ("LC4", -1, 1, 2),
                                           ("DNp01", -1, 1, 3), ("MeVC1", -1, -1, 4)]
    l1, mi1, lc4, dn, mevc1 = 6, 7, 8, 9, 10
    edges = sorted([(r, l1, 40) for r in range(6)] + [(l1, mi1, 30), (0, lc4, 1), (lc4, dn, 5), (mevc1, l1, 12)])
    pre = np.array([e[0] for e in edges])
    indptr = np.zeros(len(neurons) + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=len(neurons)), out=indptr[1:])
    graph = Compiled(directory=Path("toy"), manifest={"strings": {
        "type": types, "superclass": superclasses, "side": ["unknown", "left", "right", "midline"]}}, arrays={
        "neuron_body_id": np.arange(len(neurons), dtype=np.int64),
        "neuron_type": np.array([types.index(t) for t, *_ in neurons], dtype=np.uint32),
        "neuron_superclass": np.array([s for *_, s in neurons], dtype=np.uint32),
        "neuron_side": np.full(len(neurons), 2, dtype=np.uint8),
        "neuron_column": np.array([c for _, c, _, _ in neurons], dtype=np.int32),
        "neuron_sign": np.array([s for _, _, s, _ in neurons], dtype=np.int8),
        "csr_indptr": indptr, "csr_indices": np.array([e[1] for e in edges], dtype=np.int32),
        "csr_count": np.array([e[2] for e in edges], dtype=np.uint16),
        "column_side": np.full(1, 2, dtype=np.uint8)})
    return graph, {"L1": l1, "Mi1": mi1, "LC4": lc4, "DNp01": dn, "MeVC1": mevc1}


def build():
    graph, ids = toy()
    lobe = build_optic_lobe(graph, ENSEMBLE, 0)
    hybrid = build_hybrid(graph, lobe, bridge_gain_hz=BETA)
    graded_unit = {k: int(np.flatnonzero((lobe.neuron == v) & ~lobe.stand_in)[0])
                   for k, v in {**ids, "R": 0}.items() if k in ("L1", "Mi1", "R")}
    spiking_unit = {k: int(np.flatnonzero(hybrid.spiking_neuron == v)[0])
                    for k, v in ids.items() if k in ("LC4", "DNp01", "MeVC1")}
    return hybrid, graded_unit, spiking_unit


def test_the_bridge_and_the_feedback_have_their_closed_forms():
    hybrid, gu, su = build()
    assert hybrid.report["bridge_connections"] == 1 and hybrid.report["feedback_connections"] == 1
    engine = HybridReference(hybrid)
    p = hybrid.params

    # at grey the spiking side receives nothing: LC4 stays at rest
    grey = engine.run(np.full((100, 1), 0.5), record_graded=np.array([gu["R"]]),
                      trace_spiking=np.array([su["LC4"]]))
    assert np.abs(grey.spikes.traces_mv[:, 0] - p.v0_mv).max() < 1e-3

    # a steady light: once the photoreceptor has settled, LC4's voltage is rest plus g's steady state under the
    # bridge's drive
    lit = engine.run(np.full((300, 1), 1.0), record_graded=np.array([gu["R"]]),
                     trace_spiking=np.array([su["LC4"]]))
    deviation = max(lit.graded[-1, 0], 0.0) - lit.grey_release[gu["R"]]
    assert deviation == pytest.approx(0.5, rel=1e-3)                # the light raises its release by 0.5
    drive_per_s = BETA * deviation * -1 * p.w_syn_mv                # one histaminergic synapse: sign -1
    # g is topped up once per step and decays within it: G = I dt / (1 - b), and the exact step then holds
    # v - v0 = G c / (1 - a), with (a, b, c) the step's closed-form coefficients
    a, b, c = p.decay()
    g_steady = drive_per_s * (p.dt_ms / 1000.0) / (1.0 - b)
    assert lit.spikes.traces_mv[-1, 0] == pytest.approx(p.v0_mv + g_steady * c / (1.0 - a), abs=1e-3 * abs(g_steady))
    assert lit.spikes.spike_counts().sum() == 0

    # feedback: MeVC1 firing at 200 Hz (GABAergic) lowers L1 by 0.01 x 2 x 200 / beta, through L1's own leak
    silent = engine.run(np.full((400, 1), 0.5), record_graded=np.array([gu["L1"]]))
    active = engine.run(np.full((400, 1), 0.5), drive=Drive(activate={su["MeVC1"]: 200.0}), seed=3,
                        record_graded=np.array([gu["L1"]]))
    rate = active.spikes.spike_counts()[su["MeVC1"]] / (active.spikes.steps * p.dt_ms / 1000.0)
    shift = active.graded[200:, 0].mean() - silent.graded[200:, 0].mean()
    assert rate == pytest.approx(200.0, rel=0.1)
    assert shift == pytest.approx(-0.02 * rate / BETA, rel=0.15)


def test_the_torch_hybrid_matches_the_reference():
    pytest.importorskip("torch")
    from flycns.dynamics.hybrid import HybridTorch

    hybrid, gu, su = build()
    frames = np.r_[np.full((50, 1), 0.5), np.full((150, 1), 1.0)]
    record = np.array([gu["L1"], gu["Mi1"]])
    a = HybridReference(hybrid).run(frames, drive=Drive(activate={su["MeVC1"]: 100.0}), seed=1,
                                    record_graded=record, trace_spiking=np.array([su["LC4"]]))
    b = HybridTorch(hybrid, device="cpu").run(frames, drive=Drive(activate={su["MeVC1"]: 100.0}), seed=1,
                                              record_graded=record, trace_spiking=np.array([su["LC4"]]))
    assert np.array_equal(a.spikes.neuron_index, b.spikes.neuron_index)
    assert np.abs(a.graded - b.graded).max() < 1e-4
    assert np.abs(a.spikes.traces_mv - b.spikes.traces_mv).max() < 1e-3
