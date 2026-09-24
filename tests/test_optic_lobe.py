"""The transfer rules of ``flycns.optic_lobe`` on a toy optic lobe, built in memory, with the shipped flyvis numbers.

Three right-eye columns. Column 0 holds six R1-R6 terminals, column 1 three, column 2 none; each column has an L1,
an L2 and an Mi1; two T4a cells have no column of their own; one CT1 and one Dm3 (a type flyvis does not model)
span the columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flycns.compiled import Compiled
from flycns.flyvis import load_ensemble
from flycns.optic_lobe import TransferRules, build_optic_lobe, home_columns

ENSEMBLE = load_ensemble()
N_TOT = np.bincount(ENSEMBLE.group_pair, weights=ENSEMBLE.group_n_syn, minlength=len(ENSEMBLE.pair_sign))
PAIRS = ENSEMBLE.pair_index()


def toy():
    types = ["R1-R6", "L1", "L2", "Mi1", "T4a", "CT1", "Dm3"]
    neurons = []                    # (type, column, sign, superclass)
    for c, n in ((0, 6), (1, 3)):
        neurons += [("R1-R6", c, -1, "ol_sensory")] * n
    for c in range(3):
        neurons += [("L1", c, -1, "ol_intrinsic"), ("L2", c, 1, "ol_intrinsic"), ("Mi1", c, 1, "ol_intrinsic")]
    neurons += [("T4a", -1, 1, "ol_intrinsic"), ("T4a", -1, 1, "ol_intrinsic"), ("CT1", -1, -1, "ol_intrinsic"),
                ("Dm3", -1, -1, "ol_intrinsic")]
    index = {}
    for i, (t, c, _, _) in enumerate(neurons):
        index.setdefault((t, c), []).append(i)
    t4a = index[("T4a", -1)]
    ct1, dm3 = index[("CT1", -1)][0], index[("Dm3", -1)][0]
    edges = []
    for c in (0, 1):
        for r in index[("R1-R6", c)]:
            edges += [(r, index[("L1", c)][0], 40), (r, index[("L2", c)][0], 40)]
    edges += [(index[("Mi1", 0)][0], t4a[0], 30), (index[("Mi1", 1)][0], t4a[0], 20),
              (index[("Mi1", 2)][0], t4a[1], 500)]                      # far more than flyvis's 68: capped
    edges += [(index[("Mi1", c)][0], ct1, 20) for c in range(3)] + [(ct1, t4a[0], 12)]
    edges += [(dm3, index[("L1", c)][0], 10 * (c + 1)) for c in range(3)]
    edges.sort()
    pre = np.array([e[0] for e in edges])
    indptr = np.zeros(len(neurons) + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=len(neurons)), out=indptr[1:])
    superclasses = ["ol_sensory", "ol_intrinsic"]
    graph = Compiled(directory=Path("toy"), manifest={"strings": {
        "type": types, "superclass": superclasses, "side": ["unknown", "left", "right", "midline"]}}, arrays={
        "neuron_body_id": np.arange(len(neurons), dtype=np.int64),
        "neuron_type": np.array([types.index(t) for t, *_ in neurons], dtype=np.uint32),
        "neuron_superclass": np.array([superclasses.index(s) for *_, s in neurons], dtype=np.uint32),
        "neuron_side": np.full(len(neurons), 2, dtype=np.uint8),
        "neuron_column": np.array([c for _, c, _, _ in neurons], dtype=np.int32),
        "neuron_sign": np.array([s for _, _, s, _ in neurons], dtype=np.int8),
        "csr_indptr": indptr, "csr_indices": np.array([e[1] for e in edges], dtype=np.int32),
        "csr_count": np.array([e[2] for e in edges], dtype=np.uint16),
        "column_side": np.full(3, 2, dtype=np.uint8)})
    return graph, index


def strength(pre: str, post: str, model: int = 0) -> tuple[float, float]:
    """flyvis's signed strength per synapse and synapses per target, summed over a class's flyvis types."""
    pres = [f"R{k}" for k in range(1, 7)] if pre == "R1-R6" else [pre]
    ids = [PAIRS[(p, post)] for p in pres]
    signed = sum(ENSEMBLE.pair_sign[i] * N_TOT[i] * ENSEMBLE.strength[model][i] for i in ids)
    count = sum(N_TOT[i] for i in ids)
    return signed / count, count


def drive(lobe, pre_units, post_unit) -> float:
    net = lobe.network
    m = np.isin(net.source, pre_units) & (net.target == post_unit)
    return float(net.weight[m].sum())


def test_transfer_rules_on_a_toy_optic_lobe():
    graph, index = toy()
    lobe = build_optic_lobe(graph, ENSEMBLE, model=0)
    unit_of = {int(n): u for u, n in enumerate(lobe.neuron) if n >= 0 and not lobe.stand_in[u]}
    L1 = [unit_of[index[("L1", c)][0]] for c in range(3)]

    # rule 4: R1-R6 onto L1 of the complete column carries flyvis's strength (240 synapses, flyvis's column 241)
    per_synapse, flyvis_count = strength("R1-R6", "L1")
    assert flyvis_count == pytest.approx(241.0)
    rs0 = [unit_of[i] for i in index[("R1-R6", 0)]]
    assert drive(lobe, rs0, L1[0]) == pytest.approx(240 * per_synapse, rel=1e-6)
    # ... and a neuron driven harder than flyvis's column is capped at flyvis's drive
    mi1_t4, n_mi1 = strength("Mi1", "T4a")
    t4a = [unit_of[i] for i in index[("T4a", -1)]]
    assert drive(lobe, [unit_of[index[("Mi1", 2)][0]]], t4a[1]) == pytest.approx(n_mi1 * mi1_t4, rel=1e-6)
    assert drive(lobe, [unit_of[index[("Mi1", c)][0]] for c in (0, 1)], t4a[0]) == pytest.approx(50 * mi1_t4)

    # rule 5: each L1 receives flyvis's initial drive, 0.01 x 2, from Dm3 (inhibitory), whatever the count
    dm3 = unit_of[index[("Dm3", -1)][0]]
    for u in L1:
        assert drive(lobe, [dm3], u) == pytest.approx(-0.02)
    assert not lobe.mapped[dm3] and lobe.network.bias[dm3] == pytest.approx(0.5)

    # rule 6: stand-ins make up what columns 1 and 2 lack; they see their column's light and are flagged
    stand = np.flatnonzero(lobe.stand_in)
    assert sorted(lobe.column[stand].tolist()) == [1, 2]
    s1 = stand[lobe.column[stand] == 1][0]
    s2 = stand[lobe.column[stand] == 2][0]
    rs1 = [unit_of[i] for i in index[("R1-R6", 1)]]
    assert drive(lobe, rs1 + [s1], L1[1]) == pytest.approx(240 * per_synapse, rel=1e-6)
    assert drive(lobe, [s2], L1[2]) == pytest.approx(240 * per_synapse, rel=1e-6)
    lit = dict(zip(lobe.network.input_neuron.tolist(), lobe.network.input_column.tolist(), strict=True))
    assert lit[s1] == 1 and lit[s2] == 2 and all(lit[u] == 0 for u in rs0)
    assert len(lit) == 9 + 2

    # CT1: its connections move to per-column compartments of the kind flyvis pairs the partner with
    names = np.array(lobe.classes, dtype=object)[lobe.unit_class]
    m10 = np.flatnonzero(names == "CT1(M10)")
    assert sorted(lobe.column[m10].tolist()) == [0, 1, 2]
    assert lobe.report["ct1_compartments"]["connections_left_on_whole_ct1"] == 0
    into = lobe.network.target[np.isin(lobe.network.source, [unit_of[index[("Mi1", c)][0]] for c in range(3)])]
    assert set(into.tolist()) & set(m10.tolist()) == set(m10.tolist())
    out_of = lobe.network.source[lobe.network.target == t4a[0]]
    assert any(names[u] == "CT1(M10)" and lobe.column[u] == 0 for u in out_of)   # T4a[0]'s home column is 0

    # every connection is either transferred or defaulted, and none carries a NaN
    assert np.isfinite(lobe.network.weight).all()
    assert lobe.transferred.sum() > 0 and (~lobe.transferred).sum() > 0


def test_home_columns_follow_the_strongest_input():
    column = np.array([0, 1, -1, -1])
    src, tar = np.array([0, 1, 2]), np.array([2, 2, 3])
    count = np.array([5.0, 9.0, 1.0])
    assert home_columns(column, src, tar, count, n_columns=2).tolist() == [0, 1, 1, 1]


def test_rules_can_be_narrowed():
    graph, _ = toy()
    only_flyvis = build_optic_lobe(graph, ENSEMBLE, 0, TransferRules(include_unmapped=False, stand_ins=False,
                                                                     ct1_compartments=False))
    names = set(np.array(only_flyvis.classes, dtype=object)[only_flyvis.unit_class].tolist())
    assert "Dm3" not in names and not only_flyvis.stand_in.any()
