"""The parity scenarios: fixed-seed runs both implementations perform, written under ``parity/`` by
``scripts/make_parity_fixtures.py`` and reproduced by the TypeScript tests.

Every builder is a pure function of its seed, so the committed fixtures can be regenerated and compared
(``tests/test_parity_fixtures.py``). The circuits are the ones the Brian2 transcription checks the reference on.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

from .bundle import (
    Scenario,
    graded_arrays,
    hybrid_arrays,
    hybrid_constants,
    lattice_arrays,
    lif_arrays,
    lif_constants,
)
from .dynamics.graded import GradedNetwork
from .dynamics.lif import Drive, LIFParams


def circuit(seed: int, n: int = 40, edges: int = 220):
    """A random signed circuit with a fixed drive: inputs 0-5 get strong events (one crosses threshold) and weaker
    subthreshold ones. Returns (n, pre, post, weights_mv, indptr, events)."""
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < edges:
        a, b = rng.integers(n, size=2)
        if a != b:
            pairs.add((int(a), int(b)))
    pairs = sorted(pairs)
    pre = np.array([p for p, _ in pairs])
    post = np.array([q for _, q in pairs])
    counts = rng.integers(1, 40, size=len(pairs))
    signs = np.where(rng.random(n) < 0.7, 1, -1)
    weights_mv = signs[pre] * counts * 0.275
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    events: dict[int, tuple[list, list]] = {}
    for neuron in range(6):
        for step in np.sort(rng.choice(2000, size=30, replace=False)):
            dv = 68.75 if rng.random() < 0.6 else float(rng.uniform(1.0, 6.0))
            events.setdefault(int(step), ([], []))
            events[int(step)][0].append(neuron)
            events[int(step)][1].append(dv)
    return n, pre, post, weights_mv, indptr, events


def _lif_scenario(seed: int, drive: Drive, steps: int, params: LIFParams | None = None, n: int = 40,
                  edges: int = 220, trace=()) -> Scenario:
    params = params or LIFParams()
    n, _, post, weights, indptr, _ = circuit(seed, n, edges)
    return Scenario(arrays=lif_arrays(indptr, post, weights), constants={"engine": "lif", "lif": lif_constants(params),
                                                                         "n_neurons": n},
                    steps=steps, seed=seed, drive=drive, trace_neurons=np.array(trace, dtype=np.int64))


def lif_circuit(seed: int) -> Scenario:
    """The Brian2-checked circuit: fixed events only, three neurons traced."""
    _, _, _, _, _, events = circuit(seed)
    return _lif_scenario(seed, Drive(events=events), 2000, trace=(0, 7, 19))


def lif_poisson() -> Scenario:
    """Counter-based activation at three rates, modulated rates over frames, two silenced neurons, a few events."""
    rng = np.random.default_rng(11)
    rates = rng.uniform(0.0, 300.0, size=(20, 3))
    modulated = (np.array([3, 4, 5]), rates, 100)
    events = {250: ([10, 11, 10], [30.0, 68.75, 40.0]), 1500: ([12], [68.75])}
    drive = Drive(activate={0: 150.0, 1: 150.0, 2: 400.0}, events=events, silenced=np.array([7, 8]),
                  modulated=modulated)
    return _lif_scenario(3, drive, 2000, n=60, edges=400, trace=(0, 3, 7, 20, 41))


def lif_adaptation() -> Scenario:
    """E4's spike-frequency adaptation on, four neurons activated at 200 Hz."""
    params = LIFParams(adaptation_mv=1.5, tau_adaptation_ms=200.0)
    drive = Drive(activate={0: 200.0, 1: 200.0, 2: 200.0, 3: 200.0})
    return _lif_scenario(4, drive, 3000, params=params, n=50, edges=300, trace=(0, 9, 30))


def graded_random() -> Scenario:
    """A random graded network (some time constants below the step) over random frames of five columns."""
    rng = np.random.default_rng(21)
    n, e, columns = 50, 300, 5
    net = GradedNetwork.from_input_index(
        bias=rng.normal(0.5, 0.5, size=n),
        time_const_s=np.where(rng.random(n) < 0.3, 0.002, rng.uniform(0.01, 0.2, size=n)),
        source=rng.integers(n, size=e), target=rng.integers(n, size=e), weight=rng.normal(0, 0.3, size=e),
        input_index=np.arange(2 * columns).reshape(2, columns))
    frames = rng.uniform(0.0, 1.0, size=(60, columns))
    return Scenario(arrays=graded_arrays(net), constants={"engine": "graded", "graded": {"dt_s": 1 / 200,
                                                                                         "n_columns": columns},
                                                          "n_neurons": n}, steps=60, intensity=frames)


def toy_cns():
    """The toy CNS of the hybrid tests: R1-R6, L1 and Mi1 graded in one column; LC4, DNp01 and MeVC1 spiking."""
    from pathlib import Path

    from .compiled import Compiled

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


def _toy_hybrid():
    from .dynamics.hybrid import build_hybrid
    from .flyvis import load_ensemble
    from .optic_lobe import build_optic_lobe

    graph, ids = toy_cns()
    lobe = build_optic_lobe(graph, load_ensemble(), 0)
    hybrid = build_hybrid(graph, lobe, bridge_gain_hz=100.0)
    graded_unit = {k: int(np.flatnonzero((lobe.neuron == v) & ~lobe.stand_in)[0])
                   for k, v in {**ids, "R": 0}.items() if k in ("L1", "Mi1", "R")}
    spiking_unit = {k: int(np.flatnonzero(hybrid.spiking_neuron == v)[0])
                    for k, v in ids.items() if k in ("LC4", "DNp01", "MeVC1")}
    return hybrid, graded_unit, spiking_unit


def hybrid_toy() -> Scenario:
    """The toy CNS with the lobe's own dynamics: grey, then a step of light, MeVC1 activated at 100 Hz."""
    hybrid, gu, su = _toy_hybrid()
    frames = np.r_[np.full((20, 1), 0.5), np.full((60, 1), 1.0), np.full((20, 1), 0.25)]
    constants = {"engine": "hybrid", "lif": lif_constants(hybrid.params),
                 "graded": {"dt_s": 1 / 200, "n_columns": int(hybrid.graded.n_columns)},
                 "hybrid": hybrid_constants(hybrid, 1 / 200), "source": "own",
                 "n_spiking": int(hybrid.n_spiking), "n_graded": int(hybrid.graded.n)}
    return Scenario(arrays=hybrid_arrays(hybrid), constants=constants, steps=100, seed=5,
                    drive=Drive(activate={su["MeVC1"]: 100.0}), intensity=frames,
                    trace_neurons=np.array(sorted(su.values())), record_graded=np.arange(hybrid.graded.n),
                    t_pre_s=0.2)


def hybrid_lattice_toy() -> Scenario:
    """The toy CNS with a lattice source: two small synthetic lattice networks (three columns each, one input neuron
    per column) whose nodes stand for the toy's graded units; the code path of E3 without flyvis's 721 columns."""
    hybrid, gu, su = _toy_hybrid()
    rng = np.random.default_rng(31)
    networks = []
    for _ in range(2):
        n, e, columns = 12, 24, 3
        networks.append(GradedNetwork.from_input_index(
            bias=rng.normal(0.4, 0.3, size=n), time_const_s=rng.uniform(0.005, 0.05, size=n),
            source=rng.integers(n, size=e), target=rng.integers(n, size=e), weight=rng.normal(0, 0.4, size=e),
            input_index=np.arange(columns).reshape(1, columns)))
    mapping = SimpleNamespace(unit=np.array([gu["R"], gu["L1"], gu["Mi1"]]), side=np.array([0, 1, 0]),
                              nodes=[np.array([3, 4]), np.array([5]), np.array([6, 7, 8])])
    frames = np.stack([np.r_[np.full((30, 3), 0.5), rng.uniform(0.0, 1.0, size=(70, 3))],
                       np.r_[np.full((30, 3), 0.5), rng.uniform(0.0, 1.0, size=(70, 3))]], axis=1)
    arrays = {**hybrid_arrays(hybrid), **lattice_arrays(networks, mapping)}
    constants = {"engine": "hybrid", "lif": lif_constants(hybrid.params),
                 "graded": {"dt_s": 1 / 200, "n_columns": int(hybrid.graded.n_columns)},
                 "hybrid": hybrid_constants(hybrid, 1 / 200), "source": "lattice",
                 "lattice": {"n_columns": 3, "count": 2},
                 "n_spiking": int(hybrid.n_spiking), "n_graded": int(hybrid.graded.n)}
    return Scenario(arrays=arrays, constants=constants, steps=100, seed=6, drive=Drive(activate={su["MeVC1"]: 80.0}),
                    intensity=frames, trace_neurons=np.array(sorted(su.values())),
                    record_graded=np.arange(hybrid.graded.n), t_pre_s=0.2)


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "lif-circuit-0": lambda: lif_circuit(0),
    "lif-circuit-1": lambda: lif_circuit(1),
    "lif-circuit-2": lambda: lif_circuit(2),
    "lif-poisson": lif_poisson,
    "lif-adaptation": lif_adaptation,
    "graded-random": graded_random,
    "hybrid-toy": hybrid_toy,
    "hybrid-lattice-toy": hybrid_lattice_toy,
}
