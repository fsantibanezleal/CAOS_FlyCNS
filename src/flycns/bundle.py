"""Engine bundles and scenarios: what the browser engine loads, written by Python from the networks it derived.

A **bundle** (schema ``flycns.engine/1``) is a compiled-style directory holding the arrays an engine steps over (the
LIF's CSR and weights, the graded network, the hybrid's bridge and feedback, E3's lattices and map) and, in the
manifest's ``release`` field, every constant the step uses, as JSON numbers: a float written by ``json`` and read by
``JSON.parse`` round-trips exactly, so the TypeScript engine never recomputes an exponential the reference computed.

A **scenario** (schema ``flycns.scenario/1``) is a bundle plus one run: the drive, the stimulus, the seed, the
lead-in and what to record. ``run_reference`` runs it through the NumPy engines; ``write_expected`` stores the
result (schema ``flycns.expected/1``). The committed fixtures under ``parity/`` are scenarios with their expected
output, and the TypeScript tests must reproduce them (``docs/design/features/browser/design.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .compiled import Compiled, read_compiled, write_compiled
from .dynamics.graded import GradedNetwork, GradedReference
from .dynamics.hybrid import HybridNetwork, HybridReference, LatticeSource, _Hybrid, _NumpyOps
from .dynamics.lif import Drive, LIFParams, LIFReference, Run

SCHEMA_ENGINE = "flycns.engine/1"
SCHEMA_SCENARIO = "flycns.scenario/1"
SCHEMA_EXPECTED = "flycns.expected/1"

LIF_FIELDS = ("v0_mv", "v_rst_mv", "v_th_mv", "t_mbr_ms", "tau_ms", "t_rfc_ms", "t_dly_ms", "w_syn_mv", "f_poi",
              "dt_ms", "adaptation_mv", "tau_adaptation_ms")


# ------------------------------------------------------------------------------------------------- constants

def lif_constants(params: LIFParams) -> dict[str, Any]:
    """The published constants and everything the step derives from them, computed once here."""
    a, b, c = params.decay()
    return {**{k: getattr(params, k) for k in LIF_FIELDS},
            "delay_steps": params.delay_steps, "refractory_steps": params.refractory_steps,
            "decay_a": a, "decay_b": b, "decay_c": c,
            "adaptation_decay": math.exp(-params.dt_ms / params.tau_adaptation_ms),
            "w_event_mv": params.w_syn_mv * params.f_poi}


def lif_params(constants: dict[str, Any]) -> LIFParams:
    return LIFParams(**{k: constants[k] for k in LIF_FIELDS})


def hybrid_constants(hybrid: HybridNetwork, dt_graded_s: float) -> dict[str, Any]:
    engine = _Hybrid(hybrid, dt_graded_s)                      # the loop's constants, as the loop computes them
    return {"bridge_gain_hz": hybrid.bridge_gain_hz, "dt_graded_s": engine.dt_graded,
            "steps_per_graded": engine.steps_per_graded, "dt_lif_s": engine.dt_lif_s,
            "rate_decay": engine.decay, "rate_amount": 1.0 / engine.tau_filter_s}


# ---------------------------------------------------------------------------------------------------- arrays

def _i32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int32)


def _i64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)


def _f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def lif_arrays(indptr, indices, weights_mv, exact: bool = True) -> dict[str, np.ndarray]:
    """The spiking network: an exact bundle keeps float64 weights (the reference's); a large one stores float32."""
    weights = _f64(weights_mv) if exact else np.asarray(weights_mv, dtype=np.float32)
    return {"lif_indptr": _i64(indptr), "lif_indices": _i32(indices), "lif_weight_mv": weights}


def graded_arrays(network: GradedNetwork, prefix: str = "graded") -> dict[str, np.ndarray]:
    return {f"{prefix}_bias": _f64(network.bias), f"{prefix}_time_const_s": _f64(network.time_const_s),
            f"{prefix}_source": _i32(network.source), f"{prefix}_target": _i32(network.target),
            f"{prefix}_weight": _f64(network.weight), f"{prefix}_input_neuron": _i32(network.input_neuron),
            f"{prefix}_input_column": _i32(network.input_column)}


def graded_network(arrays: dict[str, np.ndarray], n_columns: int, prefix: str = "graded") -> GradedNetwork:
    a = arrays
    return GradedNetwork(bias=a[f"{prefix}_bias"], time_const_s=a[f"{prefix}_time_const_s"],
                         source=a[f"{prefix}_source"], target=a[f"{prefix}_target"], weight=a[f"{prefix}_weight"],
                         input_neuron=a[f"{prefix}_input_neuron"], input_column=a[f"{prefix}_input_column"],
                         n_columns=int(n_columns))


def hybrid_arrays(hybrid: HybridNetwork, exact: bool = True) -> dict[str, np.ndarray]:
    out = lif_arrays(hybrid.lif_indptr, hybrid.lif_indices, hybrid.lif_weights_mv, exact)
    out.update(graded_arrays(hybrid.graded))
    out.update({"bridge_source": _i32(hybrid.bridge_source), "bridge_target": _i32(hybrid.bridge_target),
                "bridge_weight": _f64(hybrid.bridge_weight), "feedback_source": _i32(hybrid.feedback_source),
                "feedback_target": _i32(hybrid.feedback_target), "feedback_weight": _f64(hybrid.feedback_weight),
                "spiking_neuron": _i64(hybrid.spiking_neuron)})
    return out


def lattice_arrays(lattice_networks, mapping) -> dict[str, np.ndarray]:
    """E3's source: flyvis's network per eye and the map from lattice nodes to optic-lobe units. The map is stored
    per mapped unit (``map_unit``, ``map_side``, ``map_indptr``) over its nodes (``map_node``, ``map_weight``)."""
    out = {}
    for k, net in enumerate(lattice_networks):
        out.update(graded_arrays(net, f"lattice{k}"))
    lengths = np.array([len(n) for n in mapping.nodes], dtype=np.int64)
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    nodes = np.concatenate(mapping.nodes) if len(mapping.nodes) else np.zeros(0, dtype=np.int64)
    weight = np.concatenate([np.full(len(n), 1.0 / len(n)) for n in mapping.nodes]) if len(mapping.nodes) else []
    out.update({"map_unit": _i32(mapping.unit), "map_side": np.asarray(mapping.side, dtype=np.uint8),
                "map_indptr": indptr, "map_node": _i32(nodes), "map_weight": np.asarray(weight, dtype=np.float32)})
    return out


def lattice_map(arrays: dict[str, np.ndarray]) -> SimpleNamespace:
    indptr = arrays["map_indptr"]
    nodes = [arrays["map_node"][indptr[i]:indptr[i + 1]].astype(np.int64) for i in range(len(indptr) - 1)]
    return SimpleNamespace(unit=arrays["map_unit"].astype(np.int64), side=arrays["map_side"].astype(np.int64),
                           nodes=nodes)


# --------------------------------------------------------------------------------------------------- bundles

def write_lif_bundle(directory: Path, indptr, indices, weights_mv, params: LIFParams | None = None,
                     exact: bool = True, meta: dict | None = None) -> dict:
    params = params or LIFParams()
    release = {"engine": "lif", "lif": lif_constants(params), "n_neurons": len(indptr) - 1, **(meta or {})}
    return write_compiled(directory, lif_arrays(indptr, indices, weights_mv, exact), {"release": release},
                          schema=SCHEMA_ENGINE)


def write_graded_bundle(directory: Path, network: GradedNetwork, dt_s: float, meta: dict | None = None) -> dict:
    release = {"engine": "graded", "graded": {"dt_s": float(dt_s), "n_columns": int(network.n_columns)},
               "n_neurons": int(network.n), **(meta or {})}
    return write_compiled(directory, graded_arrays(network), {"release": release}, schema=SCHEMA_ENGINE)


def write_hybrid_bundle(directory: Path, hybrid: HybridNetwork, dt_graded_s: float = 1 / 200, lattice=None,
                        exact: bool = True, meta: dict | None = None) -> dict:
    """``lattice`` is ``(lattice_networks, mapping)`` for an E3 bundle (``flycns.dynamics.hybrid.lattice_hybrid``);
    without it the bundle runs the lobe's own dynamics (E2, E4)."""
    arrays = hybrid_arrays(hybrid, exact)
    release = {"engine": "hybrid", "lif": lif_constants(hybrid.params),
               "graded": {"dt_s": float(dt_graded_s), "n_columns": int(hybrid.graded.n_columns)},
               "hybrid": hybrid_constants(hybrid, dt_graded_s), "source": "own",
               "n_spiking": int(hybrid.n_spiking), "n_graded": int(hybrid.graded.n), **(meta or {})}
    if lattice is not None:
        networks, mapping = lattice
        arrays.update(lattice_arrays(networks, mapping))
        release["source"] = "lattice"
        release["lattice"] = {"n_columns": int(networks[0].n_columns), "count": len(networks)}
    return write_compiled(directory, arrays, {"release": release}, schema=SCHEMA_ENGINE)


# ------------------------------------------------------------------------------------------------- scenarios

@dataclass
class Scenario:
    """A bundle and one run of it. ``steps`` counts LIF steps for a LIF scenario and frames otherwise."""

    arrays: dict[str, np.ndarray]
    constants: dict[str, Any]
    steps: int
    seed: int = 0
    drive: Drive = field(default_factory=Drive)
    intensity: np.ndarray | None = None          # graded and hybrid: (frames, columns), or (frames, 2, columns)
    trace_neurons: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    record_graded: np.ndarray | None = None      # hybrid: graded units recorded (all when None)
    t_pre_s: float = 1.0
    grey: float = 0.5

    @property
    def engine(self) -> str:
        return self.constants["engine"]


def _drive_arrays(drive: Drive, params: LIFParams) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    act = sorted(drive.activate)
    steps, neurons, dv = [], [], []
    for step in sorted(drive.events):
        ns, dvs = drive.events[step]
        for n, d in zip(ns, dvs, strict=True):               # the order within a step is the order add.at sums in
            steps.append(step)
            neurons.append(n)
            dv.append(d)
    arrays = {"drive_activate_neuron": _i64(act), "drive_activate_rate_hz": _f64([drive.activate[i] for i in act]),
              "drive_event_step": _i64(steps), "drive_event_neuron": _i64(neurons), "drive_event_dv_mv": _f64(dv),
              "drive_silenced": _i64(drive.silenced)}
    constants: dict[str, Any] = {"modulated_steps": 0}
    if drive.modulated is not None:
        mod_neurons, rates, per_frame = drive.modulated
        arrays["drive_modulated_neuron"] = _i64(mod_neurons)
        arrays["drive_modulated_rate_hz"] = _f64(rates)
        constants["modulated_steps"] = int(per_frame)
    return arrays, constants


def _drive_from(arrays: dict[str, np.ndarray], constants: dict[str, Any]) -> Drive:
    activate = {int(n): float(r) for n, r in zip(arrays["drive_activate_neuron"], arrays["drive_activate_rate_hz"],
                                                 strict=True)}
    events: dict[int, tuple[list, list]] = {}
    for s, n, d in zip(arrays["drive_event_step"], arrays["drive_event_neuron"], arrays["drive_event_dv_mv"],
                       strict=True):
        events.setdefault(int(s), ([], []))
        events[int(s)][0].append(int(n))
        events[int(s)][1].append(float(d))
    modulated = None
    if "drive_modulated_neuron" in arrays:
        modulated = (arrays["drive_modulated_neuron"], arrays["drive_modulated_rate_hz"],
                     int(constants["modulated_steps"]))
    return Drive(activate=activate, events=events, silenced=arrays["drive_silenced"], modulated=modulated)


def write_scenario(directory: Path, scenario: Scenario) -> dict:
    params = lif_params(scenario.constants["lif"]) if "lif" in scenario.constants else LIFParams()
    drive_arrays, drive_constants = _drive_arrays(scenario.drive, params)
    arrays = {**scenario.arrays, **drive_arrays, "trace_neurons": _i64(scenario.trace_neurons)}
    if scenario.intensity is not None:
        arrays["stimulus_intensity"] = _f64(scenario.intensity)
    if scenario.record_graded is not None:
        arrays["record_graded"] = _i64(scenario.record_graded)
    run = {"steps": int(scenario.steps), "seed": int(scenario.seed), "t_pre_s": float(scenario.t_pre_s),
           "grey": float(scenario.grey), **drive_constants}
    if "graded" in scenario.constants:
        # the lead-in's step count, as the reference computes it (int(t_pre_s / dt)); never recomputed elsewhere
        run["pre_steps"] = int(scenario.t_pre_s / float(scenario.constants["graded"]["dt_s"]))
    release = {**scenario.constants, "run": run}
    return write_compiled(directory, arrays, {"release": release}, schema=SCHEMA_SCENARIO)


def read_scenario(directory: Path) -> Scenario:
    c = read_compiled(Path(directory), schema=SCHEMA_SCENARIO)
    release = dict(c.manifest["release"])
    run = release.pop("run")
    arrays = dict(c.arrays)
    drive = _drive_from(arrays, run)
    for name in list(arrays):
        if name.startswith("drive_"):
            del arrays[name]
    trace = arrays.pop("trace_neurons")
    intensity = arrays.pop("stimulus_intensity", None)
    record = arrays.pop("record_graded", None)
    return Scenario(arrays=arrays, constants=release, steps=int(run["steps"]), seed=int(run["seed"]), drive=drive,
                    intensity=intensity, trace_neurons=trace, record_graded=record, t_pre_s=float(run["t_pre_s"]),
                    grey=float(run["grey"]))


# ------------------------------------------------------------------------------------------ the reference run

def _hybrid_from(scenario: Scenario) -> HybridReference:
    a, k = scenario.arrays, scenario.constants
    network = graded_network(a, k["graded"]["n_columns"])
    hybrid = HybridNetwork(lobe=SimpleNamespace(network=network), spiking_neuron=a["spiking_neuron"],
                           lif_indptr=a["lif_indptr"], lif_indices=a["lif_indices"],
                           lif_weights_mv=a["lif_weight_mv"], bridge_source=a["bridge_source"],
                           bridge_target=a["bridge_target"], bridge_weight=a["bridge_weight"],
                           feedback_source=a["feedback_source"], feedback_target=a["feedback_target"],
                           feedback_weight=a["feedback_weight"], params=lif_params(k["lif"]),
                           bridge_gain_hz=float(k["hybrid"]["bridge_gain_hz"]), report={})
    engine = HybridReference(hybrid, float(k["graded"]["dt_s"]))
    if k.get("source") == "lattice":
        networks = [graded_network(a, k["lattice"]["n_columns"], f"lattice{i}") for i in range(k["lattice"]["count"])]
        engines = [GradedReference(net, float(k["graded"]["dt_s"])) for net in networks]
        engine.source = LatticeSource(engines, lattice_map(a), network.n, _NumpyOps)
    return engine


def run_reference(scenario: Scenario) -> dict[str, np.ndarray]:
    """The NumPy reference's output for a scenario, as the arrays ``write_expected`` stores."""
    a, k = scenario.arrays, scenario.constants
    if scenario.engine == "lif":
        engine = LIFReference(a["lif_indptr"], a["lif_indices"], a["lif_weight_mv"], lif_params(k["lif"]))
        run = engine.run(scenario.steps, scenario.drive, scenario.seed, scenario.trace_neurons)
        return _run_arrays(run)
    if scenario.engine == "graded":
        engine = GradedReference(graded_network(a, k["graded"]["n_columns"]), float(k["graded"]["dt_s"]))
        final, activity = engine.run(scenario.intensity[: scenario.steps])
        return {"activity": activity, "final_state": final}
    engine = _hybrid_from(scenario)
    out = engine.run(scenario.intensity[: scenario.steps], scenario.drive, scenario.seed, scenario.t_pre_s,
                     scenario.grey, scenario.record_graded, scenario.trace_neurons)
    return {**_run_arrays(out.spikes), "graded_units": _i64(out.graded_units), "graded": out.graded,
            "grey_release": out.grey_release}


def _run_arrays(run: Run) -> dict[str, np.ndarray]:
    return {"tick_indptr": _i64(run.tick_indptr), "neuron_index": _i32(run.neuron_index),
            "trace_neurons": _i32(run.trace_neurons), "traces_mv": np.asarray(run.traces_mv, dtype=np.float32)}


def write_expected(directory: Path, outputs: dict[str, np.ndarray], meta: dict | None = None) -> dict:
    counts = {name: list(array.shape) for name, array in outputs.items()}
    return write_compiled(directory, outputs, {"release": meta or {}, "counts": counts}, schema=SCHEMA_EXPECTED)


def read_expected(directory: Path) -> Compiled:
    return read_compiled(Path(directory), schema=SCHEMA_EXPECTED)
