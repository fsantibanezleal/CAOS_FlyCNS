"""The whole CNS as one system: the graded optic lobes coupled to the published spiking model of everything else.

The optic lobes run as graded units (``flycns.optic_lobe``) every 5 ms, flyvis's step; every other neuron of the
release runs as the published leaky integrate-and-fire neuron every 0.1 ms. The coupling uses the release's synapses
in both directions, and states its one free parameter:

- **Graded to spiking (the bridge).** A graded neuron releases transmitter continuously. Its release above the grey
  steady state, ``D = max(V, 0) - max(V_grey, 0)``, acts on each spiking target as a spike train of rate
  ``beta x D`` would: every synapse adds ``sign x w_syn`` to the target's ``g`` per spike, so the connection adds
  ``beta x D x sign x n x w_syn`` to ``g`` per second (``n`` its synapse count, ``w_syn`` = 0.275 mV). At grey the
  spiking CNS receives nothing, as the published model fires nothing without input. ``beta`` (spikes per second per
  unit of release, 100 by default) is the free parameter every hybrid in the survey has, stated with the result.
- **Spiking to graded (feedback).** A spiking neuron acts on a graded target as release of ``r / beta``, ``r`` its
  firing rate filtered with the published synaptic time constant (5 ms); each graded target receives from each
  presynaptic class of spiking neurons flyvis's initialisation scale, ``0.01 x 2`` in total per unit of that release,
  spread over the class's synapses, with the transmitter's sign.

The graded drive enters ``g`` in the synapses slot of each LIF step, for neurons that are not refractory, like any
synaptic input; it is recomputed at every graded step and held in between.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..compiled import Compiled
from .graded import GradedNetwork
from .lif import LIFParams, Run

if TYPE_CHECKING:                                   # optic_lobe imports this package: import it lazily
    from ..optic_lobe import OpticLobe


@dataclass
class HybridNetwork:
    lobe: OpticLobe
    spiking_neuron: np.ndarray          # compiled neuron of each spiking unit
    lif_indptr: np.ndarray              # CSR over spiking units (by presynaptic unit)
    lif_indices: np.ndarray
    lif_weights_mv: np.ndarray
    bridge_source: np.ndarray           # graded unit
    bridge_target: np.ndarray           # spiking unit
    bridge_weight: np.ndarray           # mV added to g per second per unit of release deviation
    feedback_source: np.ndarray         # spiking unit
    feedback_target: np.ndarray         # graded unit
    feedback_weight: np.ndarray         # graded input per unit of release (= rate / beta)
    params: LIFParams
    bridge_gain_hz: float
    report: dict

    @property
    def graded(self) -> GradedNetwork:
        return self.lobe.network

    @property
    def n_spiking(self) -> int:
        return len(self.spiking_neuron)


def build_hybrid(graph: Compiled, lobe: OpticLobe, params: LIFParams | None = None, bridge_gain_hz: float = 100.0,
                 feedback_rho: float = 0.01, feedback_offsets: float = 2.0) -> HybridNetwork:
    from ..optic_lobe import CT1_COMPARTMENTS

    params = params or LIFParams()
    n_all = graph.n_neurons
    names = np.array(lobe.classes, dtype=object)[lobe.unit_class]
    own = (lobe.neuron >= 0) & ~lobe.stand_in & ~np.isin(names, CT1_COMPARTMENTS)
    unit_of = np.full(n_all, -1, dtype=np.int64)
    unit_of[lobe.neuron[own]] = np.flatnonzero(own)
    graded = unit_of >= 0
    spiking = np.flatnonzero(~graded)
    local = np.full(n_all, -1, dtype=np.int64)
    local[spiking] = np.arange(len(spiking))

    indptr = graph["csr_indptr"].astype(np.int64)
    pre = np.repeat(np.arange(n_all, dtype=np.int64), np.diff(indptr))
    post = graph["csr_indices"].astype(np.int64)
    count = graph["csr_count"].astype(np.float64)
    sign = graph["neuron_sign"].astype(np.float64)

    ss = ~graded[pre] & ~graded[post]
    rows = local[pre[ss]]
    lif_indptr = np.zeros(len(spiking) + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=len(spiking)), out=lif_indptr[1:])
    order = np.argsort(rows, kind="stable")
    lif_indices = local[post[ss]][order]
    lif_weights = (sign[pre[ss]] * count[ss] * params.w_syn_mv)[order]

    gs = graded[pre] & ~graded[post]
    bridge_weight = sign[pre[gs]] * count[gs] * params.w_syn_mv * bridge_gain_hz

    sg = ~graded[pre] & graded[post]
    fb_target = unit_of[post[sg]]
    fb_class = graph["neuron_type"][pre[sg]].astype(np.int64)
    key = fb_target * (int(fb_class.max(initial=0)) + 1) + fb_class
    _, inverse = np.unique(key, return_inverse=True)
    per_class = np.bincount(inverse, weights=count[sg])[inverse]
    feedback_weight = sign[pre[sg]] * feedback_rho * feedback_offsets * count[sg] / np.maximum(per_class, 1e-12)

    report = {"graded_units": int(lobe.network.n), "spiking_neurons": int(len(spiking)),
              "spiking_connections": int(ss.sum()), "bridge_connections": int(gs.sum()),
              "bridge_synapses": float(count[gs].sum()), "feedback_connections": int(sg.sum()),
              "feedback_synapses": float(count[sg].sum()), "bridge_gain_hz": bridge_gain_hz}
    return HybridNetwork(lobe=lobe, spiking_neuron=spiking, lif_indptr=lif_indptr, lif_indices=lif_indices,
                         lif_weights_mv=lif_weights, bridge_source=unit_of[pre[gs]], bridge_target=local[post[gs]],
                         bridge_weight=bridge_weight, feedback_source=local[pre[sg]], feedback_target=fb_target,
                         feedback_weight=feedback_weight, params=params, bridge_gain_hz=bridge_gain_hz,
                         report=report)


@dataclass
class HybridRun:
    spikes: Run                         # the spiking units' spikes (unit indices into ``spiking_neuron``)
    graded_units: np.ndarray            # the recorded graded units
    graded: np.ndarray                  # (graded steps, recorded units) voltages after each graded step
    grey_release: np.ndarray            # max(V, 0) at the grey steady state, every graded unit


class OwnSource:
    """E2: the optic lobe's own graded dynamics on the release's wiring, with the spiking CNS's feedback."""

    def __init__(self, engine):
        self.engine = engine

    def grey(self, t_pre_s, grey):
        return self.engine.steady_state(t_pre_s, grey)

    def step(self, state, frame, feedback):
        return self.engine.step(state, self.engine.column_current(frame) + feedback)

    def units(self, state):
        return state


class LatticeSource:
    """E3: flyvis's own network on each eye (two lattice engines), its cells' activity carried onto the optic-lobe
    units of ``mapping`` (``flycns.mapped.map_to_lattice``); the other units hold 0, so their release never
    deviates from grey. Feedback from the spiking CNS does not enter flyvis's lattice."""

    def __init__(self, engines, mapping, n_units: int, xp):
        self.engines = engines
        self.xp = xp
        unit_rep = np.concatenate([np.full(len(n), u) for u, n in zip(mapping.unit, mapping.nodes, strict=True)])
        side_rep = np.concatenate([np.full(len(n), s) for s, n in zip(mapping.side, mapping.nodes, strict=True)])
        node_rep = np.concatenate(mapping.nodes)
        weight = np.concatenate([np.full(len(n), 1.0 / len(n)) for n in mapping.nodes])
        self.parts = [(xp.asarray(unit_rep[side_rep == s], np.int64), xp.asarray(node_rep[side_rep == s], np.int64),
                       xp.asarray(weight[side_rep == s], np.float32)) for s in (0, 1)]
        self.n_units = n_units

    def grey(self, t_pre_s, grey):
        return [e.steady_state(t_pre_s, grey) for e in self.engines]

    def step(self, state, frame, feedback):
        return [e.step(v, e.column_current(f)) for e, v, f in zip(self.engines, state, frame, strict=True)]

    def units(self, state):
        out = self.xp.zeros(self.n_units)
        for (unit, node, weight), v in zip(self.parts, state, strict=True):
            out = self.xp.add_at(out, unit, v[node] * weight)
        return out


class _NumpyOps:
    @staticmethod
    def zeros(n):
        return np.zeros(n)

    @staticmethod
    def asarray(values, dtype):
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def add_at(out, index, values):
        np.add.at(out, index, np.asarray(values))
        return out


class _TorchOps:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device

    def zeros(self, n):
        return self.torch.zeros(n, device=self.device)

    def asarray(self, values, dtype):
        return self.torch.as_tensor(np.asarray(values, dtype=dtype), device=self.device)

    def add_at(self, out, index, values):
        return out.index_add_(0, index, values.to(out.dtype))


class _Hybrid:
    """The loop both engines share; subclasses provide the arithmetic."""

    def __init__(self, hybrid: HybridNetwork, dt_graded_s: float = 1 / 200):
        self.h = hybrid
        self.dt_graded = float(dt_graded_s)
        self.steps_per_graded = int(round(dt_graded_s * 1000.0 / hybrid.params.dt_ms))
        self.dt_lif_s = hybrid.params.dt_ms / 1000.0
        self.tau_filter_s = hybrid.params.tau_ms / 1000.0
        self.decay = float(np.exp(-self.dt_lif_s / self.tau_filter_s))

    def run(self, intensity: np.ndarray, drive=None, seed: int = 0, t_pre_s: float = 1.0, grey: float = 0.5,
            record_graded: np.ndarray | None = None, trace_spiking: np.ndarray | None = None) -> HybridRun:
        """Step through frames, one per graded step (5 ms), from the grey steady state. For E2 a frame is the
        intensity of every eye column; for E3 (``LatticeSource``) it is (left, right) lattice intensities."""
        source = self.source
        g_state = source.grey(t_pre_s, grey)
        v = source.units(g_state)
        grey_release = self._relu(v)
        record = np.zeros(0, dtype=np.int64) if record_graded is None else np.asarray(record_graded)
        state = self.lif.start(drive, seed, trace_spiking)
        rate = self._zeros(self.h.n_spiking)
        out = []
        for frame in np.asarray(intensity):
            fb = self._feedback(rate) / self.h.bridge_gain_hz
            g_state = source.step(g_state, frame, fb)
            v = source.units(g_state)
            g_input = self._bridge(self._relu(v) - grey_release) * self.dt_lif_s
            for _ in range(self.steps_per_graded):
                spiked = self.lif.advance(state, g_input)
                rate = rate * self.decay
                rate = self._add_spikes(rate, spiked, 1.0 / self.tau_filter_s)
            out.append(self._numpy(v[record]))
        graded = np.stack(out).astype(np.float32) if out else np.zeros((0, len(record)), dtype=np.float32)
        return HybridRun(spikes=self.lif.finish(state), graded_units=record, graded=graded,
                         grey_release=self._numpy(grey_release).astype(np.float32))


class HybridReference(_Hybrid):
    """NumPy (float64), for small networks and as the reference."""

    def __init__(self, hybrid: HybridNetwork, dt_graded_s: float = 1 / 200):
        from .graded import GradedReference
        from .lif import LIFReference

        super().__init__(hybrid, dt_graded_s)
        self.graded = GradedReference(hybrid.graded, dt_graded_s)
        self.lif = LIFReference(hybrid.lif_indptr, hybrid.lif_indices, hybrid.lif_weights_mv, hybrid.params)
        self.source = OwnSource(self.graded)

    def _relu(self, v):
        return np.maximum(v, 0.0)

    def _zeros(self, n):
        return np.zeros(n)

    def _numpy(self, x):
        return np.asarray(x)

    def _bridge(self, deviation):
        h = self.h
        return np.bincount(h.bridge_target, weights=h.bridge_weight * deviation[h.bridge_source],
                           minlength=h.n_spiking)

    def _feedback(self, rate):
        h = self.h
        return np.bincount(h.feedback_target, weights=h.feedback_weight * rate[h.feedback_source],
                           minlength=h.graded.n)

    def _add_spikes(self, rate, spiked, amount):
        rate[spiked] += amount
        return rate


class HybridTorch(_Hybrid):
    """PyTorch (float32), for the GPU: the bridge and the feedback as sparse matrices."""

    def __init__(self, hybrid: HybridNetwork, dt_graded_s: float = 1 / 200, device: str | None = None):
        import torch

        from .graded import GradedTorch
        from .lif import LIFTorch

        super().__init__(hybrid, dt_graded_s)
        self.torch = torch
        self.graded = GradedTorch(hybrid.graded, dt_graded_s, device)
        self.device = self.graded.device
        self.lif = LIFTorch(hybrid.lif_indptr, hybrid.lif_indices, hybrid.lif_weights_mv, hybrid.params,
                            self.device)
        self.source = OwnSource(self.graded)
        self.bridge = self._csr(hybrid.bridge_target, hybrid.bridge_source, hybrid.bridge_weight,
                                (hybrid.n_spiking, hybrid.graded.n))
        self.feedback = self._csr(hybrid.feedback_target, hybrid.feedback_source, hybrid.feedback_weight,
                                  (hybrid.graded.n, hybrid.n_spiking))

    def _csr(self, rows, cols, values, shape):
        torch = self.torch
        rows_t = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        cols_t = torch.as_tensor(np.asarray(cols, dtype=np.int64), device=self.device)
        vals_t = torch.as_tensor(np.asarray(values, dtype=np.float32), device=self.device)
        order = torch.argsort(rows_t * shape[1] + cols_t)
        crow = torch.zeros(shape[0] + 1, dtype=torch.int64, device=self.device)
        crow[1:] = torch.cumsum(torch.bincount(rows_t, minlength=shape[0]), 0)
        return torch.sparse_csr_tensor(crow, cols_t[order], vals_t[order], size=shape)

    def _relu(self, v):
        return self.torch.relu(v)

    def _zeros(self, n):
        return self.torch.zeros(n, device=self.device)

    def _numpy(self, x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def _bridge(self, deviation):
        return (self.bridge @ deviation[:, None])[:, 0]

    def _feedback(self, rate):
        return (self.feedback @ rate[:, None])[:, 0]

    def _add_spikes(self, rate, spiked, amount):
        rate[spiked] += amount
        return rate


def lattice_hybrid(hybrid: HybridNetwork, lattice_networks, mapping, dt_graded_s: float = 1 / 200,
                   device: str | None = None) -> HybridTorch:
    """E3 on the GPU: ``lattice_networks`` are flyvis's network (``flyvis.lattice_network``) for the left and the
    right eye; frames given to ``run`` are then (left, right) lattice intensities, shape (frames, 2, 721)."""
    from .graded import GradedTorch

    engine = HybridTorch(hybrid, dt_graded_s, device)
    engines = [GradedTorch(net, dt_graded_s, engine.device) for net in lattice_networks]
    engine.source = LatticeSource(engines, mapping, hybrid.graded.n, _TorchOps(engine.torch, engine.device))
    return engine
