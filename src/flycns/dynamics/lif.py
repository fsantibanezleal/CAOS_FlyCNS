"""The published whole-brain leaky integrate-and-fire model (Shiu et al., Nature 634:210-219, 2024).

Two engines with the same step: ``LIFReference`` (NumPy, the reference, deterministic) and ``LIFTorch`` (PyTorch, for
the GPU, compared to the reference by tolerance). The step follows Brian2's default schedule, because the published
model is a Brian2 program and a literal transcription of it is the parity test:

1. groups: exact integration of v and g for neurons that are not refractory;
2. thresholds: v > v_th and not refractory -> spike;
3. synapses: spikes emitted t_dly ago add their weights to g; Poisson and fixed events add to v; only neurons that
   are not refractory receive them (Brian2 discards input to refractory neurons for "unless refractory" variables);
4. resets: v <- v_rst, g <- 0 for the neurons that spiked this step.

See ``docs/design/features/lif/design.md`` for the equations.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..rng import event_threshold, hash3


@dataclass(frozen=True)
class LIFParams:
    """The published constants (``model.py``, ``default_params``)."""

    v0_mv: float = -52.0        # resting potential (Kakaria and de Bivort 2017)
    v_rst_mv: float = -52.0     # reset potential
    v_th_mv: float = -45.0      # threshold
    t_mbr_ms: float = 20.0      # membrane time constant
    tau_ms: float = 5.0         # synaptic time constant (Juergensen et al.)
    t_rfc_ms: float = 2.2       # refractory period (Lazar et al.)
    t_dly_ms: float = 1.8       # synaptic delay (Paul et al. 2015)
    w_syn_mv: float = 0.275     # weight per synapse, the model's single free parameter
    f_poi: float = 250.0        # Poisson input scaling: one event crosses threshold
    dt_ms: float = 0.1          # Brian2's default step

    @property
    def delay_steps(self) -> int:
        return int(round(self.t_dly_ms / self.dt_ms))

    @property
    def refractory_steps(self) -> int:
        return int(round(self.t_rfc_ms / self.dt_ms))

    def decay(self) -> tuple[float, float, float]:
        """(a, b, c): v' = v0 + (v - v0) a + g c, g' = g b over one step (the closed form of the linear system)."""
        a = math.exp(-self.dt_ms / self.t_mbr_ms)
        b = math.exp(-self.dt_ms / self.tau_ms)
        c = self.tau_ms / (self.tau_ms - self.t_mbr_ms) * (b - a)
        return a, b, c


@dataclass
class Drive:
    """What drives a run: Poisson activation, fixed events, silencing.

    Silencing is the published one (``silence()`` in ``model.py``): every synapse *from* a silenced neuron carries
    nothing. The neuron still receives input and may spike; its spikes reach no one.
    """

    activate: dict[int, float] = field(default_factory=dict)          # neuron -> rate (Hz)
    events: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # step -> (neurons, dv mV)
    silenced: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class Run:
    """The result of a run: spikes per step (as a CSR over steps) and the requested voltage traces."""

    n_neurons: int
    steps: int
    dt_ms: float
    tick_indptr: np.ndarray
    neuron_index: np.ndarray
    trace_neurons: np.ndarray
    traces_mv: np.ndarray       # (steps, len(trace_neurons)), float32

    def spike_counts(self) -> np.ndarray:
        return np.bincount(self.neuron_index, minlength=self.n_neurons)

    def rates_hz(self) -> np.ndarray:
        return self.spike_counts() / (self.steps * self.dt_ms / 1000.0)

    def spike_times(self) -> tuple[np.ndarray, np.ndarray]:
        """(step of each spike, neuron of each spike), ordered by step then neuron."""
        steps = np.repeat(np.arange(self.steps), np.diff(self.tick_indptr))
        return steps, self.neuron_index


def synaptic_weights(indptr: np.ndarray, indices: np.ndarray, counts: np.ndarray, signs: np.ndarray,
                     w_syn_mv: float, silenced: np.ndarray | None = None) -> np.ndarray:
    """Per-connection weight in mV: sign of the presynaptic neuron x synapse count x w_syn. The outgoing connections of
    silenced neurons get 0, as the published ``silence()`` does (``syn.w['{i} == i'] = 0``)."""
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    weights = signs[rows].astype(np.float64) * counts.astype(np.float64) * w_syn_mv
    if silenced is not None and len(silenced):
        mute = np.zeros(len(indptr) - 1, dtype=bool)
        mute[np.asarray(silenced, dtype=np.int64)] = True
        weights[mute[rows]] = 0.0
    return weights


class LIFReference:
    """The NumPy reference engine: float64 state, deliveries accumulated in a fixed order."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, weights_mv: np.ndarray,
                 params: LIFParams | None = None):
        self.params = params or LIFParams()
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = np.asarray(weights_mv, dtype=np.float64)
        self.n = len(self.indptr) - 1

    def _deliver(self, spikers: np.ndarray) -> np.ndarray:
        if len(spikers) == 0:
            return np.zeros(self.n)
        starts, ends = self.indptr[spikers], self.indptr[spikers + 1]
        lengths = ends - starts
        if lengths.sum() == 0:
            return np.zeros(self.n)
        offsets = np.repeat(starts - np.r_[0, np.cumsum(lengths)[:-1]], lengths) + np.arange(lengths.sum())
        return np.bincount(self.indices[offsets], weights=self.weights[offsets], minlength=self.n)

    def run(self, steps: int, drive: Drive | None = None, seed: int = 0,
            trace_neurons: np.ndarray | None = None) -> Run:
        p = self.params
        drive = drive or Drive()
        a, b, c = p.decay()
        v = np.full(self.n, p.v0_mv)
        g = np.zeros(self.n)
        last = np.full(self.n, -(10**9), dtype=np.int64)
        refractory = np.full(self.n, p.refractory_steps, dtype=np.int64)
        act = np.array(sorted(drive.activate), dtype=np.int64)
        thresholds = np.array([event_threshold(drive.activate[i], p.dt_ms / 1000.0) for i in act], dtype=np.uint64)
        refractory[act] = 0                                     # activated neurons have no refractory period
        mute = np.zeros(self.n, dtype=bool)
        mute[np.asarray(drive.silenced, dtype=np.int64)] = True
        w_poi = p.w_syn_mv * p.f_poi
        pending: deque[np.ndarray] = deque([np.zeros(0, dtype=np.int64)] * p.delay_steps)
        trace_neurons = np.zeros(0, dtype=np.int64) if trace_neurons is None else np.asarray(trace_neurons)
        traces = np.zeros((steps, len(trace_neurons)), dtype=np.float32)
        tick_indptr = np.zeros(steps + 1, dtype=np.int64)
        spikes_out = []
        for k in range(steps):
            # 1. groups
            free = (k - last) >= refractory
            v_new = p.v0_mv + (v - p.v0_mv) * a + g * c
            v = np.where(free, v_new, v)
            g = np.where(free, g * b, g)
            # 2. thresholds (a neuron that spikes is refractory from this step on)
            spike = np.flatnonzero(free & (v > p.v_th_mv))
            last[spike] = k
            free[spike] = False
            # 3. synapses: delayed deliveries into g, events into v. Both variables are 'unless refractory' in the
            # published model, and Brian2 then applies synaptic updates only to postsynaptic neurons that are not
            # refractory: input arriving during the refractory period is discarded.
            pending.append(spike[~mute[spike]])                 # a silenced neuron's spikes reach no one
            g += self._deliver(pending.popleft()) * free
            if len(act):
                hit = hash3(seed, act, k) < thresholds
                targets = act[hit]
                v[targets[free[targets]]] += w_poi
            if k in drive.events:
                neurons, dv = drive.events[k]
                neurons = np.asarray(neurons, dtype=np.int64)
                dv = np.asarray(dv, dtype=np.float64)
                keep = free[neurons]
                np.add.at(v, neurons[keep], dv[keep])
            # 4. resets
            v[spike] = p.v_rst_mv
            g[spike] = 0.0
            spikes_out.append(spike)
            tick_indptr[k + 1] = tick_indptr[k] + len(spike)
            if len(trace_neurons):
                traces[k] = v[trace_neurons]
        neuron_index = np.concatenate(spikes_out) if spikes_out else np.zeros(0, dtype=np.int64)
        return Run(self.n, steps, p.dt_ms, tick_indptr, neuron_index.astype(np.int32), trace_neurons.astype(np.int32),
                   traces)


class LIFTorch:
    """The same step in PyTorch (float32), for the GPU. Deliveries are pushed from the neurons that spiked."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, weights_mv: np.ndarray,
                 params: LIFParams | None = None, device: str | None = None):
        import torch

        self.torch = torch
        self.params = params or LIFParams()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.indptr = torch.as_tensor(np.asarray(indptr, dtype=np.int64), device=self.device)
        self.indices = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        self.weights = torch.as_tensor(np.asarray(weights_mv, dtype=np.float32), device=self.device)
        self.n = len(indptr) - 1

    def _deliver(self, spikers):
        torch = self.torch
        out = torch.zeros(self.n, device=self.device)
        if spikers.numel() == 0:
            return out
        starts, ends = self.indptr[spikers], self.indptr[spikers + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return out
        base = torch.repeat_interleave(starts - torch.cumsum(lengths, 0) + lengths, lengths)
        offsets = base + torch.arange(total, device=self.device)
        out.index_add_(0, self.indices[offsets], self.weights[offsets])
        return out

    def run(self, steps: int, drive: Drive | None = None, seed: int = 0,
            trace_neurons: np.ndarray | None = None) -> Run:
        torch = self.torch
        p = self.params
        drive = drive or Drive()
        a, b, c = p.decay()
        dev = self.device
        v = torch.full((self.n,), p.v0_mv, device=dev)
        g = torch.zeros(self.n, device=dev)
        last = torch.full((self.n,), -(10**9), dtype=torch.int64, device=dev)
        refractory = torch.full((self.n,), p.refractory_steps, dtype=torch.int64, device=dev)
        act_np = np.array(sorted(drive.activate), dtype=np.int64)
        thresholds = np.array([event_threshold(drive.activate[i], p.dt_ms / 1000.0) for i in act_np], dtype=np.uint64)
        act = torch.as_tensor(act_np, device=dev)
        if len(act_np):
            refractory[act] = 0
        mute = torch.zeros(self.n, dtype=torch.bool, device=dev)
        if len(drive.silenced):
            mute[torch.as_tensor(np.asarray(drive.silenced, dtype=np.int64), device=dev)] = True
        w_poi = p.w_syn_mv * p.f_poi
        empty = torch.zeros(0, dtype=torch.int64, device=dev)
        pending = deque([empty] * p.delay_steps)
        trace_np = np.zeros(0, dtype=np.int64) if trace_neurons is None else np.asarray(trace_neurons)
        trace_t = torch.as_tensor(trace_np, device=dev)
        traces = torch.zeros((steps, len(trace_np)), device=dev)
        spikes_out = []
        for k in range(steps):
            free = (k - last) >= refractory
            v = torch.where(free, p.v0_mv + (v - p.v0_mv) * a + g * c, v)
            g = torch.where(free, g * b, g)
            spike = torch.nonzero(free & (v > p.v_th_mv)).flatten()
            last[spike] = k
            free[spike] = False
            pending.append(spike[~mute[spike]])
            g = g + self._deliver(pending.popleft()) * free       # refractory neurons discard input (see above)
            if len(act_np):
                hit = hash3(seed, act_np, k) < thresholds       # the same counter-based events as the reference
                if hit.any():
                    targets = act[torch.as_tensor(np.flatnonzero(hit), device=dev)]
                    targets = targets[free[targets]]
                    v[targets] += w_poi
            if k in drive.events:
                neurons, dv = drive.events[k]
                neurons_t = torch.as_tensor(np.asarray(neurons, dtype=np.int64), device=dev)
                dv_t = torch.as_tensor(np.asarray(dv, dtype=np.float32), device=dev)
                keep = free[neurons_t]
                v.index_add_(0, neurons_t[keep], dv_t[keep])
            v[spike] = p.v_rst_mv
            g[spike] = 0.0
            spikes_out.append(spike)
            if len(trace_np):
                traces[k] = v[trace_t]
        counts = np.array([s.numel() for s in spikes_out], dtype=np.int64)
        tick_indptr = np.r_[0, np.cumsum(counts)]
        neuron_index = torch.cat(spikes_out).cpu().numpy() if len(spikes_out) else np.zeros(0)
        return Run(self.n, steps, p.dt_ms, tick_indptr, neuron_index.astype(np.int32), trace_np.astype(np.int32),
                   traces.cpu().numpy().astype(np.float32))
