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
    # E4's spike-frequency adaptation (off in the published model): each spike lowers the neuron's effective rest by
    # adaptation_mv, which decays with tau_adaptation_ms (1.5 mV and 200 ms in flyverse's stabilised model)
    adaptation_mv: float = 0.0
    tau_adaptation_ms: float = 200.0

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
    # Poisson input whose rate changes over time (light on photoreceptors): neurons, rates (frames x neurons, Hz),
    # and the steps each frame lasts. Like activated neurons, these have no refractory period.
    modulated: tuple[np.ndarray, np.ndarray, int] | None = None


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


def _modulated(drive: Drive, p: LIFParams) -> tuple[np.ndarray, np.ndarray, int]:
    """The modulated Poisson input as (neurons, uint64 thresholds per frame, steps per frame)."""
    if drive.modulated is None:
        return np.zeros(0, dtype=np.int64), np.zeros((1, 0), dtype=np.uint64), 1
    neurons, rates, steps = drive.modulated
    rates = np.asarray(rates, dtype=np.float64)
    p_event = np.clip(rates * p.dt_ms / 1000.0, 0.0, 1.0)
    thresholds = np.minimum(np.floor(p_event * 2**32), 2**32 - 1).astype(np.uint64)
    return np.asarray(neurons, dtype=np.int64), thresholds, int(steps)


def stabilised_weights(indptr: np.ndarray, indices: np.ndarray, counts: np.ndarray, signs: np.ndarray,
                       types: np.ndarray, w_syn_mv: float, cap: float = 60.0, same_type_factor: float = 0.1,
                       fan_in_limit: float = 5000.0) -> np.ndarray:
    """E4's weights, flyverse's stabilisers in a stated order: each connection's count capped at ``cap``
    synapse-equivalents; connections between neurons of the same type scaled by ``same_type_factor``; then every
    neuron whose input so obtained exceeds ``fan_in_limit`` synapse-equivalents has all its inputs scaled down to the
    limit. Returns the per-connection weight in mV, signed by the presynaptic neuron."""
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    equivalents = np.minimum(np.asarray(counts, dtype=np.float64), cap)
    same = np.asarray(types)[rows] == np.asarray(types)[indices]
    equivalents = np.where(same, equivalents * same_type_factor, equivalents)
    fan_in = np.bincount(indices, weights=equivalents, minlength=len(indptr) - 1)
    scale = np.where(fan_in > fan_in_limit, fan_in_limit / np.maximum(fan_in, 1e-12), 1.0)
    return np.asarray(signs, dtype=np.float64)[rows] * equivalents * scale[indices] * w_syn_mv


class LIFState:
    """Everything a run carries from one step to the next; ``k`` is the next step to take."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class LIFReference:
    """The NumPy reference engine: float64 state, deliveries accumulated in a fixed order.

    ``run`` takes a whole run; ``start``, ``advance`` and ``finish`` take it one step at a time, with an optional extra
    input to ``g`` per step (the hybrid's graded drive), applied in the synapses slot to neurons that are not
    refractory, like any synaptic input.
    """

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

    def start(self, drive: Drive | None = None, seed: int = 0, trace_neurons: np.ndarray | None = None) -> LIFState:
        p = self.params
        drive = drive or Drive()
        act = np.array(sorted(drive.activate), dtype=np.int64)
        refractory = np.full(self.n, p.refractory_steps, dtype=np.int64)
        refractory[act] = 0                                     # activated neurons have no refractory period
        mod_neurons, mod_thresholds, mod_steps = _modulated(drive, p)
        refractory[mod_neurons] = 0
        mute = np.zeros(self.n, dtype=bool)
        mute[np.asarray(drive.silenced, dtype=np.int64)] = True
        return LIFState(mod_neurons=mod_neurons, mod_thresholds=mod_thresholds, mod_steps=mod_steps,
                        adaptation=np.zeros(self.n),
            k=0, v=np.full(self.n, p.v0_mv), g=np.zeros(self.n), last=np.full(self.n, -(10**9), dtype=np.int64),
            refractory=refractory, act=act, mute=mute, drive=drive, seed=seed,
            thresholds=np.array([event_threshold(drive.activate[i], p.dt_ms / 1000.0) for i in act],
                                dtype=np.uint64),
            pending=deque([np.zeros(0, dtype=np.int64)] * p.delay_steps),
            trace_neurons=np.zeros(0, dtype=np.int64) if trace_neurons is None else np.asarray(trace_neurons),
            traces=[], spikes=[])

    def advance(self, s: LIFState, g_input: np.ndarray | None = None) -> np.ndarray:
        """One step; returns the neurons that spiked."""
        p = self.params
        a, b, c = p.decay()
        k = s.k
        # 1. groups (with adaptation, the effective rest is v0 minus the adaptation, held over the step)
        free = (k - s.last) >= s.refractory
        rest = p.v0_mv - s.adaptation if p.adaptation_mv else p.v0_mv
        v_new = rest + (s.v - rest) * a + s.g * c
        s.v = np.where(free, v_new, s.v)
        s.g = np.where(free, s.g * b, s.g)
        # 2. thresholds (a neuron that spikes is refractory from this step on)
        spike = np.flatnonzero(free & (s.v > p.v_th_mv))
        s.last[spike] = k
        free[spike] = False
        # 3. synapses: delayed deliveries into g, events into v. Both variables are 'unless refractory' in the
        # published model, and Brian2 then applies synaptic updates only to postsynaptic neurons that are not
        # refractory: input arriving during the refractory period is discarded.
        s.pending.append(spike[~s.mute[spike]])                 # a silenced neuron's spikes reach no one
        s.g += self._deliver(s.pending.popleft()) * free
        if g_input is not None:
            s.g += np.asarray(g_input, dtype=np.float64) * free
        if len(s.act):
            hit = hash3(s.seed, s.act, k) < s.thresholds
            targets = s.act[hit]
            s.v[targets[free[targets]]] += p.w_syn_mv * p.f_poi
        if len(s.mod_neurons):
            frame = min(k // s.mod_steps, len(s.mod_thresholds) - 1)
            hit = hash3(s.seed, s.mod_neurons, k) < s.mod_thresholds[frame]
            targets = s.mod_neurons[hit]
            s.v[targets[free[targets]]] += p.w_syn_mv * p.f_poi
        if k in s.drive.events:
            neurons, dv = s.drive.events[k]
            neurons = np.asarray(neurons, dtype=np.int64)
            dv = np.asarray(dv, dtype=np.float64)
            keep = free[neurons]
            np.add.at(s.v, neurons[keep], dv[keep])
        # 4. resets
        s.v[spike] = p.v_rst_mv
        s.g[spike] = 0.0
        if p.adaptation_mv:
            s.adaptation *= math.exp(-p.dt_ms / p.tau_adaptation_ms)
            s.adaptation[spike] += p.adaptation_mv
        s.spikes.append(spike)
        if len(s.trace_neurons):
            s.traces.append(s.v[s.trace_neurons].astype(np.float32))
        s.k = k + 1
        return spike

    def finish(self, s: LIFState) -> Run:
        counts = np.array([len(x) for x in s.spikes], dtype=np.int64)
        tick_indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        neuron_index = np.concatenate(s.spikes) if s.spikes else np.zeros(0, dtype=np.int64)
        traces = (np.stack(s.traces) if s.traces else np.zeros((s.k, len(s.trace_neurons)), dtype=np.float32))
        return Run(self.n, s.k, self.params.dt_ms, tick_indptr, neuron_index.astype(np.int32),
                   s.trace_neurons.astype(np.int32), traces)

    def run(self, steps: int, drive: Drive | None = None, seed: int = 0,
            trace_neurons: np.ndarray | None = None) -> Run:
        s = self.start(drive, seed, trace_neurons)
        for _ in range(steps):
            self.advance(s)
        return self.finish(s)


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

    def start(self, drive: Drive | None = None, seed: int = 0, trace_neurons: np.ndarray | None = None) -> LIFState:
        torch = self.torch
        p = self.params
        drive = drive or Drive()
        dev = self.device
        act_np = np.array(sorted(drive.activate), dtype=np.int64)
        act = torch.as_tensor(act_np, device=dev)
        refractory = torch.full((self.n,), p.refractory_steps, dtype=torch.int64, device=dev)
        if len(act_np):
            refractory[act] = 0
        mod_np, mod_thresholds, mod_steps = _modulated(drive, p)
        if len(mod_np):
            refractory[torch.as_tensor(mod_np, device=dev)] = 0
        mute = torch.zeros(self.n, dtype=torch.bool, device=dev)
        if len(drive.silenced):
            mute[torch.as_tensor(np.asarray(drive.silenced, dtype=np.int64), device=dev)] = True
        trace_np = np.zeros(0, dtype=np.int64) if trace_neurons is None else np.asarray(trace_neurons)
        empty = torch.zeros(0, dtype=torch.int64, device=dev)
        return LIFState(
            mod_np=mod_np, mod_t=torch.as_tensor(mod_np, device=dev), mod_thresholds=mod_thresholds,
            mod_steps=mod_steps, adaptation=torch.zeros(self.n, device=dev),
            k=0, v=torch.full((self.n,), p.v0_mv, device=dev), g=torch.zeros(self.n, device=dev),
            last=torch.full((self.n,), -(10**9), dtype=torch.int64, device=dev), refractory=refractory,
            act_np=act_np, act=act, mute=mute, drive=drive, seed=seed,
            thresholds=np.array([event_threshold(drive.activate[i], p.dt_ms / 1000.0) for i in act_np],
                                dtype=np.uint64),
            pending=deque([empty] * p.delay_steps), trace_np=trace_np,
            trace_t=torch.as_tensor(trace_np, device=dev), traces=[], spikes=[])

    def advance(self, s: LIFState, g_input=None):
        torch = self.torch
        p = self.params
        a, b, c = p.decay()
        dev = self.device
        k = s.k
        free = (k - s.last) >= s.refractory
        rest = p.v0_mv - s.adaptation if p.adaptation_mv else p.v0_mv
        s.v = torch.where(free, rest + (s.v - rest) * a + s.g * c, s.v)
        s.g = torch.where(free, s.g * b, s.g)
        spike = torch.nonzero(free & (s.v > p.v_th_mv)).flatten()
        s.last[spike] = k
        free[spike] = False
        s.pending.append(spike[~s.mute[spike]])
        s.g = s.g + self._deliver(s.pending.popleft()) * free       # refractory neurons discard input (see above)
        if g_input is not None:
            s.g = s.g + g_input * free
        if len(s.act_np):
            hit = hash3(s.seed, s.act_np, k) < s.thresholds          # the same counter-based events as the reference
            if hit.any():
                targets = s.act[torch.as_tensor(np.flatnonzero(hit), device=dev)]
                targets = targets[free[targets]]
                s.v[targets] += p.w_syn_mv * p.f_poi
        if len(s.mod_np):
            frame = min(k // s.mod_steps, len(s.mod_thresholds) - 1)
            hit = hash3(s.seed, s.mod_np, k) < s.mod_thresholds[frame]
            if hit.any():
                targets = s.mod_t[torch.as_tensor(np.flatnonzero(hit), device=dev)]
                targets = targets[free[targets]]
                s.v[targets] += p.w_syn_mv * p.f_poi
        if k in s.drive.events:
            neurons, dv = s.drive.events[k]
            neurons_t = torch.as_tensor(np.asarray(neurons, dtype=np.int64), device=dev)
            dv_t = torch.as_tensor(np.asarray(dv, dtype=np.float32), device=dev)
            keep = free[neurons_t]
            s.v.index_add_(0, neurons_t[keep], dv_t[keep])
        s.v[spike] = p.v_rst_mv
        s.g[spike] = 0.0
        if p.adaptation_mv:
            s.adaptation = s.adaptation * math.exp(-p.dt_ms / p.tau_adaptation_ms)
            s.adaptation[spike] += p.adaptation_mv
        s.spikes.append(spike)
        if len(s.trace_np):
            s.traces.append(s.v[s.trace_t].clone())
        s.k = k + 1
        return spike

    def finish(self, s: LIFState) -> Run:
        torch = self.torch
        counts = np.array([x.numel() for x in s.spikes], dtype=np.int64)
        tick_indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        neuron_index = torch.cat(s.spikes).cpu().numpy() if s.spikes else np.zeros(0)
        traces = (torch.stack(s.traces).cpu().numpy().astype(np.float32) if s.traces
                  else np.zeros((s.k, len(s.trace_np)), dtype=np.float32))
        return Run(self.n, s.k, self.params.dt_ms, tick_indptr, neuron_index.astype(np.int32),
                   s.trace_np.astype(np.int32), traces)

    def run(self, steps: int, drive: Drive | None = None, seed: int = 0,
            trace_neurons: np.ndarray | None = None) -> Run:
        s = self.start(drive, seed, trace_neurons)
        for _ in range(steps):
            self.advance(s)
        return self.finish(s)


def photoreceptor_drive(neurons: np.ndarray, columns: np.ndarray, intensity: np.ndarray,
                        rate_at_full_hz: float = 300.0, steps_per_frame: int = 50) -> Drive:
    """E1's transduction: light as Poisson input to spiking photoreceptors, at ``rate_at_full_hz`` times the
    intensity of each photoreceptor's column (grey, 0.5, gives the published model's 150 Hz activation rate), frame
    by frame. ``neurons`` index the spiking network; ``columns`` are their eye columns; ``intensity`` is
    (frames, columns)."""
    rates = rate_at_full_hz * np.asarray(intensity, dtype=np.float64)[:, np.asarray(columns, dtype=np.int64)]
    return Drive(modulated=(np.asarray(neurons, dtype=np.int64), rates, int(steps_per_frame)))
