"""Graded (non-spiking) point neurons, as flyvis computes them (Lappalainen et al., Nature 2024).

``PPNeuronIGRSynapses`` in flyvis 1.2.0: passive point neurons with instantaneous graded release,

    tau_eff dV/dt = -V + b + sum_j w_ij max(V_j, 0) + x,    tau_eff = max(tau, dt),

integrated by forward Euler. ``GradedReference`` (NumPy, float64) and ``GradedTorch`` (PyTorch, float32) take the
same network and give the same run; both are held to flyvis's own recording of its network 000 (see
``docs/design/features/graded/design.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GradedNetwork:
    """A network of graded neurons.

    Light enters through ``input_neuron``: each of those neurons receives the intensity of the column
    ``input_column`` names (flyvis adds the same intensity to R1 to R8 of a column; on MaleCNS a column holds as many
    photoreceptors as the release reconstructed there, plus stand-ins).
    """

    bias: np.ndarray            # (n,) resting potential of each neuron
    time_const_s: np.ndarray    # (n,) time constant of each neuron, seconds
    source: np.ndarray          # (e,) presynaptic neuron of each connection
    target: np.ndarray          # (e,) postsynaptic neuron of each connection
    weight: np.ndarray          # (e,) sign x synapse count x unitary strength
    input_neuron: np.ndarray    # (k,) the neurons that receive light
    input_column: np.ndarray    # (k,) the column each of them sees
    n_columns: int

    @classmethod
    def from_input_index(cls, bias, time_const_s, source, target, weight, input_index) -> GradedNetwork:
        """flyvis's layout: ``input_index`` has one row per input type and one column per eye column."""
        input_index = np.asarray(input_index)
        types, columns = input_index.shape
        return cls(bias=bias, time_const_s=time_const_s, source=source, target=target, weight=weight,
                   input_neuron=input_index.reshape(-1), input_column=np.tile(np.arange(columns), types),
                   n_columns=columns)

    @property
    def n(self) -> int:
        return len(self.bias)

    def column_current(self, intensity: np.ndarray) -> np.ndarray:
        """Per-neuron input for one frame of per-column intensities (zero off the input neurons)."""
        current = np.zeros(self.n)
        np.add.at(current, np.asarray(self.input_neuron, dtype=np.int64),
                  np.asarray(intensity, dtype=np.float64)[np.asarray(self.input_column, dtype=np.int64)])
        return current


class GradedReference:
    """The NumPy engine: float64 state; each neuron's input summed with ``bincount`` in connection order."""

    def __init__(self, network: GradedNetwork, dt_s: float):
        self.net = network
        self.dt = float(dt_s)
        self.bias = np.asarray(network.bias, dtype=np.float64)
        self.rate = 1.0 / np.maximum(np.asarray(network.time_const_s, dtype=np.float64), self.dt)
        self.source = np.asarray(network.source, dtype=np.int64)
        self.target = np.asarray(network.target, dtype=np.int64)
        self.weight = np.asarray(network.weight, dtype=np.float64)

    def column_current(self, intensity: np.ndarray) -> np.ndarray:
        return self.net.column_current(intensity)

    def synaptic_input(self, v: np.ndarray) -> np.ndarray:
        release = np.maximum(v[self.source], 0.0)
        return np.bincount(self.target, weights=self.weight * release, minlength=self.net.n)

    def step(self, v: np.ndarray, current: np.ndarray) -> np.ndarray:
        """One Euler step with a per-neuron input ``current``."""
        velocity = self.rate * (-v + self.bias + self.synaptic_input(v) + current)
        return v + velocity * self.dt

    def steady_state(self, t_pre_s: float, grey: float = 0.5, initial: np.ndarray | None = None) -> np.ndarray:
        """The state after ``t_pre_s`` of uniform intensity ``grey``, from the resting potentials unless given."""
        v = self.bias.copy() if initial is None else np.asarray(initial, dtype=np.float64).copy()
        current = self.net.column_current(np.full(self.net.n_columns, grey))
        for _ in range(int(t_pre_s / self.dt)):
            v = self.step(v, current)
        return v

    def run(self, intensity: np.ndarray, initial: np.ndarray | None = None,
            record: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Step through frames of per-column intensity (frames x columns); returns (final state, activity), the
        activity after each step for the neurons in ``record`` (all when omitted), as float32."""
        v = self.bias.copy() if initial is None else np.asarray(initial, dtype=np.float64).copy()
        record = np.arange(self.net.n) if record is None else np.asarray(record)
        out = np.zeros((len(intensity), len(record)), dtype=np.float32)
        for k, frame in enumerate(np.asarray(intensity)):
            v = self.step(v, self.net.column_current(frame))
            out[k] = v[record]
        return v, out


class GradedTorch:
    """The same dynamics in PyTorch (float32), for the GPU."""

    def __init__(self, network: GradedNetwork, dt_s: float, device: str | None = None):
        import torch

        self.torch = torch
        self.net = network
        self.dt = float(dt_s)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        as_f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)  # noqa: E731
        as_i64 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)    # noqa: E731
        self.bias = as_f32(network.bias)
        self.rate = 1.0 / torch.clamp(as_f32(network.time_const_s), min=self.dt)
        self.source = as_i64(network.source)
        self.target = as_i64(network.target)
        self.weight = as_f32(network.weight)
        self.input_neuron = as_i64(network.input_neuron)
        self.input_column = as_i64(network.input_column)

    def column_current(self, intensity):
        torch = self.torch
        current = torch.zeros(self.net.n, device=self.device)
        frame = torch.as_tensor(np.asarray(intensity, dtype=np.float32), device=self.device)
        return current.index_add_(0, self.input_neuron, frame[self.input_column])

    def step(self, v, current):
        synaptic = self.torch.zeros_like(v).index_add_(0, self.target, self.weight * self.torch.relu(v[self.source]))
        return v + self.rate * (-v + self.bias + synaptic + current) * self.dt

    def steady_state(self, t_pre_s: float, grey: float = 0.5, initial=None):
        torch = self.torch
        v = self.bias.clone() if initial is None else torch.as_tensor(np.asarray(initial, dtype=np.float32),
                                                                       device=self.device).clone()
        current = self.column_current(np.full(self.net.n_columns, grey))
        for _ in range(int(t_pre_s / self.dt)):
            v = self.step(v, current)
        return v

    def run(self, intensity: np.ndarray, initial=None, record: np.ndarray | None = None):
        torch = self.torch
        v = self.bias.clone() if initial is None else torch.as_tensor(np.asarray(initial, dtype=np.float32),
                                                                       device=self.device).clone()
        record_t = torch.as_tensor(np.arange(self.net.n) if record is None else np.asarray(record),
                                   device=self.device)
        out = torch.zeros((len(intensity), len(record_t)), device=self.device)
        for k, frame in enumerate(np.asarray(intensity)):
            v = self.step(v, self.column_current(frame))
            out[k] = v[record_t]
        return v.cpu().numpy(), out.cpu().numpy()

    def run_batch(self, intensity: np.ndarray, initial, record: np.ndarray) -> np.ndarray:
        """Several stimuli at once: ``intensity`` is (stimuli, frames, columns), every stimulus starts from
        ``initial``; returns the recorded neurons after each step, (stimuli, frames, recorded), as float32. Each
        stimulus is the same computation as ``run``; batching only shares the kernel launches."""
        torch = self.torch
        stimuli = torch.as_tensor(np.asarray(intensity, dtype=np.float32), device=self.device)
        batch, frames, _ = stimuli.shape
        matrix = self._matrix()
        v = torch.as_tensor(np.asarray(initial, dtype=np.float32), device=self.device)[:, None].repeat(1, batch)
        record_t = torch.as_tensor(np.asarray(record, dtype=np.int64), device=self.device)
        out = torch.zeros((batch, frames, len(record_t)), device=self.device)
        rate, bias = self.rate[:, None], self.bias[:, None]
        for k in range(frames):                                   # state is (neurons, stimuli)
            current = torch.zeros_like(v).index_add_(0, self.input_neuron, stimuli[:, k, self.input_column].T)
            synaptic = matrix @ torch.relu(v)
            v = v + rate * (-v + bias + synaptic + current) * self.dt
            out[:, k] = v[record_t].T
        return out.cpu().numpy()

    def _matrix(self):
        """The weights as a sparse (target x source) CSR matrix, built once: batches multiply by it instead of
        materialising one release per connection and stimulus."""
        if getattr(self, "_csr", None) is None:
            torch = self.torch
            order = torch.argsort(self.target * self.net.n + self.source)
            rows = torch.bincount(self.target[order], minlength=self.net.n)
            crow = torch.zeros(self.net.n + 1, dtype=torch.int64, device=self.device)
            crow[1:] = torch.cumsum(rows, 0)
            self._csr = torch.sparse_csr_tensor(crow, self.source[order], self.weight[order],
                                                size=(self.net.n, self.net.n))
        return self._csr
