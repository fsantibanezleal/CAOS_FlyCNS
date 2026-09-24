# The graded optic lobe (U3b): design

Written 2026-09-23 before the code of this unit. U3b has three parts, released in order: **the graded engine and the
flyvis ensemble** (this document's first half, 0.04.000); **the transfer onto the MaleCNS optic lobes** (0.05.000);
**the coupling to the spiking model and the stabilisers** (0.06.000). Each part adds its requirements to
`requirements.md` before its code.

## Why a graded optic lobe

Most neurons of the lamina and medulla do not fire spikes: they signal with graded changes of membrane potential and
release transmitter continuously. A whole-CNS model in which they spike does not carry vision past the lamina (the
flyverse project's audits, dossier 01 section 2.14 in the plan). The published model of the fly visual system that
predicts neural activity from the connectome, flyvis (Lappalainen et al., *Nature* 2024,
doi:10.1038/s41586-024-07939-3; code MIT), treats every visual neuron as a passive graded point neuron, and was
trained end to end on optic flow; ON and OFF pathways and T4/T5 direction selectivity emerge in it and match
measurements across 26 studies. flycns uses the same dynamics and flyvis's trained numbers.

## The dynamics, as flyvis 1.2.0 computes them

`PPNeuronIGRSynapses` (passive point neurons, instantaneous graded release):

$$\tau_i^\text{eff}\,\frac{dV_i}{dt} = -V_i + b_i + \sum_j w_{ij}\,\max(V_j, 0) + x_i(t), \qquad
\tau_i^\text{eff} = \max(\tau_i, \Delta t),$$

integrated by forward Euler, $V \leftarrow V + \Delta t\,\dot V$. $b_i$ is the resting potential of the neuron's
type, $\tau_i$ its time constant, $x_i$ the input (the stimulus intensity, added to the photoreceptors R1 to R8 of the
neuron's column, the same value for all eight), and

$$w_{ij} = \sigma_{t_j t_i}\; N_{t_j t_i}(\Delta u, \Delta v)\; \alpha_{t_j t_i},$$

with $\sigma$ the sign of the pair of types (fixed, from the literature), $N$ the mean synapse count of the pair at
the columnar offset between the two neurons (fixed, from the FIB-25 and FIB-19 reconstructions), and $\alpha$ the
unitary strength of the pair of types (trained, non-negative). The state starts at the resting potentials and is
brought to its steady state by two seconds of uniform grey (intensity 0.5). flyvis trains at $\Delta t$ = 1/50 s
and evaluates at 1/200 s.

**The ensemble, measured.** 50 networks, each with 65 resting potentials, 65 time constants and 604 strengths (734
trained numbers); signs (376 excitatory, 228 inhibitory pairs) and the 2,355 mean counts are identical in all 50.
73.7% of the trained time constants lie below the training step, where they acted as 20 ms; the slowest types are C2
(median 239 ms), Tm2 (136 ms) and C3 (67 ms). Resting potentials range from -1.41 to 2.63 (median 0.53). The trained
strength times the pair's mean count has a median of 0.036 (10th to 90th percentile 0 to 0.29); 3% of strengths sit
at zero. The same pair's strength varies widely across networks (median coefficient of variation 1.20), so results
are reported over the ensemble, never from one network alone.

## Two engines, and flyvis itself as the oracle

| Engine | Arithmetic | Input to each neuron |
|---|---|---|
| `GradedReference` (NumPy) | float64 | `bincount` over the connections, by target, in a fixed order |
| `GradedTorch` (PyTorch, CUDA or CPU) | float32 | `index_add_` over the connections |

The oracle is flyvis running its own network: `scripts/extract_flyvis_ensemble.py`, run in a separate environment
with flyvis 1.2.0, writes network 000 in full (45,669 neurons, 1,513,231 connections with their weights, the input
index of the 721 columns) and what flyvis computes on it for a fixed stimulus (two seconds of grey to the steady
state; then a full-field flash from 0.5 to 1.0 between 100 and 300 ms; then an ON edge sweeping the lattice at one
column per 20 ms from 400 ms), every neuron at every 5 ms step. The engines must reproduce that recording.

## The ensemble as data

The extraction also writes the 50 networks' parameters in flyvis's own order (types, pairs of types and count groups
exactly as flyvis's parameter objects list them, checked against the values the network uses) with the SHA-256 of
every source checkpoint and flyvis's MIT notice. The 225 KB directory ships inside the package
(`flycns/data/flyvis-1.2.0-ensemble/`), so the transfer onto MaleCNS needs neither flyvis nor the checkpoints.
