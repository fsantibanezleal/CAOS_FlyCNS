# flycns · software design document

Written 2026-09-23, before any code; revised 2026-09-23 after the spiking model's measurements (U3a): section 3
names the modules as built, section 5 states the published silencing and refractoriness exactly, and section 7
compares whole-CNS runs in the two ways a model with two states allows. `flycns` compiles a fly connectome release into a graph a simulator can run,
models the two compound eyes over the release's own optic-lobe columns, and simulates the whole central nervous
system with the published neuron models, in Python and in the browser, with the two implementations held to each
other by parity tests. Its first consumer is Destello; it is written for any consumer.

## 1. Problem

Every community project that runs a fly connectome re-implements the same four things, usually differently and
rarely checked: reading a release into a signed graph, placing neurons in space, feeding the eyes, and integrating
the neuron model. The published model (Shiu et al., *Nature* 634:210-219, 2024, doi:10.1038/s41586-024-07763-9) is a
Brian2 program; the published visual-system model (Lappalainen et al., *Nature* 2024, doi:10.1038/s41586-024-07939-3)
is a type-level PyTorch network on a 721-column lattice. Neither runs the male CNS through its own two eyes, and
neither runs in a browser. `flycns` does, once, with tests.

## 2. Non-goals

- Not a general neural simulator. Point neurons only: the published leaky integrate-and-fire model and a graded
  (non-spiking) rate model for the optic lobe. No compartments, no conductance-based channels.
- No learning of weights. Synapse counts and signs are data; the package never changes them (null-model generators
  produce NEW graphs and say so).
- One release first: MaleCNS v1.0. The compiler is an adapter per release; others are later additions.
- No rendering. The package produces activity; drawing it is the consumer's job.

## 3. Package layout

| Part | Language | Content |
|---|---|---|
| `flycns.release` | Python | release adapters (MaleCNS v1.0), SHA-256 locked sources |
| `flycns.compiled` | Python | the compiled format: writer and hash-checked reader of the neuron table, signed CSR and partitions |
| `flycns.nulls` | Python | null-model generators: degree-preserving rewiring, size-matched random graphs, sign shuffles |
| `flycns.rng` | Python | the counter-based generator (MurmurHash3_x86_32) shared with TypeScript and WGSL |
| `flycns.flyvis` | Python + data | the trained numbers of flyvis's 50 pretrained networks, in flyvis's order, with provenance |
| `flycns.eyes` | Python | per-eye column tables, modelled viewing directions (the vertical set by the dorsal rim), ommatidium sampling of a scene |
| `flycns.optic_lobe` | Python | both MaleCNS optic lobes as graded units with flyvis's numbers transferred, stand-in photoreceptors, CT1 compartments |
| `flycns.motion` | Python | moving edges on the eyes, and flyvis's direction-selectivity measures |
| `flycns.dynamics.hybrid` | Python | the whole CNS coupled: graded optic lobes and the LIF through the bridge and the feedback (E2, E4); flyvis's lattices mapped (E3) |
| `flycns.mapped` | Python | E3's geometry: where flyvis's lattice columns look, and which MaleCNS neurons they stand for |
| `flycns.dynamics` | Python (NumPy reference, PyTorch GPU) | LIF (Shiu), graded optic lobe (flyvis-style), the graded-to-spiking bridge, stabilisers, stimulation and silencing |
| `flycns.record` | Python | spike and graded-activity recordings in the shared binary format |
| `@fasl-work/flycns` | TypeScript + WGSL (+ WASM fallback) | loaders for the compiled format and recordings; the same dynamics on WebGPU; a worker-based fallback |
| `parity/` | both | fixed-seed scenarios run by both implementations, compared by the tolerances in section 7 |

## 4. The compiled format (the contract between the two languages)

A directory with a `manifest.json` (schema version, release, source hashes, array names, dtypes, shapes, byte
offsets, SHA-256 of every array file) and little-endian binary arrays:

| Array | Type | Content |
|---|---|---|
| `neuron_body_id` | int64 | release body ID |
| `neuron_type`, `neuron_class`, `neuron_superclass` | uint32 indices into string tables | annotations |
| `neuron_side` | uint8 | left, right, midline, unknown |
| `neuron_position` | float32 x 3 (micrometres) | soma, else skeleton root, else synapse centroid; the source per neuron is recorded |
| `neuron_sign` | int8 | +1, -1, 0 from the transmitter rule (section 5) |
| `neuron_partition` | uint8 | optic lobe left/right, central brain, nerve cord, sensory, motor |
| `column_eye`, `column_hex`, `column_bodies` | per-eye tables | the release's column assignments and the photoreceptor and lamina bodies of each column |
| `csr_indptr`, `csr_indices`, `csr_count` | int64, int32, uint16 | synapses by presynaptic neuron; counts saturate at 65,535 and the number saturated is recorded |

Everything a simulation needs is in the directory; a consumer never reads the release files.

## 5. Model definitions (each constant cited in the code and the docs)

- **Transmitter to sign.** Acetylcholine +1; GABA and glutamate -1; histamine -1 (photoreceptors; MaleCNS predicts
  histamine, FlyWire does not); dopamine, octopamine and serotonin +1 as in Shiu et al. (a documented simplification);
  unclear 0 (the neuron's outputs carry no weight and the count of such neurons and synapses is reported).
- **LIF (Shiu et al. 2024, `model.py`):** `dv/dt = (v0 - v + g)/t_mbr`, `dg/dt = -g/tau`, threshold `v > v_th`, reset
  `v = v_rst, g = 0`, refractory `t_rfc`, a presynaptic spike adds `w = sign x count x w_syn` to `g` after `t_dly`.
  v0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms, t_rfc = 2.2 ms, t_dly = 1.8 ms, w_syn = 0.275 mV,
  dt = 0.1 ms. Activation is Poisson input at a rate with weight `w_syn x f_poi` (f_poi = 250), and activated
  neurons have no refractory period. Silencing zeroes a neuron's outgoing synapses, as the published `silence()`
  does; the neuron still receives input and spikes. Input reaching a refractory neuron is discarded, as Brian2 does
  for variables declared `(unless refractory)`.
- **Graded optic lobe (Lappalainen et al. 2024 form):** non-spiking leaky voltage per neuron with a per-type time
  constant and resting potential, presynaptic output rectified, synaptic weight = sign x count x unitary strength per
  type pair, parameters transferred from the published ensemble; types outside the ensemble receive documented
  defaults and are counted. The engines compute flyvis 1.2.0's `PPNeuronIGRSynapses` exactly (forward Euler,
  `tau_eff = max(tau, dt)`), which is checked against flyvis running its own network (section 7). The transfer
  onto MaleCNS keeps every synapse and caps each neuron's drive from each presynaptic class at flyvis's (without
  the cap the release's denser lateral wiring diverges within 100 ms), splits CT1 into flyvis's per-column
  compartments, and fills columns the release left without photoreceptors with flagged stand-ins
  (`docs/design/features/graded/design.md`, part 2).
- **Graded to spiking.** A spiking neuron's `g` receives the rectified output of its graded presynaptic partners
  through their synapse counts, with one bridge gain documented as a free parameter (as every hybrid in the survey
  has one).
- **Stabilisers (optional, off in the published-model mode):** spike-frequency adaptation, a per-connection
  saturation cap, fan-in normalisation. Each carries its source and is reported when on.
- **Randomness is counter-based.** Poisson inputs draw from MurmurHash3_x86_32 of the key (neuron, tick) seeded with
  the run's seed, identical in Python, WGSL and the fallback, so the same seed gives the same input spikes
  everywhere.

## 6. Determinism

Accumulation of synaptic input uses fixed-point integers in the browser kernels (no float atomics), and a fixed
reduction order in the Python reference, so a run is a pure function of (graph, stimulus, seed, parameters) on the
CPU paths. GPU paths are compared by tolerance (section 7), never claimed bit-identical.

## 7. Parity policy

| Pair | Tolerance |
|---|---|
| Python reference vs a literal Brian2 transcription of `model.py`, small circuits | identical spike times and neuron indices |
| Graded engines vs flyvis 1.2.0 running its own network 000 (45,669 neurons, 200 steps) | every neuron at every step within 1e-5 (measured 2.4e-6) |
| Hybrid GPU vs NumPy, small CNS | identical spikes; graded activity within 1e-4 |
| Direction-selectivity measures vs flyvis's own, on its lattice, networks 000 to 004 | DSI within 1e-5; preferred direction within 0.01 degrees wherever flyvis's DSI exceeds 0.01 |
| NumPy reference vs PyTorch GPU, whole CNS | moderate drive, 200 ms: active-neuron Jaccard >= 0.99 and per-neuron count correlation >= 0.99; strong drive, 500 ms: five GPU trials against five independent reference trials correlate at least at the 5th percentile of the reference against itself over the 126 splits of ten seeds |
| Python reference vs the TypeScript fallback, fixed seeds | identical spike trains on the parity circuits; whole CNS as the GPU row |
| Python reference vs WebGPU, whole CNS | as the GPU row, with Jaccard and correlation >= 0.98 in the moderate window |

A browser path that fails its tolerance is not shipped as the live engine.

Why two regimes (measured 2026-09-23, `docs/models/02_lif.md`): on the whole MaleCNS the published model has two
states, and a run switches from the low one (about 8,000 spikes per 50 ms under the moderate drive) into the high one
(about 44,000) at a random moment. Float32 and float64 runs share every input event but part at the first threshold
the two arithmetics decide differently, so beyond that point they are compared as samples of one process, never spike
by spike.

## 8. Risks

| Risk | Handling |
|---|---|
| Float nondeterminism on GPUs | fixed-point accumulation in the browser; tolerances stated; no bit-identity claims on GPU |
| Memory: 25.6 M synapses in a browser | uint16 counts and int32 targets (about 150 MB raw), sharded; partition-first loading |
| flyvis parameter transfer covers only its 64 types | coverage reported per type; defaults documented |
| Publishing needs account actions | PyPI trusted publisher and an npm token requested at release; consumers pin a git tag meanwhile |

## 9. Requirements in force at scaffolding

```
R-001  THE repository SHALL contain no em-dash and no emoji in any tracked text file.
       Gate: scripts/check_content_standards.py

R-002  THE continuous-integration workflows SHALL run only cheap checks, trigger only on develop, main
       and manual dispatch, carry a concurrency group, and give every job a timeout.
       Gate: scripts/check_ci_budget.py

R-003  THE repository SHALL keep a design document in which every requirement names a gate that exists.
       Gate: scripts/check_sdd.py
```
