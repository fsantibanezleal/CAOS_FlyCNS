# The browser engine (U3c): design

Written 2026-09-25, before the code of this unit. The Python engines (U3a, U3b) are complete and measured; this unit
carries the same dynamics into the browser, in TypeScript on the CPU and in WGSL on WebGPU, and holds both to the
Python reference by parity fixtures the reference itself writes.

## What this unit produces

`@fasl-work/flycns`, the npm package, with:

- loaders for the compiled format, for recordings and for engine bundles, each array checked against the SHA-256
  in its manifest before use;
- the counter-based generator (MurmurHash3_x86_32) in TypeScript and in WGSL, pinned to the Python one;
- a CPU engine in float64 that reproduces the NumPy reference exactly: the published LIF, the graded optic lobe, the
  hybrid with either source (the lobe's own dynamics, E2 and E4; flyvis's lattices carried onto the lobe, E3);
- a WebGPU engine in float32 with fixed-point synaptic accumulation, compared by tolerance;
- a worker protocol, so a page runs the CPU engine off its main thread when there is no WebGPU;
- the parity fixtures under `parity/`, produced by the Python reference and checked current by a Python test, run by
  the TypeScript tests on every push.

Rendering is not here (SDD section 2): the engine produces spikes and voltages; Destello draws them.

## The engine bundle: the contract between the two languages

The compiled graph is the neuron table and the raw synapses; what an engine runs is derived from it (the weights,
the optic-lobe units, the bridge). Rather than repeat that derivation in TypeScript, Python writes an **engine
bundle**: a directory in the compiled style (`flycns.compiled.write_compiled`, schema `flycns.engine/1`) holding the
arrays an engine needs and, in the manifest, every constant the step uses, as JSON numbers (a float written by
Python's `json` and read by `JSON.parse` round-trips exactly).

| Part | Arrays | Manifest constants |
|---|---|---|
| LIF | `lif_indptr` (int64), `lif_indices` (int32), `lif_weight_mv` (float64 for an exact bundle, float32 for a large one) | the published constants, `delay_steps`, `refractory_steps`, the decay factors `a`, `b`, `c` of one step, the adaptation decay |
| graded | `graded_bias`, `graded_time_const_s`, `graded_weight` (float64), `graded_source`, `graded_target`, `graded_input_neuron`, `graded_input_column` (int32) | `dt_s`, `n_columns` |
| hybrid | both, plus `bridge_source`, `bridge_target`, `bridge_weight`, `feedback_source`, `feedback_target`, `feedback_weight` | `bridge_gain_hz`, `steps_per_graded`, `dt_lif_s`, `rate_decay`, `rate_amount` |
| lattice source (E3) | two lattice networks (`lattice0_*`, `lattice1_*`) and the map (`map_unit`, `map_node`, `map_side`, `map_weight`) | `n_lattice_columns` |

The exact bundles carry float64, so the compiled format now allows `float64` (the graph itself never uses it; the
reader and both loaders accept it). A **scenario** is an engine bundle plus a run: the drive (activated neurons and
rates, fixed events, silenced neurons, modulated rates), the stimulus (frames of column intensity), the seed, the
lead-in and the traced neurons (schema `flycns.scenario/1`). Its `expected/` directory is the reference's output: a
recording (`flycns.recording/1`) and, for graded and hybrid runs, the recorded graded activity and the grey release.

## The CPU engine: the reference, operation for operation

The NumPy reference is a pure function of (bundle, scenario) in float64 with a fixed reduction order, so a TypeScript
engine that performs the same double-precision operations in the same order gives the same bits. The rules that make
this true, each checked by a fixture:

- deliveries are accumulated into a zeroed array in the reference's order (spiking neurons ascending, each one's
  connections in CSR order) and only then added to `g`; `np.bincount` sums its weights in input order;
- every constant that the reference computes with `math.exp` or a division (`a`, `b`, `c`, the rate decay, the
  steps of the lead-in) is read from the manifest, never recomputed: `Math.exp` and `math.exp` may differ by one
  unit in the last place;
- Poisson thresholds follow the reference's two expressions exactly (`rate * (dt_ms / 1000)` for activation,
  `(rate * dt_ms) / 1000` for modulated rates), because the two roundings differ;
- the graded step is `v + rate * (((-v + bias) + syn) + current) * dt` with the sums in that order;
- traces are rounded to float32 as the reference stores them.

The engine exposes the reference's shape: `start`, `advance(gInput?)`, `finish`, and `run`; the hybrid runs frames.

## The WebGPU engine

Storage buffers per neuron (`v`, `g`, `last`, refractory steps, mute, thresholds, adaptation, rate) and per
connection (CSR targets and weights), and a ring of `delay_steps + 1` spike-flag slots for the delay. One LIF step is
five dispatches in one compute pass (each dispatch sees the previous one's writes):

1. **integrate**: per neuron, the exact integration when free, the threshold, the spike flag, the ring slot
   (a silenced neuron's flag is not written to the ring: its spikes reach no one);
2. **deliver**: per neuron that spiked `delay_steps` ago, its CSR row is pushed with `atomicAdd` on **int32
   fixed-point** accumulators, the weights pre-quantised on the CPU at a scale chosen from the largest possible sum
   into any neuron (every presynaptic neuron firing at once), so the sum cannot overflow and, being an integer sum,
   is the same whatever the order of the threads;
3. **apply**: per neuron, the delivery and the graded input enter `g` when free; the counter-based events (the WGSL
   hash) and the modulated ones enter `v` when free;
4. **events**: per fixed event of this step (rare; the parity circuits use them), `v` gets its increment when free;
5. **reset and record**: resets, adaptation, the filtered rate, the spike appended to the batch's list, the traced
   voltages written.

A batch of steps (one graded frame, 50 steps) is one command buffer; the spike list and traces are read back once per
batch and sorted by (step, neuron) on the CPU. The graded step is two dispatches (per-connection accumulation of
`weight x max(V, 0)` into fixed point, then the per-neuron update); the bridge and the feedback are one per-connection
accumulation each plus a per-neuron conversion. GPU results are compared by tolerance (SDD section 7), never claimed
bit-identical to the reference, though the fixed-point sums make each kernel deterministic given its inputs.

Node runs the same kernels through Dawn (`webgpu` on npm, installed only locally with `--no-save`; it is 95 MB and
CI has no GPU), so the GPU parity tests run on the developer's machine before a release and are listed as not run in
CI, never as passed.

## Memory on the whole CNS

The LIF part of MaleCNS: 166,700 neurons and 25.6 M connections. Targets as int32 (102 MB) and weights as float32
(102 MB) fit a browser but not comfortably; a large bundle therefore carries `lif_weight_mv` as float32 and the page
loads it partition-first. The optic-lobe part: 96 k units, 9.07 M connections, float32 weights (36 MB). The measured
payload decides Destello's deploy (its plan, U9); it is not decided here.

## Parity tolerances (from the SDD, section 7)

| Pair | Tolerance | Fixture |
|---|---|---|
| reference vs TypeScript CPU, LIF | identical spike trains and float32 traces | three Brian2-checked circuits with fixed events; one with Poisson activation, modulated rates, silencing and traces; one with adaptation |
| reference vs TypeScript CPU, graded | identical activity | a random graded network over random frames |
| reference vs TypeScript CPU, hybrid | identical spikes and graded activity | the toy CNS of the hybrid tests, own source and lattice source |
| reference vs WebGPU, LIF and hybrid, small | identical spikes; graded activity within 1e-4 | the same fixtures |
| reference vs WebGPU, whole CNS | active-neuron Jaccard and count correlation at least 0.98 over 200 ms of the moderate drive | measured with the compiled MaleCNS, before a release that touches the kernels; recorded in `docs/models/06_browser.md` |
