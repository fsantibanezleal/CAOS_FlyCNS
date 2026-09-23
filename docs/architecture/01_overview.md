# Overview: from a connectome release to a nervous system in motion

## The path

```
release files (hash-checked)  ->  compiler  ->  compiled directory  ->  simulation  ->  recording
   MaleCNS v1.0 feathers          flycns.release    manifest + arrays     Python or browser    spikes + graded activity
```

1. **Release files.** The official MaleCNS v1.0 tables (annotations, transmitter predictions, connection weights)
   are downloaded from the release bucket and accepted only if their SHA-256 matches the locked value.
2. **Compiler.** A release adapter turns the tables into one compiled directory: a neuron table (body ID, type,
   class, side, position, sign, partition), the per-eye column tables, and the synapses as a signed CSR matrix.
3. **Compiled directory.** A `manifest.json` plus little-endian binary arrays, the only thing either simulator reads.
   Python and TypeScript read the same bytes, which is what makes parity testable.
4. **Simulation.** The optic lobe runs as graded (non-spiking) neurons, the rest of the CNS as leaky
   integrate-and-fire neurons with the published constants; graded outputs drive spiking neurons through their
   synapse counts.
5. **Recording.** Spikes by tick and graded activity sampled per frame, in a binary format both languages read.

## The two implementations

| | Python reference | Browser engine |
|---|---|---|
| Package | `flycns` | `@fasl-work/flycns` |
| Arithmetic | NumPy (CPU), PyTorch (GPU) | WGSL compute on WebGPU; a worker fallback |
| Accumulation | fixed reduction order | fixed-point integers (no float atomics) |
| Randomness | counter-based hash of (seed, neuron, tick) | the same hash |
| Role | canonical recordings and benchmarks | interactive simulation in a web page |

Parity tolerances between them are stated in the design document, section 7; a browser path that misses them is not
shipped as the live engine of any consumer.
