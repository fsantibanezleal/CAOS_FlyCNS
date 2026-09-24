# Changelog

All notable changes are recorded here, newest first, grouped Added / Changed / Fixed / Removed. Versions are
`X.XX.XXX` (the `VERSION` file, the tags and this log); the manifests carry the semantic form (`0.0.0`).

## [0.04.000] - 2026-09-24

### Added

- The graded engines (`flycns.dynamics.GradedReference`, NumPy float64; `GradedTorch`, PyTorch float32): flyvis's
  `PPNeuronIGRSynapses` dynamics (Lappalainen et al., *Nature* 2024), forward Euler with `tau_eff = max(tau, dt)`.
  On flyvis 1.2.0's own network 000 (45,669 neurons, 1,513,231 connections) and a fixed stimulus they reproduce
  flyvis's run, every neuron at every step, to 2.4e-6 and 1.4e-6.
- The trained numbers of flyvis's 50 pretrained networks, shipped as package data (`flycns.flyvis.load_ensemble`):
  65 resting potentials and time constants and 604 strengths per network, the fixed signs and 2,355 mean counts, in
  flyvis's own order, with the SHA-256 of every source checkpoint and flyvis's MIT notice.
- `scripts/extract_flyvis_ensemble.py`: the extraction, run in a separate environment with flyvis 1.2.0, which also
  records network 000 in full and flyvis's run on it (the parity target).
- Wiki: the graded visual neurons, with the equations, a diagram, the ensemble's measured spread and the parity
  numbers.

### Fixed

- The README still described 0.02.000; it now lists what 0.03.000 and 0.04.000 added, and how to run a simulation.
- Manifests written on Windows had CRLF line endings, so the same inputs gave different bytes on Windows and Linux;
  `write_compiled` now writes LF everywhere.

## [0.03.000] - 2026-09-23

### Added

- The published whole-brain spiking model (`flycns.dynamics`): the leaky integrate-and-fire neuron of Shiu et al.
  (*Nature* 634:210-219, 2024) with its published constants, integrated exactly, in Brian2's step order, on a NumPy
  reference engine (float64) and a PyTorch engine (float32, GPU); Poisson activation with the published weight and no
  refractory period for activated neurons; the published silencing (outgoing synapses only), through the weights or
  through the drive.
- A counter-based generator (`flycns.rng`): MurmurHash3_x86_32 of (neuron, step) seeded by the run's seed, so every
  implementation reproduces the same input events; checked against the independent `mmh3` package and pinned by
  five vectors for the TypeScript and WGSL implementations.
- Null graphs (`flycns.nulls`): degree-preserving rewiring within partition blocks, with a collision repair that
  never creates a collision (all 306,510 collisions repaired on MaleCNS v1.0); size-matched random graphs per block;
  sign shuffles.
- Recordings (`flycns.record`): spikes as a CSR over steps plus chosen voltage traces, hash-checked.
- Tests: identical spike trains against a literal Brian2 transcription of the published program on three circuits
  (run in continuous integration, which now installs Brian2); activation, the generator, silencing, recordings and
  the null graphs on small graphs; on the whole MaleCNS, the GPU engine against the reference and the exact
  degree-preserving null.
- Wiki: the spiking model page, with the equations, the step diagram, what the model does on the whole male CNS
  (two states, the switch carried by the mushroom-body loop), the agreement criteria and the null graphs.

### Changed

- The design document names the modules as built, states the published silencing and refractoriness exactly, and
  compares whole-CNS runs in the two ways a model with two states allows.

### Fixed

- The date of 0.02.000, which was written in UTC; releases are dated in local time.

## [0.02.000] - 2026-09-23

### Added

- The two compound eyes (`flycns.eyes`): the lattice's orientation measured on the release (medulla column centres
  from the per-synapse column labels, body axes from landmark neuropils, neighbour offsets from 3D distance); the
  anterior-posterior mirror of the first optic chiasm; an ideal hexagonal lattice placed on the sphere and scaled to
  the eye's measured extent (10 degrees into the opposite hemisphere to 155 degrees behind, Zhao et al. 2025); an
  acceptance-weighted sampler of equirectangular panoramas (Gaussian, 8.23 degrees full width at half maximum,
  Gonzalez-Bellido et al. 2011).
- Tests on a synthetic lattice of known orientation (axes, neighbours, chiasm, extent, spacing, mirror symmetry,
  sampling normalisation, a bright spot) and on the real release (879 and 892 columns; dorsal-rim columns above the
  colour columns in both eyes, an independent check of the vertical orientation).
- Wiki: the eye model with its equations, sources, measured numbers and limits.

## [0.01.001] - 2026-09-23

### Fixed

- Continuous integration installed the package without the `release` extra, so the tests could not import pandas to
  write the synthetic release, and CI failed on `develop` and `main` after 0.01.000. It now installs `.[dev,release]`.

## [0.01.000] - 2026-09-23

### Added

- The MaleCNS v1.0 compiler (`flycns.release.compile_malecns_v1`): the four official tables accepted only on their
  locked SHA-256 (three checked against two independent public lock files, the synapse table against the bucket's
  MD5); the 166,700 neurons with a superclass, ordered by body ID; positions from the soma, else the to-soma point,
  else the synapse centroid, with the source recorded; transmitters from the consensus prediction, else the per-body
  prediction; signs by the published whole-brain rule with histamine inhibitory (the photoreceptors); side and
  partition (optic lobe per side, central brain, nerve cord); every connection between retained neurons as a CSR
  matrix with 16-bit counts and saturations counted; both eyes' column tables from the hex assignments, each
  photoreceptor joined to the column it drives most, and pale, yellow or dorsal-rim column kinds from the R7/R8
  subtypes.
- The compiled directory (`flycns.compiled`): a manifest and little-endian arrays, every array hashed and verified on
  read, byte-identical across compilations.
- Tests: a synthetic MaleCNS-shaped release exercising every rule (hash refusal, retention, edge accumulation and
  saturation, position fallback, sign rule, sides and partitions, photoreceptor-to-column assignment, column kinds,
  tamper refusal, determinism) and a data test pinning the compiled release to the published counts.
- Wiki: the compiled format, and the MaleCNS v1.0 page (tables, hashes, every rule with its source, the coordinate
  frame's handedness).

## [0.00.000] - 2026-09-23

### Added

- The software design document (`docs/design/SDD.md`), written before any code: the compiled format shared by
  Python and TypeScript, the model definitions with their sources, determinism and the parity policy.
- The repository scaffold: the Python package `flycns` (src layout) and the npm package `@fasl-work/flycns`
  (TypeScript), each with a version test that checks the three version sources agree; guards for content
  standards, the CI budget and the design document; CI on `develop` and `main`; the PyPI trusted-publishing
  workflow.
