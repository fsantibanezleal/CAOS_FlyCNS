# Changelog

All notable changes are recorded here, newest first, grouped Added / Changed / Fixed / Removed. Versions are
`X.XX.XXX` (the `VERSION` file, the tags and this log); the manifests carry the semantic form (`0.0.0`).

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
