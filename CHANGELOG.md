# Changelog

All notable changes are recorded here, newest first, grouped Added / Changed / Fixed / Removed. Versions are
`X.XX.XXX` (the `VERSION` file, the tags and this log); the manifests carry the semantic form (`0.0.0`).

## [0.00.000] - 2026-09-23

### Added

- The software design document (`docs/design/SDD.md`), written before any code: the compiled format shared by
  Python and TypeScript, the model definitions with their sources, determinism and the parity policy.
- The repository scaffold: the Python package `flycns` (src layout) and the npm package `@fasl-work/flycns`
  (TypeScript), each with a version test that checks the three version sources agree; guards for content
  standards, the CI budget and the design document; CI on `develop` and `main`; the PyPI trusted-publishing
  workflow.
