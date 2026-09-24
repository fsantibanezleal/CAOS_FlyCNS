# The two compound eyes: design

## What is real and what is modelled

| Part | Source | Status |
|---|---|---|
| The set of columns of each eye (879 left, 892 right) and their hexagonal coordinates | MaleCNS v1.0 annotations | real |
| Which photoreceptor terminals belong to which column; pale, yellow, dorsal rim | compiled from the release's connectivity (U1) | real, derived |
| The lattice's orientation (which hex direction points dorsal, which anterior) and its neighbour structure | measured on the release: column centres in the medulla from the per-synapse column labels; neighbours from 3D distance | real, derived |
| The equator row | measured on the release where the lamina allows it (cartridges on the equator receive more R1-R6 terminals, the criterion used by Nern et al. 2025); otherwise the row the lattice centre sits on, flagged | derived, flagged when inferred |
| Each column's viewing direction | a model: the measured lattice laid on the sphere with one inter-ommatidial angle, oriented by the measured axes and scaled to the eye's measured extent (about 10 degrees into the opposite hemisphere in front to about 155 degrees behind, binocular overlap under 20 degrees; Zhao et al., Nature 646:135-142, 2025, doi:10.1038/s41586-025-09276-5) | modelled, labelled |
| Each ommatidium's acceptance | a Gaussian around its direction; its width a parameter with its source | modelled |

The medulla's anterior-posterior axis is mirrored relative to the eye by the first optic chiasm (the lamina to
medulla projection crosses over horizontally); dorsal-ventral is preserved. The model applies that inversion when it
turns medulla geometry into viewing directions, and a test checks it against the lamina where lamina neurons exist.

## Sampling a scene

A scene is given as an equirectangular panorama (luminance, optionally depth and object labels) seen from the head.
For each eye, a sparse sampling matrix `S` (ommatidia x panorama pixels) holds the acceptance weights, normalised per
ommatidium and weighted by each pixel's solid angle; a frame's ommatidial luminance is `S @ panorama`. Ground truth
per ommatidium (depth along the central ray, the object under the central ray, the acceptance-weighted share of each
object) comes from the same matrix applied to the depth and label panoramas, so the truth and the input are sampled
identically.

## Interfaces

- `flycns.eyes.EyeModel.from_compiled(compiled, geometry)`: per-eye tables (column index, hex, kind, direction unit
  vector, azimuth, elevation, neighbour list, equator flag).
- `flycns.eyes.sampling_matrix(eye, width, height, acceptance_deg)`: the sparse `S`.
- `flycns.eyes.sample(eye_matrices, panorama)`: luminance per ommatidium for both eyes.
- The geometry measured from the release is computed once by `flycns.eyes.measure_lattice(synapse_table, compiled)`
  and stored beside the compiled directory as `eyes.json` with its own hash.
