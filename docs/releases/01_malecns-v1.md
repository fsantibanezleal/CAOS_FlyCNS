# MaleCNS v1.0

The complete connectome of the male *Drosophila melanogaster* central nervous system, brain and ventral nerve cord:
166,700 proofread neurons and 11,710 cell types (Berg, Beckett, Costa, Schlegel, Januszewski, Marin et al., "Sexual
dimorphism in the complete Drosophila male central nervous system connectome", *Cell* 189(18):5504-5526.e15, 2026,
doi:[10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015)). Its visual system is described in the
companion paper (Hoeller, Zhao, Nern et al., "The organization of visual pathways in the Drosophila brain", *Cell*
189:5552-5570.e10, 2026, doi:[10.1016/j.cell.2026.08.014](https://doi.org/10.1016/j.cell.2026.08.014)), and the male
optic lobe in Nern et al., *Nature* 641:1225-1237 (2025), doi:[10.1038/s41586-025-08746-0](https://doi.org/10.1038/s41586-025-08746-0).
Data: CC BY 4.0, [male-cns.janelia.org/download](https://male-cns.janelia.org/download/).

## The tables flycns reads

| Table | Size | SHA-256 (locked) |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 B | `2177e246...a9a3b2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 B | `95c92892...879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 B | `e35da783...56afc1` |
| `syn-points-male-cns-v1.0-minconf-0.5.feather` | 13,061,489,098 B | `c16b1b63...f8f284` |

The full hashes are in `flycns/release/malecns_v1.py`. The first three were checked against two independent public
lock files kept by other projects that read the same release; the synapse table against the MD5 the release bucket
publishes for it. A table that differs is refused, never used.

## How the tables become a graph

**Which neurons.** The rows that carry a `superclass`: exactly the 166,700 neurons of the paper. Rows without one are
fragments, glia and orphans. Neurons are ordered by body ID, and that order is the neuron index everywhere.

**Where each neuron is.** The release coordinates are 8 nm voxels; flycns stores micrometres. A neuron's position is
its soma (`somaLocation`); failing that, the point where its neurite leaves toward a soma outside the volume
(`tosomaLocation`); failing that, the mean position of all its synapse points. The last case is common for sensory
neurons, whose cell bodies lie outside the CNS (photoreceptors in the retina, mechanosensory and gustatory neurons in
the body), and for some optic-lobe neurons without an annotated soma. Which source was used is stored per neuron, so
a display can draw a soma differently from a centroid. **In the release frame, +x points to the fly's left** (measured:
neurons named `_L` sit at larger x than those named `_R`); a renderer that assumes the opposite shows a mirrored fly.

**What each neuron releases, and with what sign.** The release predicts a transmitter per body from its synapses
(Eckstein et al., *Cell* 187:2574, 2024, doi:[10.1016/j.cell.2024.03.016](https://doi.org/10.1016/j.cell.2024.03.016)).
flycns uses `consensus_nt` (which also draws on the cell type and on ground truth) and falls back to the per-body
`predicted_nt`. The sign follows the published whole-brain model (Shiu et al., *Nature* 634:210-219, 2024,
doi:[10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)): acetylcholine excitatory; GABA and
glutamate inhibitory; dopamine, octopamine and serotonin treated as excitatory, a simplification that paper states.
One addition: **histamine is inhibitory.** It is the photoreceptors' transmitter, acting on histamine-gated chloride
channels in the lamina. FlyWire's vocabulary has no histamine, so a FlyWire model cannot drive vision through its
photoreceptors with the right sign; MaleCNS can. A transmitter that stays unclear gives the neuron no output weight,
and the manifest counts such neurons.

**Which side, which part of the CNS.** The side is the soma side, else the root side, else the `_L`/`_R` suffix of the
instance name, else the side of the midline the neuron's position falls on. Optic-lobe neurons and photoreceptors go to
the optic lobe of their side; nerve-cord superclasses to the nerve cord; everything else, including the visual
projection and centrifugal neurons that link the optic lobes to the central brain, to the central brain.

**Connections.** Every body-to-body weight in the connection table whose two ends are retained neurons, duplicate pairs
summed, stored as a CSR matrix by presynaptic neuron with counts in 16 bits (a count above 65,535 is saturated and
counted).

**The two eyes.** Both optic lobes are mapped column by column: optic-lobe neurons carry two hexagonal coordinates
(`assignedOlHex1`, `assignedOlHex2`). flycns builds one column table per eye from them. Photoreceptor terminals carry
no coordinates, so each is assigned to the column whose neurons receive most of its synapses: the lamina cartridge for
R1-R6, the medulla column for R7 and R8. A column's kind comes from its R7 and R8 subtypes: pale (`p`), yellow (`y`)
or dorsal rim (`d`); disagreements are counted and resolved by majority.

## What the compiled MaleCNS holds

Measured by the compiler on 2026-09-23 (the full set is in the compiled manifest):

| Quantity | Value | Cross-check |
|---|---|---|
| Neurons | 166,700 | the paper's count |
| Connections | 25,582,938 | the same total published by FLYBOARD, DOOMFLY and Xenova's browser simulation |
| Synapses in those connections | 124,177,617 | the same total published by mps-malecns-model |
| Saturated counts / self connections | 0 / 101 | |
| Position from soma / to-soma point / synapse centroid / none | 139,662 / 976 / 25,941 / 121 | |
| Transmitter unclear (no output weight) / histamine | 2,100 / 7,905 | |
| Columns, left / right | 879 / 892 | 892 is the count Nern et al. (2025) report for the right medulla |
| Photoreceptors / joined to a column / distinct columns reached | 6,091 / 5,895 / 1,466 | flyverse reports 5,895 on 1,466 columns by its own method |
| Column kinds: pale / yellow / dorsal rim / unknown (conflicts) | 333 / 487 / 83 / 868 (4) | |

**The photoreceptor gap.** Only 1,466 of the 1,771 columns reach a reconstructed photoreceptor terminal, and 945
columns have no R1-R6 terminal at all (mean 1.89 per column where the animal has 6; the left eye is thinner than the
right). The release reconstructs photoreceptor axons only where they enter the imaged volume, and the lamina is the
least complete neuropil. A simulation that injected light only through reconstructed photoreceptors would leave half
the visual field dark; the dynamics therefore feed every column, through its real photoreceptors where they exist and
through a documented stand-in where they do not, and count both.
