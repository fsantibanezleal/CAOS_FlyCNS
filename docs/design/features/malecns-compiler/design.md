# MaleCNS v1.0 compiler: design

## Inputs (all CC BY 4.0, from `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`)

| Table | Columns used |
|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | bodyId, superclass, class, type, instance, somaSide, rootSide, somaLocation, tosomaLocation, assignedOlHex1, assignedOlHex2 |
| `body-neurotransmitters-male-cns-v1.0.feather` | body, consensus_nt, predicted_nt |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | body_pre, body_post, weight (about 151 M segment pairs, read in record batches) |
| `syn-points-male-cns-v1.0-minconf-0.5.feather` | x, y, z (8 nm voxels), body; read in batches, only for bodies that need a synapse-centroid position |

Locked SHA-256 values live in `flycns/release/malecns_v1.py`; the download size and URL of each table are recorded
beside its hash.

## Rules

1. **Retained neurons**: rows with a non-null `superclass` (166,700 in v1.0). Order: ascending body ID, which fixes
   the neuron index used everywhere else.
2. **Position** (micrometres, voxel size 8 nm): `somaLocation`; else `tosomaLocation`; else the mean of the neuron's
   synapse points (pre and post) streamed from `syn-points`; the source per neuron is stored (0 soma, 1 to-soma,
   2 synapse centroid, 3 none).
3. **Transmitter**: `consensus_nt`; where it is `unclear` or missing, `predicted_nt`; the resolved label and which
   field supplied it are stored. **Sign** by the rule of R-105; the rule follows Shiu et al. 2024 except that
   histamine, absent from FlyWire, is inhibitory (photoreceptors).
4. **Side**: `somaSide`, else `rootSide`, else the `_L`/`_R` suffix of `instance`, else unknown.
5. **Partition**: `ol_intrinsic` and `ol_sensory` neurons go to the optic lobe of their side; superclasses beginning
   `vnc` go to the nerve cord; everything else, including visual projection and visual centrifugal neurons (which
   span the optic lobe and the central brain), goes to the central brain. The partition decides which dynamics a
   neuron runs under (graded in the optic lobe, spiking elsewhere).
6. **Edges**: stream `connectome-weights` in batches; keep a pair when both ends are retained; sum duplicates;
   counts stored as uint16 with saturation at 65,535 (the number saturated recorded). CSR by presynaptic index,
   targets ascending within a row.
7. **Columns**: per side, the distinct `(assignedOlHex1, assignedOlHex2)` pairs of neurons of that side; each column
   records the retained neurons assigned to it. **Photoreceptors** (types R1-R6, R7*, R8*) carry no hex assignment in
   the table: each is assigned to the column that receives the largest summed synapse count from it through
   column-assigned partners of its side (the lamina cartridge for R1-R6, the medulla column for R7 and R8). **Column
   type**: pale if its R7/R8 are the `p` subtypes, yellow for `y`, dorsal rim for `d`, else unknown; conflicts are
   counted and resolved by majority.

## Output: the compiled directory

`manifest.json` (schema `flycns.compiled/1`): release name and version, each source table with its hash and bytes,
counts (neurons, edges, saturated counts, position sources, transmitter sources, columns per eye), string tables
(types, classes, superclasses, transmitters, partitions), and one entry per array: name, dtype, shape, file,
SHA-256. Arrays are little-endian `.bin` files:

| Array | dtype | Shape |
|---|---|---|
| `neuron_body_id` | int64 | N |
| `neuron_type`, `neuron_class` | uint32 (index into string table) | N |
| `neuron_superclass`, `neuron_side`, `neuron_partition`, `neuron_nt`, `neuron_nt_source`, `neuron_position_source` | uint8 | N |
| `neuron_sign` | int8 | N |
| `neuron_position_um` | float32 | N x 3 |
| `neuron_column` | int32 (index into the column table, -1 when none) | N |
| `csr_indptr` | int64 | N + 1 |
| `csr_indices` | int32 | E |
| `csr_count` | uint16 | E |
| `column_side` | uint8 | C |
| `column_hex` | int16 | C x 2 |
| `column_kind` | uint8 (unknown, pale, yellow, dorsal rim) | C |

## Modules

- `flycns.compiled`: `write_compiled(path, arrays, meta)` and `read_compiled(path) -> Compiled` with hash
  verification; the `Compiled` object exposes the arrays and the string tables.
- `flycns.release.base`: hash checking (`check_table`), batch streaming helpers.
- `flycns.release.malecns_v1`: `compile_malecns_v1(tables_dir, out_dir, *, synapse_centroids=True)` implementing the
  rules above; pure functions for each rule so the synthetic tests exercise them without the real tables.
