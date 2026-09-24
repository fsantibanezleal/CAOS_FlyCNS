# The compiled directory

A compiled connectome is a directory: `manifest.json` plus one little-endian binary file per array. It is the only
thing either simulator reads, which is what makes the Python reference and the browser engine comparable: they read
the same bytes.

## Why a directory of raw arrays

- **Both languages read it without a parser.** A Python `np.frombuffer` and a TypeScript `new Int32Array(buffer)`
  see the same numbers.
- **Every array is verified.** The manifest records each array's dtype, shape, byte length and SHA-256; the reader
  recomputes the hash and refuses the directory on any difference. A simulation never runs on a graph that is not the
  one that was compiled (requirement R-107).
- **It is reproducible.** Arrays are written in name order from deterministic rules, so compiling the same tables
  twice gives the same bytes (R-108).

## The manifest

| Key | Content |
|---|---|
| `schema` | `flycns.compiled/1` |
| `release` | name, version, citation, DOI, licence, download page |
| `sources` | every release table used: file, URL, bytes, SHA-256 |
| `counts` | what the compiler measured: neurons, edges, synapses, saturated counts, self edges, position sources, transmitters and their sources, signs, sides, partitions, columns per eye, column kinds and conflicts, photoreceptors and how many joined a column |
| `strings` | the tables that index arrays point into: cell types, classes, superclasses, transmitters, sides, partitions, position sources, column kinds |
| `arrays` | name, dtype, shape, file, bytes, SHA-256 of every array |

## The arrays

N is the number of neurons, E the number of connections, C the number of eye columns.

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `neuron_body_id` | int64 | N | the release's body ID; neurons are ordered by it |
| `neuron_type`, `neuron_class` | uint32 | N | index into the `type` and `class` string tables |
| `neuron_superclass` | uint8 | N | index into `superclass` |
| `neuron_side` | uint8 | N | unknown, left, right, midline |
| `neuron_partition` | uint8 | N | optic lobe left, optic lobe right, central brain, nerve cord |
| `neuron_nt`, `neuron_nt_source` | uint8 | N | resolved transmitter and which field supplied it |
| `neuron_sign` | int8 | N | +1 excitatory, -1 inhibitory, 0 no weight |
| `neuron_position_um` | float32 | N x 3 | micrometres in the release frame (NaN when no position exists) |
| `neuron_position_source` | uint8 | N | soma, to-soma point, synapse centroid, none |
| `neuron_column` | int32 | N | index into the column table, -1 when the neuron belongs to no column |
| `csr_indptr` | int64 | N + 1 | row pointers, by presynaptic neuron |
| `csr_indices` | int32 | E | postsynaptic neuron of each connection, ascending within a row |
| `csr_count` | uint16 | E | synapse count, saturated at 65,535 (saturations counted in the manifest) |
| `column_side` | uint8 | C | left or right eye |
| `column_hex` | int16 | C x 2 | the release's two hexagonal coordinates of the column |
| `column_kind` | uint8 | C | unknown, pale, yellow, dorsal rim |

The weight a simulator puts on a connection is `neuron_sign[pre] x csr_count x w`, with `w` the model's unitary
strength; the compiled directory stores counts and signs separately so that null models and alternative sign rules
never need a recompilation.
