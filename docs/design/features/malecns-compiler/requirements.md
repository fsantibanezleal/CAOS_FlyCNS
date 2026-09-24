# MaleCNS v1.0 compiler: requirements

Written before the code (U1, issue #5). Each requirement names the gate that fails when it is violated. Gates marked
`data` need the release files on disk and are skipped (and listed) where they are absent; each of them has a synthetic
counterpart that runs everywhere.

```
R-101  WHEN a release table's SHA-256 differs from its locked value,
       THE compiler SHALL refuse the table and name the file, the expected hash and the found hash.
       Gate: tests/test_release.py::test_rejects_a_table_whose_hash_differs

R-102  THE compiler SHALL keep exactly the bodies that carry a superclass, each body ID once.
       Gate: tests/test_release.py::test_keeps_superclass_bodies_once

R-103  THE compiler SHALL keep every connection whose two ends are retained neurons, summing duplicate pairs,
       and SHALL record how many counts were saturated at 65,535.
       Gate: tests/test_release.py::test_keeps_edges_between_retained_neurons_only

R-104  THE compiler SHALL give each retained neuron a position from its soma, else its to-soma point, else the
       centroid of its synapses, and SHALL record which of the three was used.
       Gate: tests/test_release.py::test_position_falls_back_in_order

R-105  THE compiler SHALL map transmitters to signs as acetylcholine +1; GABA, glutamate and histamine -1;
       dopamine, octopamine and serotonin +1; unclear or missing 0.
       Gate: tests/test_release.py::test_sign_rule

R-106  THE compiler SHALL build each eye's column table from the release's hex assignments and SHALL assign each
       photoreceptor to the column whose neurons receive most of its synapses.
       Gate: tests/test_release.py::test_photoreceptor_joins_the_column_it_drives_most

R-107  WHEN a compiled directory is read, THE reader SHALL check every array's SHA-256 against the manifest and
       SHALL refuse the directory on any mismatch.
       Gate: tests/test_compiled.py::test_tampered_array_is_refused

R-108  THE compiler SHALL produce byte-identical arrays when run twice on the same tables.
       Gate: tests/test_compiled.py::test_compilation_is_deterministic

R-109  WHERE the MaleCNS v1.0 tables are present, THE compiled graph SHALL hold 166,700 neurons, 879 columns on the
       left eye and 892 on the right, and photoreceptors whose sign is inhibitory.
       Gate: tests/test_malecns_data.py::test_compiled_malecns_matches_the_release
```
