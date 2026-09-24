# MaleCNS v1.0 compiler: tasks

| # | Task | Satisfies |
|---|---|---|
| 1 | `flycns.compiled`: writer and hash-verifying reader for the compiled directory | R-107, R-108 |
| 2 | `flycns.release.base`: table hash check, batch streaming | R-101 |
| 3 | rule functions: retained set, sign rule, side, partition, position fallback, edge accumulation, columns, photoreceptor assignment, column kind | R-102 to R-106 |
| 4 | `compile_malecns_v1` orchestrating the rules over the real tables, with progress and a compile report | R-109 |
| 5 | synthetic release fixture (a few dozen bodies written as feathers) and the tests on it | R-101 to R-108 |
| 6 | the data test on the real tables; the compiled MaleCNS written to the vault | R-109 |
| 7 | wiki: the compiled format and the MaleCNS rules with their sources | documentation |

## Convergence (2026-09-23)

| Requirement | Gate | Result |
|---|---|---|
| R-101 hash refusal | `tests/test_release.py::test_rejects_a_table_whose_hash_differs` | pass |
| R-102 retained bodies once | `tests/test_release.py::test_keeps_superclass_bodies_once` | pass |
| R-103 edges between retained neurons, duplicates summed, saturation counted | `tests/test_release.py::test_keeps_edges_between_retained_neurons_only` | pass |
| R-104 position fallback order and source | `tests/test_release.py::test_position_falls_back_in_order` | pass |
| R-105 sign rule | `tests/test_release.py::test_sign_rule` (9 cases) and `test_sign_rule_applied_in_the_compiled_graph` | pass |
| R-106 columns and photoreceptor assignment | `tests/test_release.py::test_photoreceptor_joins_the_column_it_drives_most` | pass |
| R-107 tampered array refused | `tests/test_compiled.py::test_tampered_array_is_refused` | pass |
| R-108 deterministic compilation | `tests/test_compiled.py::test_compilation_is_deterministic` | pass |
| R-109 the compiled release matches the published counts | `tests/test_malecns_data.py::test_compiled_malecns_matches_the_release` | pass (run locally on the real tables; skipped in CI where they are absent) |

Measured on the real tables (210 s on the reference desktop): 166,700 neurons; 25,582,938 connections carrying
124,177,617 synapses, both equal to totals published independently by other projects; 0 saturated counts; 101 self
connections; positions from the soma 139,662, the to-soma point 976, the synapse centroid 25,941, none 121;
transmitters unclear for 2,100 neurons (no output weight), histamine for 7,905; 879 left and 892 right columns;
6,091 photoreceptors, 5,895 joined to a column over 1,466 distinct columns, the same two numbers flyverse reports by
its own method. R1-R6 coverage is thin: 945 of 1,771 columns have no reconstructed R1-R6 terminal, which the dynamics
unit must handle explicitly.
