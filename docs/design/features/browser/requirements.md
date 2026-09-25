# The browser engine (U3c): requirements

```
R-501  THE TypeScript hash SHALL equal the Python hash on the pinned vectors and on ten thousand random keys, and
       THE WGSL hash SHALL equal it on the same keys.
       Gate: ts/test/rng.test.ts::the_hash_matches_the_pinned_vectors_and_the_reference

R-502  THE loaders SHALL refuse a compiled directory whose arrays do not match the manifest's SHA-256, whose schema
       is not the expected one, or whose byte length does not match the declared shape, and SHALL read every allowed
       dtype, float64 included, into the matching typed array.
       Gate: ts/test/compiled.test.ts::the_loader_checks_every_array_against_its_manifest

R-503  THE CPU LIF engine SHALL give the reference's spike trains and float32 traces on every LIF scenario under
       parity/, with fixed events, counter-based activation, modulated rates, silencing and adaptation.
       Gate: ts/test/parity.test.ts::the_cpu_lif_engine_reproduces_the_reference_on_every_lif_scenario

R-504  THE CPU graded engine SHALL give the reference's activity, bit for bit, on the graded scenario.
       Gate: ts/test/parity.test.ts::the_cpu_graded_engine_reproduces_the_reference

R-505  THE CPU hybrid SHALL give the reference's spikes and graded activity on the toy CNS with the lobe's own
       source and with the lattice source.
       Gate: ts/test/parity.test.ts::the_cpu_hybrid_reproduces_the_reference_with_both_sources

R-506  THE parity fixtures SHALL be the reference's current output: regenerating every scenario's expected
       directory SHALL give the committed array hashes.
       Gate: tests/test_parity_fixtures.py::test_committed_fixtures_are_the_references_current_output

R-507  THE engine bundle SHALL carry every constant an engine step uses as a JSON number that round-trips exactly,
       and a bundle written from a network SHALL read back into the same network.
       Gate: tests/test_bundle.py::test_bundles_round_trip_with_their_constants

R-508  WHERE a WebGPU adapter is present, THE GPU LIF engine SHALL give the reference's spikes on every LIF
       scenario, and THE GPU hybrid SHALL give the reference's spikes and its graded activity within 1e-4 on the
       toy CNS.
       Gate: ts/test/gpu.test.ts::the_gpu_engines_reproduce_the_reference_on_the_scenarios

R-509  THE worker protocol SHALL run a scenario through the CPU engine from messages alone and answer with the
       same spikes the engine gives directly.
       Gate: ts/test/worker.test.ts::the_worker_protocol_gives_the_engines_spikes
```
