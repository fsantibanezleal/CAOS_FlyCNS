# The spiking model (U3a): requirements

```
R-301  THE reference engine SHALL integrate the published equations with the published constants, in Brian2's step
       order, so that a literal Brian2 transcription of the published model gives identical spike times and neuron
       indices on the parity circuits.
       Gate: tests/test_lif_parity.py::test_reference_matches_a_literal_brian2_transcription

R-302  WHEN an activated neuron receives a Poisson event, THE engines SHALL add w_syn times f_poi to its voltage, and
       activated neurons SHALL have no refractory period.
       Gate: tests/test_lif.py::test_activation_forces_spikes_without_refractoriness

R-303  THE Poisson events SHALL be a pure function of (seed, neuron, step) through the counter-based hash, and their
       long-run rate SHALL match the requested rate.
       Gate: tests/test_lif.py::test_poisson_events_are_counter_based_and_have_the_requested_rate

R-304  WHEN a neuron is silenced, THE engines SHALL carry no synaptic current from it, and SHALL still deliver input
       to it, as the published silence() does; silencing through the drive and through the weights SHALL give the
       same run.
       Gate: tests/test_lif.py::test_a_silenced_neuron_still_listens_but_reaches_no_one

R-305  THE PyTorch engine SHALL agree with the reference on the whole MaleCNS graph: over the first 200 ms of a
       moderate drive (the 57 gustatory neurons of types LB1a-LB1e at 150 Hz) to an active-neuron Jaccard index and
       a per-neuron spike-count correlation of at least 0.99 each; and over 500 ms of a strong drive (all 1,428
       gustatory neurons at 150 Hz), where the network has two states and single runs diverge, the mean rates of
       five GPU trials SHALL correlate with those of five independent reference trials at least at the 5th percentile
       of the reference's correlation with itself over the 126 splits of ten reference seeds into two groups of five.
       Gate: tests/test_lif.py::test_torch_engine_matches_the_reference

R-306  THE recorder SHALL store the spikes of every step and the chosen voltage traces with their hashes, and reading
       a recording back SHALL give the same spikes and traces.
       Gate: tests/test_record.py::test_recording_round_trips

R-307  THE null graphs SHALL keep what each promises: N1 every neuron's in-degree and out-degree, each neuron's
       outgoing counts and the connection count of every partition block, dropping and counting any collision no
       swap can repair (none on MaleCNS v1.0); N2 the connection count and the count multiset of every partition
       block; N3 the wiring, the counts and the multiset of signs. Each SHALL be a simple graph (no self connection,
       no repeated pair) and a pure function of its seed.
       Gate: tests/test_nulls.py::test_null_graphs_keep_what_they_promise
```
