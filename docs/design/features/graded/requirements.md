# The graded optic lobe (U3b): requirements

## Part 1: the graded engine and the flyvis ensemble (0.04.000)

```
R-401  THE NumPy graded engine SHALL integrate flyvis's published dynamics (tau_eff dV/dt = -V + b + sum w max(V,0)
       + x, tau_eff = max(tau, dt), forward Euler) so that, on flyvis 1.2.0's network 000 and the fixed stimulus of
       the extraction, it reproduces flyvis's steady state and every neuron's activity at every step to within
       1e-5 in absolute value (measured: 2.4e-6, float32 rounding in flyvis).
       Gate: tests/test_graded_parity.py::test_reference_reproduces_flyvis_on_its_own_network

R-402  THE PyTorch graded engine SHALL agree with the NumPy engine and with flyvis on that run to within 1e-5 in
       absolute value.
       Gate: tests/test_graded_parity.py::test_torch_engine_matches_the_reference_on_flyvis

R-403  THE package SHALL carry the parameters of flyvis's 50 pretrained networks in flyvis's own order, with the
       SHA-256 of every source checkpoint and flyvis's licence, and the parameters it carries for network 000 SHALL
       give exactly the weights, resting potentials and time constants of the extracted network 000.
       Gate: tests/test_flyvis_ensemble.py::test_ensemble_ships_with_its_provenance_and_order

R-404  ONE step of the graded engines SHALL equal the published formula on a small graph, including the rule
       tau_eff = max(tau, dt), the rectification of presynaptic activity, and input applied to the input neurons
       only.
       Gate: tests/test_graded.py::test_one_step_is_the_published_formula
```

## Part 2: the transfer onto the MaleCNS optic lobes (0.05.000)

```
R-411  THE transfer SHALL hold its rules on a toy optic lobe: flyvis's strength per synapse on a mapped pair; each
       neuron's drive from a presynaptic class capped at flyvis's; flyvis's initial drive (0.01 x 2) per default
       class and neuron; stand-in photoreceptors making up exactly what incomplete columns lack, seeing their
       column's light; CT1's connections moved to the compartments of their partners' columns.
       Gate: tests/test_optic_lobe.py::test_transfer_rules_on_a_toy_optic_lobe

R-412  WHERE the compiled MaleCNS is present, THE transfer SHALL keep all 95,925 optic-lobe neurons, name the flyvis
       types it cannot map (Am, Mi11, Mi12, Mi3, Tm28), cover at least 94% of flyvis's trained drive, place the two
       datasets' synapse counts on one scale (drive-weighted median ratio between 0.9 and 1.25), add 1,712 R1-R6,
       1,506 R7 and 1,655 R8 stand-ins, and move all 41,557 CT1 connections into 3,536 compartments.
       Gate: tests/test_optic_lobe_data.py::test_transfer_of_malecns_states_its_coverage

R-413  THE direction-selectivity measures SHALL be flyvis's: on flyvis's lattice, for networks 000 to 004, the DSI
       within 1e-5 of flyvis's own and the preferred direction within 0.01 degrees wherever flyvis's DSI exceeds
       0.01; on known inputs (a cosine tuning, an edge of known speed) exactly what they state.
       Gate: tests/test_motion_parity.py::test_direction_selectivity_matches_flyvis_on_its_lattice

R-414  THE measures SHALL give a cosine tuning its DSI and preferred direction, apply flyvis's time window, and render
       an edge that crosses columns at the stated speed and direction.
       Gate: tests/test_motion.py::test_measures_on_known_tuning_and_a_moving_edge

R-415  WHERE the compiled MaleCNS and a GPU are present, THE T4 cells of the transferred optic lobe SHALL prefer
       their known directions (Maisak et al. 2013) through the modelled eyes: for networks 000 and 001, whose own T4
       cells are selective with the known directions on flyvis's lattice, each T4 subtype's DSI-weighted mean
       preferred direction on each eye within 30 degrees of the known one.
       Gate: tests/test_optic_lobe_data.py::test_t4_on_the_real_wiring_prefers_the_known_directions
```

## Part 3: the whole CNS coupled, engines E1 to E4 (0.06.000)

```
R-421  THE bridge SHALL drive a spiking target from a graded source's release above grey exactly as a spike train of
       rate beta times that deviation would (its voltage settling where the exact step puts it), SHALL carry nothing
       at grey, and the feedback SHALL shift a graded target by the class's initial drive times the filtered rate
       over beta.
       Gate: tests/test_hybrid.py::test_the_bridge_and_the_feedback_have_their_closed_forms

R-422  THE PyTorch hybrid SHALL give the reference's spikes and, to 1e-4, its graded activity on a small CNS.
       Gate: tests/test_hybrid.py::test_the_torch_hybrid_matches_the_reference

R-423  THE stabilisers SHALL do what they state: adaptation off is the published model spike for spike and on lowers
       the rate of driven neurons; modulated Poisson input at a constant rate is the published activation event for
       event, and a zero rate gives no event.
       Gate: tests/test_stabilisers.py::test_stabilisers_do_what_they_state

R-424  THE stabilised weights SHALL cap each connection, damp same-type connections and normalise large fan-in in
       the stated order.
       Gate: tests/test_stabilisers.py::test_stabilised_weights_cap_damp_and_normalise

R-425  THE lattice geometry of E3 SHALL round-trip every column of flyvis's lattice and mirror flyvis's frame into
       the eyes' frame so that flyvis's own T4a prefers front-to-back motion there.
       Gate: tests/test_mapped.py::test_flyvis_t4a_prefers_front_to_back_in_the_eyes_frame

R-426  WHERE the compiled MaleCNS and a GPU are present, E2 SHALL leave the whole spiking CNS silent at grey and carry
       a full-field flash to visual projection, central-brain, descending and nerve-cord motor neurons.
       Gate: tests/test_whole_cns_data.py::test_e2_carries_light_from_the_eyes_to_the_motor_neurons

R-427  E1 (the published model everywhere, light as Poisson input to photoreceptors) SHALL show its documented
       failure: the photoreceptors fire and no other neuron does.
       Gate: tests/test_whole_cns_data.py::test_e1_photoreceptors_fire_and_the_spiking_lamina_passes_nothing

R-428  E3 SHALL map more than 70,000 MaleCNS units onto flyvis's lattices and carry a flash to the visual projection
       and motor neurons, silent at grey.
       Gate: tests/test_whole_cns_data.py::test_e3_carries_flyvis_activity_to_the_central_brain

R-429  E4's stabilisers SHALL leave the published model one state under the strong gustatory drive: ten seeds'
       spike totals within 5% of each other, where the published model's spread by more than half.
       Gate: tests/test_whole_cns_data.py::test_e4_stabilisers_leave_the_published_model_one_state
```
