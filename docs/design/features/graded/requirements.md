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
