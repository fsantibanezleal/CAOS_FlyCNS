"""Engine bundles: the constants round-trip exactly and a bundle reads back into the network it was written from."""

from __future__ import annotations

import json

import numpy as np

from flycns.bundle import (
    SCHEMA_ENGINE,
    graded_network,
    lattice_map,
    lif_constants,
    lif_params,
    read_scenario,
    run_reference,
    write_graded_bundle,
    write_hybrid_bundle,
    write_lif_bundle,
    write_scenario,
)
from flycns.compiled import read_compiled
from flycns.dynamics import LIFParams
from flycns.parity import SCENARIOS, graded_random, hybrid_lattice_toy, lif_poisson


def test_bundles_round_trip_with_their_constants(tmp_path):
    # every constant is a JSON number that survives a round trip through text, and the step's derived constants
    # are the ones the reference computes
    params = LIFParams(adaptation_mv=1.5)
    constants = lif_constants(params)
    text = json.dumps(constants)
    assert json.loads(text) == constants
    a, b, c = params.decay()
    assert (constants["decay_a"], constants["decay_b"], constants["decay_c"]) == (a, b, c)
    assert constants["delay_steps"] == 18 and constants["refractory_steps"] == 22
    assert lif_params(constants) == params

    # a LIF bundle keeps float64 weights (the format now allows float64) and reads back exactly
    indptr, indices = np.array([0, 1, 2, 2]), np.array([1, 2])
    weights = np.array([0.275 * 200, -0.275 * 7])
    write_lif_bundle(tmp_path / "lif", indptr, indices, weights, params, meta={"note": "chain"})
    c = read_compiled(tmp_path / "lif", schema=SCHEMA_ENGINE)
    assert c["lif_weight_mv"].dtype == np.float64 and np.array_equal(c["lif_weight_mv"], weights)
    assert c.manifest["release"]["lif"] == constants and c.manifest["release"]["note"] == "chain"
    assert np.array_equal(c["lif_indptr"], indptr) and np.array_equal(c["lif_indices"], indices)

    # a graded bundle reads back into the same network
    graded = graded_random()
    net = graded_network(graded.arrays, 5)
    write_graded_bundle(tmp_path / "graded", net, 1 / 200)
    c = read_compiled(tmp_path / "graded", schema=SCHEMA_ENGINE)
    back = graded_network(c.arrays, c.manifest["release"]["graded"]["n_columns"])
    for name in ("bias", "time_const_s", "source", "target", "weight", "input_neuron", "input_column"):
        assert np.array_equal(getattr(back, name), getattr(net, name)), name
    assert back.n_columns == 5

    # a hybrid bundle with a lattice source carries both lattices and the map, per unit over its nodes
    lattice = hybrid_lattice_toy()
    from flycns.bundle import _hybrid_from

    engine = _hybrid_from(lattice)
    write_hybrid_bundle(tmp_path / "hybrid", engine.h, 1 / 200,
                        lattice=([e.net for e in engine.source.engines], lattice_map(lattice.arrays)))
    c = read_compiled(tmp_path / "hybrid", schema=SCHEMA_ENGINE)
    assert c.manifest["release"]["source"] == "lattice" and c.manifest["release"]["lattice"]["count"] == 2
    mapping = lattice_map(c.arrays)
    assert [n.tolist() for n in mapping.nodes] == [[3, 4], [5], [6, 7, 8]]
    assert c.manifest["release"]["hybrid"]["steps_per_graded"] == 50

    # a scenario round-trips its drive (rates, events in order, silencing, modulated rates) and its stimulus
    scenario = lif_poisson()
    write_scenario(tmp_path / "scenario", scenario)
    back = read_scenario(tmp_path / "scenario")
    assert back.drive.activate == scenario.drive.activate
    assert back.drive.events == scenario.drive.events
    assert np.array_equal(back.drive.silenced, scenario.drive.silenced)
    assert np.array_equal(back.drive.modulated[1], scenario.drive.modulated[1]) and back.drive.modulated[2] == 100
    assert back.steps == 2000 and back.seed == 3 and np.array_equal(back.trace_neurons, scenario.trace_neurons)
    first, second = run_reference(scenario), run_reference(back)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert set(SCENARIOS) >= {"lif-circuit-0", "graded-random", "hybrid-toy", "hybrid-lattice-toy"}
