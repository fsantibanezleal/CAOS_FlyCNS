"""The graded engines against flyvis 1.2.0 running its own network 000.

Needs the extraction of ``scripts/extract_flyvis_ensemble.py`` (``FLYCNS_FLYVIS_EXTRACT``, default
``E:/_Datos/destello/models/flyvis-1.2.0``): the network (45,669 neurons, 1,513,231 connections) and what flyvis
computed on it for a fixed stimulus, every neuron at every step. Skipped, and listed by ``-rs``, when absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flycns.compiled import read_compiled
from flycns.dynamics.graded import GradedNetwork, GradedReference

EXTRACT = Path(os.environ.get("FLYCNS_FLYVIS_EXTRACT", "E:/_Datos/destello/models/flyvis-1.2.0"))
present = pytest.mark.skipif(not (EXTRACT / "responses-000" / "manifest.json").is_file(),
                             reason=f"no flyvis extraction at {EXTRACT}")


def flyvis_run():
    lattice = read_compiled(EXTRACT / "lattice-000")
    responses = read_compiled(EXTRACT / "responses-000")
    network = GradedNetwork(bias=lattice["node_bias"], time_const_s=lattice["node_time_const_s"],
                            source=lattice["edge_source"], target=lattice["edge_target"],
                            weight=lattice["edge_weight"], input_index=lattice["input_index"])
    return network, responses, responses.manifest["release"]


@pytest.mark.data
@present
def test_reference_reproduces_flyvis_on_its_own_network():
    network, responses, run = flyvis_run()
    assert network.n == 45_669 and len(network.weight) == 1_513_231
    engine = GradedReference(network, run["dt_s"])
    steady = engine.steady_state(run["t_pre_s"], run["grey"])
    assert np.abs(steady - responses["initial_activity"]).max() < 1e-5
    _, activity = engine.run(responses["stimulus"], initial=steady)
    assert activity.shape == responses["activity"].shape == (200, 45_669)
    assert np.abs(activity - responses["activity"]).max() < 1e-5
    # the stimulus moves the network: a comparison of flat traces would prove nothing
    assert np.ptp(responses["activity"], axis=0).max() > 1.0


@pytest.mark.data
@pytest.mark.gpu
@present
def test_torch_engine_matches_the_reference_on_flyvis():
    torch = pytest.importorskip("torch")
    from flycns.dynamics.graded import GradedTorch

    network, responses, run = flyvis_run()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = GradedTorch(network, run["dt_s"], device=device)
    steady = engine.steady_state(run["t_pre_s"], run["grey"]).cpu().numpy()
    assert np.abs(steady - responses["initial_activity"]).max() < 1e-5
    _, activity = engine.run(responses["stimulus"], initial=steady)
    assert np.abs(activity - responses["activity"]).max() < 1e-5
    reference = GradedReference(network, run["dt_s"])
    _, ref_activity = reference.run(responses["stimulus"], initial=reference.steady_state(run["t_pre_s"], run["grey"]))
    assert np.abs(activity - ref_activity).max() < 1e-5
