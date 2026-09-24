"""The four engines on the whole MaleCNS: what a flash does to each, and what the stabilisers do to the published model.

A full-field flash: 200 ms of grey (the steady state the engines start from), then 300 ms of full intensity on every
column of both eyes. Needs the compiled MaleCNS, the eye geometry, the flyvis extraction and a CUDA device; skipped,
and listed by ``-rs``, otherwise. About four minutes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

COMPILED = Path(os.environ.get("FLYCNS_MALECNS_COMPILED", "E:/_Datos/destello/compiled/malecns-v1.0"))
GEOMETRY = Path(str(COMPILED) + "-eyes-geometry.json")
EXTRACT = Path(os.environ.get("FLYCNS_FLYVIS_EXTRACT", "E:/_Datos/destello/models/flyvis-1.2.0"))
GREY_FRAMES, FLASH_FRAMES = 40, 60
SPLIT = GREY_FRAMES * 50                                 # LIF steps of grey

pytestmark = [pytest.mark.data, pytest.mark.gpu, pytest.mark.skipif(
    not ((COMPILED / "manifest.json").is_file() and GEOMETRY.is_file() and (EXTRACT / "lattice-000").is_dir()),
    reason=f"no compiled MaleCNS, eye geometry or flyvis extraction at {COMPILED}")]


@pytest.fixture(scope="module")
def world():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from flycns.compiled import read_compiled
    from flycns.flyvis import load_ensemble

    graph = read_compiled(COMPILED, verify=False)
    return graph, load_ensemble(), np.array(graph.strings["superclass"], dtype=object)[graph["neuron_superclass"]]


def flash(columns: int, shape=None) -> np.ndarray:
    shape = shape or (columns,)
    return np.r_[np.full((GREY_FRAMES, *shape), 0.5), np.full((FLASH_FRAMES, *shape), 1.0)].astype(np.float32)


def counts(run, neurons_of_units, superclass) -> dict[str, tuple[int, int]]:
    steps, units = run.spike_times()
    names = superclass[neurons_of_units[units]]
    return {k: (int(((steps < SPLIT) & (names == k)).sum()), int(((steps >= SPLIT) & (names == k)).sum()))
            for k in ("visual_projection", "cb_intrinsic", "descending_neuron", "vnc_motor")}


def test_e2_carries_light_from_the_eyes_to_the_motor_neurons(world):
    from flycns.dynamics.hybrid import HybridTorch, build_hybrid
    from flycns.optic_lobe import build_optic_lobe

    graph, ensemble, superclass = world
    hybrid = build_hybrid(graph, build_optic_lobe(graph, ensemble, 0))
    run = HybridTorch(hybrid, device="cuda").run(flash(len(graph["column_side"])))
    c = counts(run.spikes, hybrid.spiking_neuron, superclass)
    assert all(grey == 0 for grey, _ in c.values())        # grey: nothing, as the published model fires nothing
    assert c["visual_projection"][1] > 10_000 and c["cb_intrinsic"][1] > 10_000
    assert c["descending_neuron"][1] > 1_000 and c["vnc_motor"][1] > 500


def test_e1_photoreceptors_fire_and_the_spiking_lamina_passes_nothing(world):
    from flycns.dynamics import LIFTorch, synaptic_weights
    from flycns.dynamics.lif import photoreceptor_drive

    graph, _, superclass = world
    types = np.array(graph.strings["type"], dtype=object)[graph["neuron_type"]]
    column = graph["neuron_column"].astype(np.int64)
    photo = np.flatnonzero(np.array([isinstance(t, str) and (t == "R1-R6" or t.startswith(("R7", "R8")))
                                     for t in types]) & (column >= 0))
    weights = synaptic_weights(graph["csr_indptr"], graph["csr_indices"], graph["csr_count"], graph["neuron_sign"],
                               0.275)
    engine = LIFTorch(graph["csr_indptr"], graph["csr_indices"], weights, device="cuda")
    run = engine.run((GREY_FRAMES + FLASH_FRAMES) * 50,
                     photoreceptor_drive(photo, column[photo], flash(len(graph["column_side"]))), seed=0)
    steps, neurons = run.spike_times()
    assert np.isin(neurons, photo).sum() > 100_000
    # histaminergic photoreceptors inhibit the lamina, and inhibition of a silent spiking neuron carries nothing
    assert np.isin(neurons, photo).all()


def test_e3_carries_flyvis_activity_to_the_central_brain(world):
    from flycns.compiled import read_compiled
    from flycns.dynamics.hybrid import build_hybrid, lattice_hybrid
    from flycns.eyes import build_eyes
    from flycns.flyvis import lattice_network
    from flycns.mapped import map_to_lattice
    from flycns.optic_lobe import build_optic_lobe, unit_directions

    graph, ensemble, superclass = world
    lattice = read_compiled(EXTRACT / "lattice-000")
    eyes = build_eyes(graph["column_side"], graph["column_hex"], graph["column_kind"],
                      json.loads(GEOMETRY.read_text(encoding="utf-8")))
    column_dirs = np.full((len(graph["column_side"]), 3), np.nan)
    for side in ("left", "right"):
        column_dirs[eyes[side].column_index] = eyes[side].directions
    lobe = build_optic_lobe(graph, ensemble, 0)
    directions, _ = unit_directions(lobe, column_dirs)
    mapping = map_to_lattice(lobe, lattice, ensemble.types, directions, eyes)
    assert mapping.report["units_mapped"] > 70_000
    hybrid = build_hybrid(graph, lobe)
    net = lattice_network(lattice, ensemble, 0)
    run = lattice_hybrid(hybrid, [net, net], mapping, device="cuda").run(flash(721, (2, 721)))
    c = counts(run.spikes, hybrid.spiking_neuron, superclass)
    assert all(grey == 0 for grey, _ in c.values())
    assert c["visual_projection"][1] > 10_000 and c["vnc_motor"][1] > 500


def test_e4_stabilisers_leave_the_published_model_one_state(world):
    from flycns.dynamics import Drive, LIFParams, LIFTorch, synaptic_weights
    from flycns.dynamics.lif import stabilised_weights

    graph, _, _ = world
    classes = np.array(graph.strings["class"], dtype=object)[graph["neuron_class"]]
    drive = Drive(activate={int(i): 150.0 for i in np.flatnonzero(classes == "gustatory")})
    args = (graph["csr_indptr"], graph["csr_indices"], graph["csr_count"], graph["neuron_sign"])
    totals = {}
    for name, weights, params in (
            ("published", synaptic_weights(*args, 0.275), LIFParams()),
            ("stabilised", stabilised_weights(*args, graph["neuron_type"], 0.275), LIFParams(adaptation_mv=1.5))):
        engine = LIFTorch(graph["csr_indptr"], graph["csr_indices"], weights, params, device="cuda")
        totals[name] = np.array([engine.run(5000, drive, seed=s).spike_counts().sum() for s in range(10)])
    published, stabilised = totals["published"], totals["stabilised"]
    assert published.max() > 1.5 * published.min()                  # two states: the switch comes or not
    assert stabilised.max() < 1.05 * stabilised.min()                # one state
