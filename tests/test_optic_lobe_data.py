"""The transferred optic lobes of MaleCNS v1.0: what the transfer covers, and what the real wiring then computes.

Needs the compiled MaleCNS and the measured eye geometry (``FLYCNS_MALECNS_COMPILED``); the direction-selectivity
test also needs a CUDA device. Skipped, and listed by ``-rs``, otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from flycns.compiled import read_compiled
from flycns.eyes import build_eyes
from flycns.flyvis import load_ensemble
from flycns.motion import KNOWN_PREFERRED_DEG, angular_distance_deg, selectivity_on_eye
from flycns.optic_lobe import build_optic_lobe, unit_directions

COMPILED = Path(os.environ.get("FLYCNS_MALECNS_COMPILED", "E:/_Datos/destello/compiled/malecns-v1.0"))
GEOMETRY = Path(str(COMPILED) + "-eyes-geometry.json")
present = pytest.mark.skipif(not ((COMPILED / "manifest.json").is_file() and GEOMETRY.is_file()),
                             reason=f"no compiled MaleCNS or eye geometry at {COMPILED}")


@pytest.mark.data
@present
def test_transfer_of_malecns_states_its_coverage():
    lobe = build_optic_lobe(read_compiled(COMPILED, verify=False), load_ensemble(), model=0)
    r = lobe.report
    assert r["real_units"] == 95_925
    assert r["flyvis_types_without_malecns_class"] == ["Am", "Mi11", "Mi12", "Mi3", "Tm28"]
    assert r["flyvis_drive_covered_fraction"] > 0.94
    assert 0.9 < r["count_ratio_median_weighted_by_drive"] < 1.25          # the two datasets' synapse scales agree
    # stand-ins fill the columns the release left without photoreceptors, and only those
    assert (r["stand_in_rule"]["R1-R6"]["stand_ins"], r["stand_in_rule"]["R7"]["stand_ins"],
            r["stand_in_rule"]["R8"]["stand_ins"]) == (1712, 1506, 1655)
    assert r["photoreceptors_lit"] == 5_895 + r["stand_ins"]
    # CT1: every connection of both CT1 cells moved to a per-column compartment
    ct1 = r["ct1_compartments"]
    assert ct1["connections_left_on_whole_ct1"] == 0 and ct1["connections_moved"] == 41_557
    assert ct1["M10"] + ct1["Lo1"] == ct1["compartments"] == 3_536
    assert np.isfinite(lobe.network.weight).all()


@pytest.mark.data
@pytest.mark.gpu
@present
def test_t4_on_the_real_wiring_prefers_the_known_directions():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from flycns.dynamics.graded import GradedTorch

    graph = read_compiled(COMPILED, verify=False)
    ensemble = load_ensemble()
    eyes = build_eyes(graph["column_side"], graph["column_hex"], graph["column_kind"],
                      json.loads(GEOMETRY.read_text(encoding="utf-8")))
    column_dirs = np.full((len(graph["column_side"]), 3), np.nan)
    for side in ("left", "right"):
        column_dirs[eyes[side].column_index] = eyes[side].directions
    # networks 000 and 001: their own T4 cells are direction selective with the known preferences on flyvis's lattice
    for model in (0, 1):
        lobe = build_optic_lobe(graph, ensemble, model)
        directions, _ = unit_directions(lobe, column_dirs)
        engine = GradedTorch(lobe.network, 1 / 200, device="cuda")
        steady = engine.steady_state(2.0, 0.5).cpu().numpy()
        for side in ("left", "right"):
            summary = selectivity_on_eye(engine, lobe, eyes[side], steady, directions, lobe.side).summary()
            for t in ("T4a", "T4b", "T4c", "T4d"):
                assert summary[t]["neurons"] >= 20
                error = angular_distance_deg(summary[t]["mean_preferred_deg"], KNOWN_PREFERRED_DEG[t])
                assert error < 30.0, (model, side, t, summary[t])
