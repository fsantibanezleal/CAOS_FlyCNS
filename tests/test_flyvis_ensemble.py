"""The flyvis ensemble inside the package: provenance, order, and the numbers network 000 really uses."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flycns.compiled import read_compiled
from flycns.flyvis import ensemble_directory, load_ensemble

EXTRACT = Path(os.environ.get("FLYCNS_FLYVIS_EXTRACT", "E:/_Datos/destello/models/flyvis-1.2.0"))


def test_the_shipped_ensemble_carries_its_provenance():
    ensemble = load_ensemble()                          # every array hash-checked on reading
    assert ensemble.n_models == 50
    assert len(ensemble.types) == 65 and ensemble.types[:9] == ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "L1")
    assert len(ensemble.pair_sign) == 604 and len(ensemble.group_n_syn) == 2355
    assert ensemble.bias.shape == ensemble.time_const_s.shape == (50, 65)
    assert ensemble.strength.shape == (50, 604) and (ensemble.strength >= 0).all()   # clamped non-negative
    assert set(np.unique(ensemble.pair_sign)) == {-1, 1}
    source = ensemble.manifest["sources"][0]
    assert len(source["checkpoints_sha256"]) == 50
    assert source["license"].startswith("MIT")
    assert ensemble.manifest["release"] == {"model": "flyvis", "version": "1.2.0", "ensemble": "flow/0000",
                                            "paper": "doi:10.1038/s41586-024-07939-3"}
    assert (ensemble_directory() / "LICENSE-flyvis.txt").read_text(encoding="utf-8").startswith("MIT License")
    pairs = ensemble.pair_index()
    assert pairs[("R1", "L1")] == 0 and ensemble.pair_sign[0] == -1      # histaminergic photoreceptor to L1


@pytest.mark.data
@pytest.mark.skipif(not (EXTRACT / "lattice-000" / "manifest.json").is_file(),
                    reason=f"no flyvis extraction at {EXTRACT}")
def test_ensemble_ships_with_its_provenance_and_order():
    ensemble = load_ensemble()
    lattice = read_compiled(EXTRACT / "lattice-000")
    types = lattice["node_type"].astype(np.int64)
    assert np.array_equal(ensemble.bias[0][types], lattice["node_bias"])
    assert np.array_equal(ensemble.time_const_s[0][types], lattice["node_time_const_s"])
    # every connection's weight from the shipped numbers: sign x mean count at the offset x strength; flyvis's
    # offsets run from the presynaptic to the postsynaptic column (target minus source)
    u, v = lattice["node_u"].astype(np.int64), lattice["node_v"].astype(np.int64)
    source, target = lattice["edge_source"].astype(np.int64), lattice["edge_target"].astype(np.int64)
    pair = lattice["edge_pair"].astype(np.int64)
    group_of = {key: g for g, key in enumerate(zip(ensemble.group_pair.tolist(), ensemble.group_du.tolist(),
                                                   ensemble.group_dv.tolist(), strict=True))}
    du, dv = (u[target] - u[source]).tolist(), (v[target] - v[source]).tolist()
    group = np.array([group_of[key] for key in zip(pair.tolist(), du, dv, strict=True)])
    weight = ensemble.pair_sign[pair] * ensemble.group_n_syn[group] * ensemble.strength[0][pair]
    assert np.abs(weight - lattice["edge_weight"]).max() < 1e-6
    assert ensemble.manifest["sources"][0]["checkpoints_sha256"] == lattice.manifest["sources"][0]["checkpoints_sha256"]
