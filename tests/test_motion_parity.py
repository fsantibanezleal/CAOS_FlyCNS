"""flycns's direction-selectivity measures against flyvis's own, on flyvis's lattice.

Needs the extraction's ``lattice-000`` and ``moving-edges`` (``FLYCNS_FLYVIS_EXTRACT``) and a CUDA device; skipped and
listed by ``-rs`` otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flycns.compiled import read_compiled
from flycns.flyvis import central_neurons, lattice_network, load_ensemble
from flycns.motion import direction_selectivity, flyvis_peaks

EXTRACT = Path(os.environ.get("FLYCNS_FLYVIS_EXTRACT", "E:/_Datos/destello/models/flyvis-1.2.0"))


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.skipif(not (EXTRACT / "moving-edges" / "manifest.json").is_file(),
                    reason=f"no flyvis moving-edge recording at {EXTRACT}")
def test_direction_selectivity_matches_flyvis_on_its_lattice():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from flycns.dynamics.graded import GradedTorch

    lattice = read_compiled(EXTRACT / "lattice-000")
    edges = read_compiled(EXTRACT / "moving-edges")
    ensemble = load_ensemble()
    centre = central_neurons(lattice)
    assert [ensemble.types[t] for t in lattice["node_type"][centre]] == edges.strings["cell_type"]
    for m in range(edges["responses"].shape[0]):
        engine = GradedTorch(lattice_network(lattice, ensemble, m), 1 / 200, device="cuda")
        steady = engine.steady_state(1.0, 0.5).cpu().numpy()
        out = np.concatenate([engine.run_batch(edges["stimulus"][b:b + 24], steady, centre)
                              for b in range(0, len(edges["stimulus"]), 24)])
        reference = edges["responses"][m]
        assert np.array_equal(np.isnan(out), np.isnan(reference))
        assert np.nanmax(np.abs(out - reference)) < 1e-4
        sel = direction_selectivity(flyvis_peaks(out, edges["time_s"], edges["sample_speed"]),
                                    edges["sample_angle_deg"], edges["sample_intensity"], edges["sample_speed"])
        assert np.abs(sel.dsi.T - edges["dsi"][m]).max() < 1e-5
        tuned = edges["dsi"][m] > 0.01
        gap = np.abs(np.angle(np.exp(1j * (sel.preferred_rad.T - edges["preferred_direction_rad"][m]))))
        assert np.degrees(gap[tuned]).max() < 0.01
        assert tuned.sum() > 50                                  # most types are tuned: the check is not vacuous
