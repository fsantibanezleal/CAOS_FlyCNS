"""E3's geometry: where flyvis's lattice columns look in an eye's frame, and the mirror between the two frames."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flycns.mapped import EXTENT, lattice_local_xy, nearest_lattice_column

EXTRACT = Path(os.environ.get("FLYCNS_FLYVIS_EXTRACT", "E:/_Datos/destello/models/flyvis-1.2.0"))


def test_lattice_positions_round_trip():
    u, v = np.meshgrid(np.arange(-EXTENT, EXTENT + 1), np.arange(-EXTENT, EXTENT + 1))
    u, v = u.ravel(), v.ravel()
    keep = np.abs(u + v) <= EXTENT
    u, v = u[keep], v[keep]
    assert len(u) == 721
    x, y = lattice_local_xy(u, v)
    ru, rv, inside = nearest_lattice_column(x, y)
    assert inside.all() and np.array_equal(ru, u) and np.array_equal(rv, v)
    # a small jitter still finds the same column; far outside the lattice is flagged
    ru, rv, _ = nearest_lattice_column(x + 1.0, y - 1.0)
    assert np.mean((ru == u) & (rv == v)) > 0.99
    assert not nearest_lattice_column(np.array([400.0]), np.array([0.0]))[2][0]
    # one u step looks 6.03 degrees lower; flyvis's 0-degree direction (v) points toward the front here
    x0, y0 = lattice_local_xy(np.array([1]), np.array([0]))
    x1, _ = lattice_local_xy(np.array([0]), np.array([1]))
    assert (x0[0], y0[0]) == pytest.approx((0.0, -6.03)) and x1[0] == pytest.approx(-6.00)


@pytest.mark.data
@pytest.mark.skipif(not (EXTRACT / "lattice-000" / "manifest.json").is_file(), reason=f"no flyvis lattice at {EXTRACT}")
def test_flyvis_t4a_prefers_front_to_back_in_the_eyes_frame():
    from flycns.compiled import read_compiled
    from flycns.dynamics.graded import GradedReference
    from flycns.flyvis import central_neurons, lattice_network, load_ensemble
    from flycns.motion import edge_on_eye

    lattice = read_compiled(EXTRACT / "lattice-000")
    ensemble = load_ensemble()
    engine = GradedReference(lattice_network(lattice, ensemble, 0), 1 / 200)
    u = lattice["node_u"][lattice["input_index"][0]].astype(float)
    v = lattice["node_v"][lattice["input_index"][0]].astype(float)
    x, y = lattice_local_xy(u, v)
    steady = engine.steady_state(1.0, 0.5)
    centre = central_neurons(lattice)
    t4a, t4b = (centre[ensemble.types.index(t)] for t in ("T4a", "T4b"))
    peaks = {}
    for angle in (0.0, 180.0):                            # front to back, back to front, in the eye's frame
        sweep = edge_on_eye(x, y, angle, 1, 19 * 5.8, 1 / 200, extent_deg=30.0, t_pre_s=0.2, t_post_s=0.2)
        _, activity = engine.run(sweep.intensity, initial=steady, record=np.array([t4a, t4b]))
        peaks[angle] = np.maximum(activity, 0).max(axis=0)
    assert peaks[0.0][0] > 1.5 * peaks[180.0][0]         # T4a: front to back
    assert peaks[180.0][1] > 1.5 * peaks[0.0][1]         # T4b: back to front
