"""E3: flyvis's own networks, one per eye, with their activity carried onto the MaleCNS neurons of the same types.

Where E2 (``flycns.optic_lobe``) runs flyvis's trained numbers on the release's wiring, E3 runs flyvis's network as
flyvis built it, on its 721-column lattice, once per eye, and gives each MaleCNS neuron of a flyvis type the activity
of the flyvis cell of that type at the lattice column that looks where the MaleCNS neuron looks. The spiking CNS then
receives that activity through the release's synapses, as in E2. It separates two questions: whether the trained
model computes the right thing (E3), and whether the release's wiring still does with the trained numbers (E2).

**Where flyvis's columns look.** flyvis's stimuli are rendered by its own renderer, so the lattice's layout in its
stimulus frame is measured from flyvis's recorded moving edges (the time each column is crossed, fitted linearly in the
hex coordinates, over the central columns and all 12 directions): one step of ``u`` moves 6.03 degrees straight down,
one step of ``v`` moves 6.00 degrees along flyvis's 0-degree direction and 3.02 degrees down. flyvis's 0-degree
direction is back to front (its T4b prefers it), the mirror of the eyes' local frame here (0 degrees front to back),
and 90 degrees is upward in both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compiled import Compiled
from .optic_lobe import CLASS_TO_FLYVIS, CT1_COMPARTMENTS, OpticLobe

#: flyvis's lattice steps in its stimulus frame (degrees; x along flyvis's 0-degree direction, y up), measured from
#: flyvis's own moving-edge stimuli (``docs/models/04_optic_lobe.md``).
U_STEP_DEG = np.array([0.0, -6.03])
V_STEP_DEG = np.array([6.00, -3.02])
EXTENT = 15


def lattice_local_xy(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where lattice column (u, v) looks, in an eye's local frame (x toward the back, y up, degrees)."""
    xf = u * U_STEP_DEG[0] + v * V_STEP_DEG[0]
    yf = u * U_STEP_DEG[1] + v * V_STEP_DEG[1]
    return -xf, yf


def nearest_lattice_column(x_deg: np.ndarray, y_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lattice column nearest to each local direction: (u, v, inside), rounding in cube coordinates."""
    basis = np.array([U_STEP_DEG, V_STEP_DEG]).T                  # columns: u step, v step
    uv = np.linalg.solve(basis, np.vstack([-np.asarray(x_deg), np.asarray(y_deg)]))
    q, r = uv
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    inside = (np.abs(rq) <= EXTENT) & (np.abs(rr) <= EXTENT) & (np.abs(rq + rr) <= EXTENT)
    return rq.astype(np.int64), rr.astype(np.int64), inside


@dataclass
class LatticeMap:
    """Which lattice cells stand for which optic-lobe units, per eye."""

    unit: np.ndarray                 # optic-lobe units that receive a lattice activity
    side: np.ndarray                 # 0 left, 1 right: which eye's lattice
    nodes: list[np.ndarray]          # per unit, the lattice nodes averaged (one, or R1 to R6 for an R1-R6 unit)
    report: dict


def map_to_lattice(lobe: OpticLobe, lattice: Compiled, types: tuple[str, ...], unit_directions: np.ndarray,
                   eyes: dict) -> LatticeMap:
    """Each unit of a flyvis-mapped class, on each eye, to the lattice cells of its flyvis types at the column that
    looks where the unit looks (``unit_directions`` from ``optic_lobe.unit_directions``)."""
    from .motion import eye_centre, local_frame
    from .optic_lobe import azimuth_elevation

    node_type = lattice["node_type"].astype(np.int64)
    node_u, node_v = lattice["node_u"].astype(np.int64), lattice["node_v"].astype(np.int64)
    index = {(int(t), int(a), int(b)): i for i, (t, a, b) in enumerate(zip(node_type, node_u, node_v, strict=True))}
    names = np.array(lobe.classes, dtype=object)[lobe.unit_class]
    flyvis_types = set(types)
    stands_for = {c: (c,) for c in set(names.tolist()) if c in flyvis_types}
    stands_for.update({c: t for c, t in CLASS_TO_FLYVIS.items()})
    stands_for.update({c: (c,) for c in CT1_COMPARTMENTS})
    units, sides, nodes = [], [], []
    outside = 0
    for s, side in enumerate(("left", "right")):
        eye = eyes[side]
        centre = eye_centre(eye.azimuth_deg, eye.elevation_deg)
        candidates = np.flatnonzero((lobe.side == side) & np.isin(names, list(stands_for))
                                    & np.isfinite(unit_directions[:, 0]))
        az, el = azimuth_elevation(unit_directions[candidates], side)
        x, y = local_frame(az, el, *centre)
        u, v, inside = nearest_lattice_column(x, y)
        outside += int((~inside).sum())
        for unit, a, b, ok in zip(candidates, u, v, inside, strict=True):
            if not ok:
                continue
            found = [index.get((types.index(t), int(a), int(b))) for t in stands_for[names[unit]]]
            found = [f for f in found if f is not None]
            if found:
                units.append(unit)
                sides.append(s)
                nodes.append(np.array(found))
    report = {"units_mapped": len(units), "units_outside_the_lattice": outside}
    return LatticeMap(unit=np.array(units, dtype=np.int64), side=np.array(sides, dtype=np.int64), nodes=nodes,
                      report=report)
