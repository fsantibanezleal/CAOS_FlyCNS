"""The two compound eyes, over the release's own columns.

What is real: the set of columns of each eye and their hexagonal coordinates (from the release), which
photoreceptors belong to which column (compiled), and the lattice's orientation in the animal (measured here from the
3D centres of the medulla columns and from landmark neuropils). What is modelled: each column's viewing direction and
each ommatidium's acceptance. The model lays the measured lattice on the sphere with one inter-ommatidial angle,
oriented by the measured axes, mirrored anterior-posterior by the first optic chiasm, and scaled so that each eye's
equator spans the extent measured by micro-CT: from 10 degrees into the opposite hemisphere in front to 155 degrees
behind (Zhao et al., Nature 646:135-142, 2025, doi:10.1038/s41586-025-09276-5).

Frames. The release frame is the EM volume's (micrometres; +x is the fly's left). The body frame is x forward
(anterior), y left, z up (dorsal); both are right-handed. Panoramas are equirectangular in the body frame: longitude 0
is straight ahead and increases toward the fly's left, latitude +90 is straight up.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Eye extent along the equator, degrees of azimuth from straight ahead toward the eye's own side
#: (negative: into the opposite hemisphere). Zhao et al. 2025, micro-CT of whole heads, three flies.
EQUATOR_FRONT_DEG = -10.0
EQUATOR_BACK_DEG = 155.0

#: Acceptance half-width (full width at half maximum) of R1-R6, dark-adapted: 8.23 deg
#: (Gonzalez-Bellido, Wardill and Juusola, PNAS 108:4224, 2011, doi:10.1073/pnas.1014438108).
ACCEPTANCE_DEG = 8.23

#: Landmark neuropils that define the body axes: the antennal lobes are anterior, the mushroom-body calyces
#: posterior; the protocerebral bridge is dorsal, the gnathal ganglia ventral; the two medullae give left-right.
LANDMARKS = ("AL(R)", "AL(L)", "CA(R)", "CA(L)", "PB", "GNG", "ME(R)", "ME(L)")

_COLUMN_LABEL = re.compile(r"ME_([LR])_col_(\d+)_(\d+)")


# ------------------------------------------------------------------------------------------- geometry measurement


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def body_axes(landmarks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Anterior, dorsal and left unit vectors in the release frame, orthonormal and right-handed.

    Left points from the right medulla to the left one; anterior from the calyces to the antennal lobes (averaged over
    the two sides, made orthogonal to left); dorsal from the gnathal ganglia to the protocerebral bridge, made
    orthogonal to both. For x forward, y left, z up, forward x left = up, so the frame is right-handed.
    """
    left = unit(landmarks["ME(L)"] - landmarks["ME(R)"])
    anterior = 0.5 * ((landmarks["AL(R)"] - landmarks["CA(R)"]) + (landmarks["AL(L)"] - landmarks["CA(L)"]))
    anterior = unit(anterior - left * anterior.dot(left))
    dorsal = landmarks["PB"] - landmarks["GNG"]
    dorsal = unit(dorsal - left * dorsal.dot(left) - anterior * dorsal.dot(anterior))
    if np.dot(np.cross(anterior, left), dorsal) < 0:
        dorsal = -dorsal
    return {"anterior": anterior, "dorsal": dorsal, "left": left}


def lattice_orientation(hexes: np.ndarray, centres: np.ndarray, axes: dict[str, np.ndarray]) -> dict:
    """How the two hex axes run in the medulla, and which offsets are nearest neighbours.

    A least-squares fit ``centre ~ c0 + hex1 * e1 + hex2 * e2`` gives the 3D step of each hex axis; the step's
    anterior and dorsal components say how that axis runs in the animal. Neighbour offsets are tallied from the six
    nearest columns in 3D.
    """
    design = np.c_[np.ones(len(hexes)), hexes.astype(np.float64)]
    coefficients, *_ = np.linalg.lstsq(design, centres, rcond=None)
    steps = coefficients[1:3]
    tally: dict[tuple[int, int], int] = {}
    for i in range(len(centres)):
        distance = np.linalg.norm(centres - centres[i], axis=1)
        for j in np.argsort(distance)[1:7]:
            offset = (int(hexes[j, 0] - hexes[i, 0]), int(hexes[j, 1] - hexes[i, 1]))
            tally[offset] = tally.get(offset, 0) + 1
    ranked = sorted(tally.items(), key=lambda item: -item[1])
    neighbours = [offset for offset, _ in ranked[:6]]
    components = [{"anterior": float(s.dot(axes["anterior"])), "dorsal": float(s.dot(axes["dorsal"])),
                   "left": float(s.dot(axes["left"])), "length_um": float(np.linalg.norm(s))} for s in steps]
    vertical = int(np.argmax([abs(c["dorsal"]) for c in components]))
    return {"hex_steps": components, "vertical_axis": vertical, "neighbour_offsets": neighbours,
            "neighbour_tally": [[list(o), n] for o, n in ranked[:10]]}


def measure_geometry(synapse_table: Path, out_file: Path | None = None, progress=None) -> dict:
    """Stream the release's synapse points once: medulla column centres (per eye) and landmark centroids.

    Uses the per-synapse labels the release carries (``medulla_l_column``, ``medulla_r_column``, ``primary``).
    Returns (and optionally writes) the geometry that :func:`build_eyes` needs.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    columns: dict[tuple[str, int, int], np.ndarray] = {}
    marks = {name: np.zeros(4) for name in LANDMARKS}
    with pa.memory_map(str(synapse_table)) as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            x = batch.column("x").to_numpy().astype(np.float64)
            y = batch.column("y").to_numpy().astype(np.float64)
            z = batch.column("z").to_numpy().astype(np.float64)
            primary = batch.column("primary")
            codes, labels = primary.indices.to_numpy(), primary.dictionary.to_pylist()
            for name in LANDMARKS:
                if name in labels:
                    hit = codes == labels.index(name)
                    if hit.any():
                        marks[name] += [x[hit].sum(), y[hit].sum(), z[hit].sum(), hit.sum()]
            for field in ("medulla_r_column", "medulla_l_column"):
                column = batch.column(field)
                codes, labels = column.indices.to_numpy(), column.dictionary.to_pylist()
                parsed = [_COLUMN_LABEL.fullmatch(label or "") for label in labels]
                valid = np.array([p is not None for p in parsed])
                hit = valid[codes]
                if not hit.any():
                    continue
                idx = codes[hit]
                n = np.bincount(idx, minlength=len(labels))
                sx = np.bincount(idx, x[hit], len(labels))
                sy = np.bincount(idx, y[hit], len(labels))
                sz = np.bincount(idx, z[hit], len(labels))
                for j in np.flatnonzero(n):
                    match = parsed[j]
                    key = (match.group(1), int(match.group(2)), int(match.group(3)))
                    columns.setdefault(key, np.zeros(4))
                    columns[key] += [sx[j], sy[j], sz[j], n[j]]
            if progress and index % 1000 == 0:
                progress(f"synapse points: batch {index}")
    voxel = 0.008
    geometry = {
        "schema": "flycns.eyes-geometry/1",
        "source": str(Path(synapse_table).name),
        "landmarks_um": {k: [float(v[0] / v[3] * voxel), float(v[1] / v[3] * voxel), float(v[2] / v[3] * voxel)]
                         for k, v in marks.items() if v[3] > 0},
        "landmark_points": {k: int(v[3]) for k, v in marks.items()},
        "medulla_columns_um": {f"{s}_{h1}_{h2}": [float(v[0] / v[3] * voxel), float(v[1] / v[3] * voxel),
                                                   float(v[2] / v[3] * voxel), int(v[3])]
                               for (s, h1, h2), v in sorted(columns.items())},
    }
    if out_file is not None:
        Path(out_file).write_text(json.dumps(geometry, indent=1) + "\n", encoding="utf-8")
    return geometry


# ------------------------------------------------------------------------------------------------------- eye model


@dataclass
class Eye:
    """One compound eye: its columns, their viewing directions (body frame) and their neighbours."""

    side: str                      # "left" or "right"
    column_index: np.ndarray       # index into the compiled column table
    hexes: np.ndarray              # (C, 2) release hex coordinates
    kinds: np.ndarray              # column kind codes from the compiled table
    directions: np.ndarray         # (C, 3) unit vectors, body frame (x forward, y left, z up)
    azimuth_deg: np.ndarray        # from straight ahead toward the eye's own side
    elevation_deg: np.ndarray
    neighbours: np.ndarray         # (C, 6) indices into this eye's columns, -1 where absent
    delta_phi_deg: float           # the inter-ommatidial angle the extent implies
    orientation: dict | None = None  # the measured lattice orientation this eye was built from

    def to_json(self) -> dict:
        return {"side": self.side, "delta_phi_deg": self.delta_phi_deg,
                "column_index": self.column_index.tolist(), "hex": self.hexes.tolist(),
                "kind": self.kinds.tolist(), "azimuth_deg": np.round(self.azimuth_deg, 4).tolist(),
                "elevation_deg": np.round(self.elevation_deg, 4).tolist(),
                "direction": np.round(self.directions, 6).tolist(), "neighbours": self.neighbours.tolist()}


def eye_plane(hexes: np.ndarray, orientation: dict) -> np.ndarray:
    """Ideal hexagonal lattice positions in the eye, in units of one inter-ommatidial step: (forward, up).

    The vertical hex axis is the one whose medulla step is mostly dorsal; it becomes the eye's vertical neighbour
    direction, up if its medulla step points dorsal. The other axis sits 120 degrees from it (so the third neighbour
    offset, the sum of the two, is also one step long), on the side where its medulla step points: posterior in the
    medulla becomes FORWARD in the eye, because the first optic chiasm mirrors the anterior-posterior axis while
    keeping dorsal-ventral.
    """
    steps = orientation["hex_steps"]
    v = orientation["vertical_axis"]
    h = 1 - v
    up_sign = 1.0 if steps[v]["dorsal"] > 0 else -1.0
    forward_sign = -1.0 if steps[h]["anterior"] > 0 else 1.0     # the chiasm: anterior in the medulla looks back
    # The third neighbour pair decides the angle between the two axes: if (1, 1) is a neighbour offset the axes sit
    # 120 degrees apart (their sum is one step long), if (1, -1) is, 60 degrees apart (their difference is).
    offsets = {tuple(o) for o in orientation["neighbour_offsets"]}
    same_sign = (1, 1) in offsets or (-1, -1) in offsets
    tilt = -up_sign * 0.5 if same_sign else up_sign * 0.5      # vertical component of the other axis, in steps
    vertical_step = np.array([0.0, up_sign])
    other_step = np.array([forward_sign * math.cos(math.radians(30)), tilt])
    basis = np.zeros((2, 2))
    basis[v] = vertical_step
    basis[h] = other_step
    return hexes.astype(np.float64) @ basis


def _directions(azimuth_deg: np.ndarray, elevation_deg: np.ndarray, side: str) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    lateral = -1.0 if side == "right" else 1.0      # the right eye looks toward -y (the fly's right)
    return np.stack([np.cos(el) * np.cos(az), lateral * np.cos(el) * np.sin(az), np.sin(el)], axis=1)


def place_eye(plane: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Viewing directions for one eye's lattice: an azimuthal-equidistant placement about the eye's centre.

    The equator is the median vertical position (as many columns above as below; an assumption, stated). The step
    angle is chosen so that the columns within half a step of the equator span exactly the measured extent, from
    ``EQUATOR_FRONT_DEG`` to ``EQUATOR_BACK_DEG``. Around the centre direction, the equidistant placement keeps every
    column's angular distance from the centre equal to its lattice distance, so neighbours stay nearly one step apart.
    """
    forward, up = plane[:, 0], plane[:, 1]
    up0 = float(np.median(up))
    near_equator = np.abs(up - up0) <= 0.5
    front, back = float(forward[near_equator].max()), float(forward[near_equator].min())
    step = (EQUATOR_BACK_DEG - EQUATOR_FRONT_DEG) / (front - back)
    centre_az = 0.5 * (EQUATOR_FRONT_DEG + EQUATOR_BACK_DEG)
    # offsets from the centre in degrees: x toward the back of the eye (increasing azimuth), y up
    x = (0.5 * (front + back) - forward) * step
    y = (up - up0) * step
    rho = np.radians(np.hypot(x, y))
    bearing = np.arctan2(y, x)
    c_az, c_el = math.radians(centre_az), 0.0
    # rotate the centre direction by rho along the bearing (spherical destination formula)
    el = np.arcsin(np.sin(c_el) * np.cos(rho) + np.cos(c_el) * np.sin(rho) * np.sin(bearing))
    az = c_az + np.arctan2(np.cos(bearing) * np.sin(rho) * np.cos(c_el), np.cos(rho) - np.sin(c_el) * np.sin(el))
    azimuth, elevation = np.degrees(az), np.degrees(el)
    return azimuth, elevation, float(step)


def neighbour_table(hexes: np.ndarray, offsets: list) -> np.ndarray:
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(hexes)}
    table = np.full((len(hexes), 6), -1, dtype=np.int32)
    for i, (a, b) in enumerate(hexes):
        for k, (da, db) in enumerate(offsets[:6]):
            table[i, k] = index.get((int(a + da), int(b + db)), -1)
    return table


def build_eyes(column_side: np.ndarray, column_hex: np.ndarray, column_kind: np.ndarray, geometry: dict) -> dict:
    """Both eyes from the compiled column table and the measured geometry."""
    landmarks = {k: np.array(v) for k, v in geometry["landmarks_um"].items()}
    axes = body_axes(landmarks)
    centres = geometry["medulla_columns_um"]
    eyes = {}
    for code, side, tag in ((1, "left", "L"), (2, "right", "R")):
        chosen = np.flatnonzero(column_side == code)
        hexes = column_hex[chosen].astype(np.int64)
        measured = [(i, centres.get(f"{tag}_{h[0]}_{h[1]}")) for i, h in zip(chosen, hexes, strict=True)]
        with_centre = [(i, c) for i, c in measured if c is not None]
        hx = np.array([column_hex[i] for i, _ in with_centre], dtype=np.int64)
        xyz = np.array([c[:3] for _, c in with_centre])
        orientation = lattice_orientation(hx, xyz, axes)
        plane = eye_plane(hexes, orientation)
        azimuth, elevation, step = place_eye(plane, side)
        eyes[side] = Eye(side=side, column_index=chosen.astype(np.int32), hexes=hexes.astype(np.int16),
                         kinds=column_kind[chosen], directions=_directions(azimuth, elevation, side),
                         azimuth_deg=azimuth, elevation_deg=elevation,
                         neighbours=neighbour_table(hexes, orientation["neighbour_offsets"]), delta_phi_deg=step,
                         orientation=orientation)
    eyes["axes"] = {k: v.tolist() for k, v in axes.items()}
    return eyes


# ---------------------------------------------------------------------------------------------------- sampling


@dataclass
class Sampler:
    """Acceptance-weighted sampling of an equirectangular panorama by one eye (CSR over panorama pixels)."""

    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    centre_pixel: np.ndarray       # the pixel under each ommatidium's central ray (for ground truth)
    width: int
    height: int

    def sample(self, panorama: np.ndarray) -> np.ndarray:
        flat = np.asarray(panorama, dtype=np.float64).reshape(-1)
        values = flat[self.indices] * self.weights
        out = np.add.reduceat(values, self.indptr[:-1]) if len(values) else np.zeros(len(self.indptr) - 1)
        empty = self.indptr[1:] == self.indptr[:-1]
        out[empty] = np.nan
        return out

    def centre(self, panorama: np.ndarray) -> np.ndarray:
        return np.asarray(panorama).reshape(-1)[self.centre_pixel]


def pixel_directions(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (body frame) of the pixel centres of an equirectangular panorama, and each pixel's solid angle."""
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi          # -pi (behind, right) .. +pi (behind, left)
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi         # +pi/2 top row .. -pi/2 bottom row
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    directions = np.stack([np.cos(lat_grid) * np.cos(lon_grid), np.cos(lat_grid) * np.sin(lon_grid),
                           np.sin(lat_grid)], axis=-1).reshape(-1, 3)
    solid = (np.cos(lat_grid) * (2 * np.pi / width) * (np.pi / height)).reshape(-1)
    return directions, solid


def sampler(eye: Eye, width: int, height: int, acceptance_deg: float = ACCEPTANCE_DEG, cutoff_sigmas: float = 3.0
            ) -> Sampler:
    """Gaussian acceptance of full width at half maximum ``acceptance_deg`` around each ommatidium's direction."""
    pixels, solid = pixel_directions(width, height)
    sigma = math.radians(acceptance_deg) / (2 * math.sqrt(2 * math.log(2)))
    cos_cut = math.cos(cutoff_sigmas * sigma)
    indptr = [0]
    indices, weights = [], []
    centre = np.zeros(len(eye.directions), dtype=np.int64)
    for c, d in enumerate(eye.directions):
        cosine = pixels @ d
        centre[c] = int(np.argmax(cosine))
        near = np.flatnonzero(cosine >= cos_cut)
        angle = np.arccos(np.clip(cosine[near], -1.0, 1.0))
        w = np.exp(-0.5 * (angle / sigma) ** 2) * solid[near]
        total = w.sum()
        if total > 0:
            indices.append(near)
            weights.append(w / total)
        indptr.append(indptr[-1] + (len(near) if total > 0 else 0))
    return Sampler(indptr=np.array(indptr, dtype=np.int64),
                   indices=np.concatenate(indices) if indices else np.zeros(0, np.int64),
                   weights=np.concatenate(weights) if weights else np.zeros(0),
                   centre_pixel=centre, width=width, height=height)
