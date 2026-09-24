"""The eye model on a synthetic lattice whose orientation is known, so every rule can be checked exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flycns.eyes import (
    EQUATOR_BACK_DEG,
    EQUATOR_FRONT_DEG,
    body_axes,
    build_eyes,
    eye_plane,
    lattice_orientation,
    sampler,
)

# Landmark centroids measured on MaleCNS v1.0 (micrometres, release frame), used as a realistic test input.
LANDMARKS = {
    "AL(R)": [328.5, 228.6, 126.5], "AL(L)": [446.8, 213.3, 120.7], "CA(R)": [302.5, 111.4, 269.9],
    "CA(L)": [470.7, 103.6, 265.2], "PB": [387.5, 128.8, 256.5], "GNG": [389.8, 321.5, 221.7],
    "ME(R)": [136.2, 276.0, 267.0], "ME(L)": [640.0, 260.8, 257.4],
}


def hex_patch(radius: int = 12, centre=(18, 19)) -> np.ndarray:
    """A hexagon-shaped patch of axial coordinates whose third neighbour pair is (1, 1) (axes 120 degrees apart)."""
    cells = []
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            if abs(a - b) <= radius:           # hexagon in the (1,0), (0,1), (1,1) neighbour system
                cells.append((centre[0] + a, centre[1] + b))
    return np.array(cells, dtype=np.int64)


def synthetic_geometry(hexes: np.ndarray, side_tag: str, axes: dict) -> dict:
    """Medulla column centres for a lattice where +hex2 runs dorsal and +hex1 posterior-and-down (120 degrees)."""
    step = 7.0
    e2 = step * axes["dorsal"]
    e1 = step * (math.cos(math.radians(210)) * axes["anterior"] + math.sin(math.radians(210)) * axes["dorsal"])
    origin = np.array([100.0, 250.0, 260.0])
    centres = {f"{side_tag}_{h1}_{h2}": list(origin + h1 * e1 + h2 * e2) + [1000] for h1, h2 in hexes}
    return centres


@pytest.fixture
def geometry_and_columns():
    axes = body_axes({k: np.array(v) for k, v in LANDMARKS.items()})
    hexes = hex_patch()
    centres = synthetic_geometry(hexes, "R", axes)
    centres.update(synthetic_geometry(hexes, "L", axes))
    geometry = {"landmarks_um": LANDMARKS, "medulla_columns_um": centres}
    column_side = np.r_[np.full(len(hexes), 1), np.full(len(hexes), 2)].astype(np.uint8)
    column_hex = np.r_[hexes, hexes].astype(np.int16)
    column_kind = np.zeros(len(column_side), dtype=np.uint8)
    return geometry, column_side, column_hex, column_kind, axes, hexes


def test_body_axes_from_landmarks_are_right_handed():
    axes = body_axes({k: np.array(v) for k, v in LANDMARKS.items()})
    frame = np.stack([axes["anterior"], axes["left"], axes["dorsal"]])
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-9)
    assert np.dot(np.cross(axes["anterior"], axes["left"]), axes["dorsal"]) == pytest.approx(1.0)
    assert axes["left"][0] > 0.99                 # +x is the fly's left in the release frame


def test_lattice_orientation_is_recovered_from_column_centres(geometry_and_columns):
    geometry, _, _, _, axes, hexes = geometry_and_columns
    centres = np.array([geometry["medulla_columns_um"][f"R_{a}_{b}"][:3] for a, b in hexes])
    orientation = lattice_orientation(hexes, centres, axes)
    assert orientation["vertical_axis"] == 1                  # +hex2 is the dorsal axis
    assert orientation["hex_steps"][1]["dorsal"] > 6.9
    assert orientation["hex_steps"][0]["anterior"] < -6.0      # +hex1 runs posterior in the medulla
    assert set(map(tuple, orientation["neighbour_offsets"])) == {(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)}


def test_chiasm_mirrors_anterior_posterior_only(geometry_and_columns):
    geometry, _, _, _, axes, hexes = geometry_and_columns
    centres = np.array([geometry["medulla_columns_um"][f"R_{a}_{b}"][:3] for a, b in hexes])
    orientation = lattice_orientation(hexes, centres, axes)
    plane = eye_plane(np.array([[1, 0], [0, 1]]), orientation)
    assert plane[0, 0] > 0                  # +hex1 ran posterior in the medulla, so it looks FORWARD in the eye
    assert plane[1, 1] > 0 and plane[1, 0] == pytest.approx(0.0)   # +hex2 ran dorsal and still looks up
    np.testing.assert_allclose(np.linalg.norm(plane, axis=1), 1.0)
    assert np.linalg.norm(plane[0] + plane[1]) == pytest.approx(1.0)   # (1, 1) is one step: 120 degrees apart


def test_equator_spans_the_measured_extent(geometry_and_columns):
    geometry, column_side, column_hex, column_kind, *_ = geometry_and_columns
    eyes = build_eyes(column_side, column_hex, column_kind, geometry)
    for side in ("left", "right"):
        eye = eyes[side]
        plane = eye_plane(eye.hexes.astype(np.int64), eye.orientation)
        near = np.abs(plane[:, 1] - np.median(plane[:, 1])) <= 0.5
        assert eye.azimuth_deg[near].min() == pytest.approx(EQUATOR_FRONT_DEG, abs=1e-6)
        assert eye.azimuth_deg[near].max() == pytest.approx(EQUATOR_BACK_DEG, abs=1e-6)
        np.testing.assert_allclose(eye.elevation_deg[near & (plane[:, 1] == np.median(plane[:, 1]))], 0.0, atol=1e-6)


def test_neighbours_are_evenly_spaced_and_eyes_mirror(geometry_and_columns):
    geometry, column_side, column_hex, column_kind, *_ = geometry_and_columns
    eyes = build_eyes(column_side, column_hex, column_kind, geometry)
    right, left = eyes["right"], eyes["left"]
    angles = []
    for i, row in enumerate(right.neighbours):
        for j in row[row >= 0]:
            angles.append(math.degrees(math.acos(np.clip(right.directions[i] @ right.directions[j], -1, 1))))
    angles = np.array(angles) / right.delta_phi_deg
    # The azimuthal-equidistant placement keeps every column's angular distance from the eye's centre exact, so
    # radial neighbours are one step apart; tangential neighbours at distance rho from the centre are compressed by
    # sin(rho)/rho (0.69 at the ends of a 165-degree equator, lower toward the corners). Real eyes also vary their
    # spacing across the eye (Zhao et al. 2025); the bound states what this model promises.
    assert np.median(angles) == pytest.approx(1.0, abs=0.1)
    assert angles.max() <= 1.05
    assert angles.min() >= 0.5
    mirrored = right.directions * np.array([1.0, -1.0, 1.0])
    np.testing.assert_allclose(left.directions, mirrored, atol=1e-9)
    assert right.directions[np.argmin(right.azimuth_deg)][1] > 0   # the right eye's front edge looks past the midline


def test_sampling_rows_sum_to_one_and_uniform_light_stays_uniform(geometry_and_columns):
    geometry, column_side, column_hex, column_kind, *_ = geometry_and_columns
    eye = build_eyes(column_side, column_hex, column_kind, geometry)["right"]
    s = sampler(eye, width=360, height=180)
    sums = np.add.reduceat(s.weights, s.indptr[:-1])
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    np.testing.assert_allclose(s.sample(np.full((180, 360), 0.7)), 0.7, atol=1e-9)


def test_a_bright_spot_lights_the_ommatidium_that_looks_at_it(geometry_and_columns):
    geometry, column_side, column_hex, column_kind, *_ = geometry_and_columns
    eye = build_eyes(column_side, column_hex, column_kind, geometry)["right"]
    width, height = 720, 360
    s = sampler(eye, width, height)
    target = len(eye.directions) // 3
    panorama = np.zeros((height, width))
    panorama.reshape(-1)[s.centre_pixel[target]] = 1.0
    response = s.sample(panorama)
    assert int(np.nanargmax(response)) == target
