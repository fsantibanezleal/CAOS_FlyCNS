"""Both MaleCNS eyes built from the compiled release and the geometry measured on its synapse table.

Skipped (and listed by ``pytest -rs``) when the compiled directory or the measured geometry is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from flycns.compiled import read_compiled
from flycns.eyes import EQUATOR_FRONT_DEG, build_eyes

COMPILED = Path(os.environ.get("FLYCNS_MALECNS_COMPILED", "E:/_Datos/destello/compiled/malecns-v1.0"))
GEOMETRY = Path(str(COMPILED) + "-eyes-geometry.json")


@pytest.mark.data
@pytest.mark.skipif(not ((COMPILED / "manifest.json").is_file() and GEOMETRY.is_file()),
                    reason=f"no compiled MaleCNS or measured eye geometry beside {COMPILED}")
def test_malecns_eyes_match_the_release_and_the_measured_extent():
    compiled = read_compiled(COMPILED)
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    eyes = build_eyes(compiled["column_side"], compiled["column_hex"], compiled["column_kind"], geometry)
    assert len(eyes["right"].column_index) == 892
    assert len(eyes["left"].column_index) == 879
    kinds = np.array(compiled.strings["column_kind"], dtype=object)
    for side in ("left", "right"):
        eye = eyes[side]
        names = kinds[eye.kinds]
        rim = eye.elevation_deg[names == "dorsal_rim"].mean()
        colour = eye.elevation_deg[(names == "pale") | (names == "yellow")].mean()
        assert rim > colour + 10.0
        assert eye.azimuth_deg.min() <= EQUATOR_FRONT_DEG
        assert 4.0 < eye.delta_phi_deg < 7.0      # the literature's inter-ommatidial angles lie in this range
