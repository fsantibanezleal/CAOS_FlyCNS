"""The compiled MaleCNS v1.0 against the release's published numbers.

Needs a compiled directory (``FLYCNS_MALECNS_COMPILED``, default ``E:/_Datos/destello/compiled/malecns-v1.0``); the
test is skipped, and listed by ``pytest -rs``, when it is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flycns.compiled import read_compiled
from flycns.release.malecns_v1 import TABLES

COMPILED = Path(os.environ.get("FLYCNS_MALECNS_COMPILED", "E:/_Datos/destello/compiled/malecns-v1.0"))


@pytest.mark.data
@pytest.mark.skipif(not (COMPILED / "manifest.json").is_file(), reason=f"no compiled MaleCNS at {COMPILED}")
def test_compiled_malecns_matches_the_release():
    compiled = read_compiled(COMPILED)
    assert compiled.n_neurons == 166_700                       # Berg et al. 2026
    # The same connection and synapse totals are published by independent projects that filter the release the
    # same way (neurons with a superclass): 25,582,938 directed connections (FLYBOARD, DOOMFLY, Xenova's simulation)
    # and 124,177,617 synaptic contacts (mps-malecns-model).
    assert compiled.n_edges == 25_582_938
    assert compiled.counts["synapses_in_edges"] == 124_177_617
    assert compiled.counts["columns"] == {"left": 879, "right": 892}
    # flyverse places 5,895 photoreceptors on the release's columns by its own method; this compiler reaches the same
    assert compiled.counts["photoreceptors_assigned_to_a_column"] == 5_895
    sources = {entry["key"]: entry["sha256"] for entry in compiled.manifest["sources"]}
    for key, table in TABLES.items():
        if key in sources:
            assert sources[key] == table.sha256
    types = compiled.strings["type"]
    names = np.array(types, dtype=object)[compiled["neuron_type"]]
    photoreceptor = np.array([isinstance(t, str) and (t == "R1-R6" or t.startswith(("R7", "R8"))) for t in names])
    assert photoreceptor.sum() > 5000
    signs = compiled["neuron_sign"][photoreceptor]
    assert np.mean(signs == -1) > 0.95                          # histaminergic: inhibitory
    assert compiled.counts["photoreceptors_assigned_to_a_column"] > 0.9 * photoreceptor.sum()
