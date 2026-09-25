"""The committed parity fixtures are the reference's current output, and the scenarios that produced them.

Regenerates every scenario and its expected output in a temporary directory and compares the array hashes with the
committed manifests: a change to a reference engine or a scenario builder fails here until
``scripts/make_parity_fixtures.py`` is run again and the result committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flycns.bundle import read_scenario, run_reference, write_expected, write_scenario
from flycns.parity import SCENARIOS

PARITY = Path(__file__).resolve().parents[1] / "parity"


def hashes(directory: Path) -> dict[str, str]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return {a["name"]: a["sha256"] for a in manifest["arrays"]}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_committed_fixtures_are_the_references_current_output(name, tmp_path):
    committed = PARITY / name
    assert (committed / "manifest.json").is_file(), f"{name} is not under parity/: run scripts/make_parity_fixtures.py"
    # the scenario builder still produces the committed scenario, byte for byte
    write_scenario(tmp_path / "scenario", SCENARIOS[name]())
    assert hashes(tmp_path / "scenario") == hashes(committed)
    # and the reference, run on the committed scenario, still gives the committed expected output
    outputs = run_reference(read_scenario(committed))
    write_expected(tmp_path / "expected", outputs)
    assert hashes(tmp_path / "expected") == hashes(committed / "expected")
    assert outputs and all(v.size > 0 for k, v in outputs.items() if k in ("neuron_index", "activity", "graded"))
