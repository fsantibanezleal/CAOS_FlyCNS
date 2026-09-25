#!/usr/bin/env python3
"""Write the parity fixtures under parity/: every scenario of ``flycns.parity`` and the reference's expected output.

Usage: python scripts/make_parity_fixtures.py [OUT_DIR]      (default: parity/ at the repository root)

Run it after any change to the reference engines or the scenarios; ``tests/test_parity_fixtures.py`` fails until the
committed fixtures are the reference's current output again.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from flycns.bundle import run_reference, write_expected, write_scenario
from flycns.parity import SCENARIOS


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, build in SCENARIOS.items():
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        scenario = build()
        write_scenario(target, scenario)
        outputs = run_reference(scenario)
        write_expected(target / "expected", outputs, {"scenario": name})
        spikes = int(outputs["neuron_index"].shape[0]) if "neuron_index" in outputs else 0
        print(f"{name}: {scenario.engine}, {scenario.steps} steps, {spikes} spikes")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "parity")
