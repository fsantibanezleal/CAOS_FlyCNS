#!/usr/bin/env python3
"""Record flyvis's own moving-edge experiment: the stimuli, the responses of five networks, and flyvis's measures.

Direction selectivity is the claim that matters most for a visual system built from the connectome: T4 cells respond
to ON edges moving in one of four directions and T5 cells to OFF edges, and flyvis's networks reproduce it. flycns
measures direction selectivity with its own code, first on flyvis's lattice and then on the MaleCNS optic lobes; this
script records flyvis's measurement so flycns's can be held to it.

Run in the flyvis environment of ``extract_flyvis_ensemble.py``. It writes ``moving-edges/`` in the compiled style:

- ``stimulus`` (samples x frames x 721 columns) and the sample table (angle in degrees, intensity 0 for OFF and 1 for
  ON, speed in columns per second of 5.8 degrees), from flyvis's ``MovingEdge`` dataset with the configuration of
  flyvis's ``moving_edge_responses`` restricted to the speeds 13, 19 and 25;
- ``responses`` (networks x samples x frames x 65 central neurons, one per cell type) for networks 000 to 004, as
  flyvis computes them;
- ``dsi`` and ``preferred_direction_rad`` (networks x intensity x 65 types), from flyvis's own
  ``direction_selectivity_index`` and ``preferred_direction``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_flyvis_ensemble import windows_safe_datamate  # noqa: E402

SPEEDS = (13, 19, 25)
N_NETWORKS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    windows_safe_datamate()
    from flyvis import NetworkView
    from flyvis.analysis.moving_bar_responses import direction_selectivity_index, preferred_direction
    from flyvis.analysis.stimulus_responses import moving_edge_responses

    from flycns.compiled import write_compiled

    datasets = [moving_edge_responses(NetworkView(f"flow/0000/{m:03d}"), speeds=SPEEDS, batch_size=24)
                for m in range(N_NETWORKS)]
    first = datasets[0]
    stimulus = first["stimulus"].values[:, :, 0, :].astype(np.float32)
    for ds in datasets[1:]:
        assert np.array_equal(ds["stimulus"].values[:, :, 0, :], first["stimulus"].values[:, :, 0, :], equal_nan=True)
    responses = np.concatenate([ds["responses"].values for ds in datasets], axis=0).astype(np.float32)
    cell_types = [str(t) for t in first["cell_type"].values]

    dsi = np.zeros((N_NETWORKS, 2, len(cell_types)), dtype=np.float32)
    pd = np.zeros((N_NETWORKS, 2, len(cell_types)), dtype=np.float32)
    for m, ds in enumerate(datasets):
        d = direction_selectivity_index(ds)          # averaged over width and speed
        p = preferred_direction(ds)
        for k, intensity in enumerate((0, 1)):
            dsi[m, k] = d.sel(intensity=intensity).values.reshape(-1)
            pd[m, k] = p.sel(intensity=intensity).values.reshape(-1)

    write_compiled(args.out / "moving-edges", {
        "stimulus": stimulus,
        "sample_angle_deg": first["angle"].values.astype(np.float32),
        "sample_intensity": first["intensity"].values.astype(np.float32),
        "sample_speed": first["speed"].values.astype(np.float32),
        "time_s": first["time"].values.astype(np.float32),
        "responses": responses,
        "dsi": dsi,
        "preferred_direction_rad": pd,
    }, {"release": {"model": "flyvis", "version": "1.2.0", "ensemble": "flow/0000",
                    "networks": [f"{m:03d}" for m in range(N_NETWORKS)],
                    "dataset": {k: v for k, v in first.attrs["config"].items() if isinstance(v, (int, float, str, list))},
                    "speeds": list(SPEEDS), "measure": "flyvis.analysis.moving_bar_responses"},
        "counts": {"samples": int(stimulus.shape[0]), "frames": int(stimulus.shape[1]), "networks": N_NETWORKS},
        "strings": {"cell_type": cell_types}})
    print(f"wrote {args.out / 'moving-edges'}: {stimulus.shape[0]} samples x {stimulus.shape[1]} frames; "
          f"{N_NETWORKS} networks")


if __name__ == "__main__":
    main()
