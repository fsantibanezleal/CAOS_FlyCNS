#!/usr/bin/env python3
"""Measure the transferred optic lobe against flyvis's own networks, all 50 of them.

For each pretrained flyvis network, on flyvis's lattice and on the MaleCNS optic lobes built by
``flycns.optic_lobe.build_optic_lobe``:

- **direction selectivity**: on the lattice, flyvis's own moving-edge stimuli (the extraction's ``moving-edges``) and
  flyvis's measures for the 65 central neurons; on MaleCNS, full-field ON and OFF edges in 12 directions and three
  speeds across each modelled eye, and the same measures for every T4 and T5 cell within 15 degrees of the eye's
  centre;
- **stability**: the largest absolute voltage after 2, 5, 10 and 20 s of uniform grey.

Writes a compiled-style directory (``flycns.optic-lobe-measure/1``) and prints a summary. Needs a CUDA device; about
a minute per network.

Usage::

    python scripts/measure_optic_lobe.py COMPILED FLYVIS_EXTRACT GEOMETRY_JSON OUT_DIR [--models 0-49]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flycns.compiled import read_compiled, write_compiled
from flycns.dynamics.graded import GradedTorch
from flycns.eyes import build_eyes
from flycns.flyvis import central_neurons, lattice_network, load_ensemble
from flycns.motion import T4_T5, direction_selectivity, flyvis_peaks, selectivity_on_eye
from flycns.optic_lobe import build_optic_lobe, unit_directions

DT = 1 / 200
MARKS_S = (2, 5, 10, 20)


def parse_models(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def long_run(engine) -> list[float]:
    v, t_prev, out = None, 0, []
    for t in MARKS_S:
        v = engine.steady_state(t - t_prev, 0.5, initial=None if v is None else v.cpu().numpy())
        t_prev = t
        out.append(float(v.abs().max()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("compiled", type=Path)
    parser.add_argument("extract", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", default="0-49")
    args = parser.parse_args()

    graph = read_compiled(args.compiled, verify=False)
    ensemble = load_ensemble()
    lattice = read_compiled(args.extract / "lattice-000")
    edges = read_compiled(args.extract / "moving-edges")
    eyes = build_eyes(graph["column_side"], graph["column_hex"], graph["column_kind"],
                      json.loads(args.geometry.read_text(encoding="utf-8")))
    column_dirs = np.full((len(graph["column_side"]), 3), np.nan)
    for side in ("left", "right"):
        column_dirs[eyes[side].column_index] = eyes[side].directions
    centre = central_neurons(lattice)
    models = parse_models(args.models)

    lat_dsi = np.zeros((len(models), 2, 65), dtype=np.float32)
    lat_pd = np.zeros((len(models), 2, 65), dtype=np.float32)
    summary_keys = ("median_dsi", "within_45_of_known", "mean_preferred_deg", "mean_error_deg", "neurons")
    mal = np.zeros((len(models), 2, len(T4_T5), len(summary_keys)), dtype=np.float32)
    stability = np.zeros((len(models), 2, len(MARKS_S)), dtype=np.float32)
    per_neuron = {k: [] for k in ("model", "eye", "type", "x_deg", "y_deg", "dsi_off", "dsi_on", "pd_off", "pd_on")}
    reports = []
    for i, m in enumerate(models):
        t0 = time.time()
        engine = GradedTorch(lattice_network(lattice, ensemble, m), DT, device="cuda")
        steady = engine.steady_state(1.0, 0.5).cpu().numpy()
        out = np.concatenate([engine.run_batch(edges["stimulus"][b:b + 24], steady, centre)
                              for b in range(0, len(edges["stimulus"]), 24)])
        sel = direction_selectivity(flyvis_peaks(out, edges["time_s"], edges["sample_speed"]),
                                    edges["sample_angle_deg"], edges["sample_intensity"], edges["sample_speed"])
        lat_dsi[i], lat_pd[i] = sel.dsi.T, np.degrees(sel.preferred_rad.T) % 360
        stability[i, 0] = long_run(engine)

        lobe = build_optic_lobe(graph, ensemble, m)
        reports.append(lobe.report)
        dirs, _ = unit_directions(lobe, column_dirs)
        engine = GradedTorch(lobe.network, DT, device="cuda")
        steady = engine.steady_state(2.0, 0.5).cpu().numpy()
        for e, side in enumerate(("left", "right")):
            res = selectivity_on_eye(engine, lobe, eyes[side], steady, dirs, lobe.side)
            s = res.summary()
            for j, t in enumerate(T4_T5):
                if t in s:
                    mal[i, e, j] = [s[t][k] for k in summary_keys]
            for k, arr in (("model", np.full(len(res.unit), m)), ("eye", np.full(len(res.unit), e)),
                           ("type", np.array([T4_T5.index(t) for t in res.unit_type])), ("x_deg", res.x_deg),
                           ("y_deg", res.y_deg), ("dsi_off", res.dsi[:, 0]), ("dsi_on", res.dsi[:, 1]),
                           ("pd_off", res.preferred_deg[:, 0]), ("pd_on", res.preferred_deg[:, 1])):
                per_neuron[k].append(arr)
        stability[i, 1] = long_run(engine)
        print(f"network {m:02d}: {time.time() - t0:.0f}s; lattice T4 DSI {np.round(lat_dsi[i, 1, 35:39], 2)}; "
              f"MaleCNS T4 mean error (left, right) {np.round(mal[i, :, :4, 3], 0).tolist()}; "
              f"stability (lattice, MaleCNS) at 20 s {stability[i, :, -1].round(2).tolist()}", flush=True)

    arrays = {"models": np.array(models, dtype=np.int32), "lattice_dsi": lat_dsi, "lattice_pd_deg": lat_pd,
              "malecns_summary": mal, "stability_max_abs_v": stability}
    arrays.update({f"neuron_{k}": np.concatenate(v).astype(np.float32 if k not in ("model", "eye", "type")
                                                              else np.int32) for k, v in per_neuron.items()})
    write_compiled(args.out, arrays, {
        "release": {"measure": "optic-lobe transfer against flyvis", "dt_s": DT, "stability_marks_s": list(MARKS_S),
                    "summary_keys": list(summary_keys), "t4_t5": list(T4_T5),
                    "lattice_types": list(ensemble.types)},
        "counts": {"models": len(models)}, "strings": {}}, schema="flycns.optic-lobe-measure/1")
    (args.out / "transfer-reports.json").write_text(json.dumps(reports, indent=1, default=float) + "\n",
                                                   encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
