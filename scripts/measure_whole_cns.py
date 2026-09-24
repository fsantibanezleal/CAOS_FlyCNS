#!/usr/bin/env python3
"""Measure the four engines on the whole MaleCNS, and what the bridge's free parameter changes.

- A full-field flash (200 ms grey, 300 ms full intensity on both eyes) through E1 (the published LIF everywhere,
  light as Poisson input to photoreceptors), E2 (the transferred graded optic lobes plus LIF), E3 (flyvis's own
  network per eye mapped onto MaleCNS, plus LIF) and E4 (E2 with the stabilisers): spikes per superclass, during
  grey and during the flash.
- The published model and the stabilised one under the strong gustatory drive, ten seeds each: spikes in 500 ms.
- E2's flash for bridge gains of 25 to 400 spikes per second per unit of release.

Writes one JSON file. Needs a CUDA device; about five minutes.

Usage: python scripts/measure_whole_cns.py COMPILED GEOMETRY_JSON FLYVIS_EXTRACT OUT_JSON
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from flycns.compiled import read_compiled
from flycns.dynamics import Drive, LIFParams, LIFTorch, synaptic_weights
from flycns.dynamics.hybrid import HybridTorch, build_hybrid, lattice_hybrid
from flycns.dynamics.lif import photoreceptor_drive, stabilised_weights
from flycns.eyes import build_eyes
from flycns.flyvis import lattice_network, load_ensemble
from flycns.mapped import map_to_lattice
from flycns.optic_lobe import build_optic_lobe, unit_directions

SUPERCLASSES = ("ol_sensory", "visual_projection", "visual_centrifugal", "cb_intrinsic", "descending_neuron",
                "ascending_neuron", "vnc_intrinsic", "vnc_motor")
GREY, FLASH = 40, 60
SPLIT = GREY * 50


def per_superclass(steps, neurons, superclass) -> dict:
    names = superclass[neurons]
    return {k: [int(((steps < SPLIT) & (names == k)).sum()), int(((steps >= SPLIT) & (names == k)).sum())]
            for k in SUPERCLASSES}


def stabilise(hybrid, params) -> None:
    """E4's weights on the hybrid's spiking part."""
    rows = np.repeat(np.arange(hybrid.n_spiking), np.diff(hybrid.lif_indptr))
    signs = np.zeros(hybrid.n_spiking)
    signs[rows] = np.sign(hybrid.lif_weights_mv)
    counts = np.abs(hybrid.lif_weights_mv) / params.w_syn_mv
    hybrid.lif_weights_mv = stabilised_weights(hybrid.lif_indptr, hybrid.lif_indices, counts, signs,
                                               hybrid.types_spiking, params.w_syn_mv)


def main() -> None:
    compiled, geometry, extract, out = (Path(a) for a in sys.argv[1:5])
    graph = read_compiled(compiled, verify=False)
    ensemble = load_ensemble()
    superclass = np.array(graph.strings["superclass"], dtype=object)[graph["neuron_superclass"]]
    types = np.array(graph.strings["type"], dtype=object)[graph["neuron_type"]]
    n_cols = len(graph["column_side"])
    frames = np.r_[np.full((GREY, n_cols), 0.5), np.full((FLASH, n_cols), 1.0)].astype(np.float32)
    result = {"flash": {}, "bridge_gain": {}, "bistability": {}, "frames": {"grey_ms": GREY * 5, "flash_ms": FLASH * 5}}

    t0 = time.time()
    lobe = build_optic_lobe(graph, ensemble, 0)
    hybrid = build_hybrid(graph, lobe)
    result["hybrid"] = hybrid.report
    run = HybridTorch(hybrid, device="cuda").run(frames)
    steps, units = run.spikes.spike_times()
    result["flash"]["E2"] = per_superclass(steps, hybrid.spiking_neuron[units], superclass)

    p4 = LIFParams(adaptation_mv=1.5)
    hybrid4 = build_hybrid(graph, lobe, params=p4)
    hybrid4.types_spiking = graph["neuron_type"][hybrid4.spiking_neuron]
    stabilise(hybrid4, p4)
    run = HybridTorch(hybrid4, device="cuda").run(frames)
    steps, units = run.spikes.spike_times()
    result["flash"]["E4"] = per_superclass(steps, hybrid4.spiking_neuron[units], superclass)

    lattice = read_compiled(extract / "lattice-000")
    eyes = build_eyes(graph["column_side"], graph["column_hex"], graph["column_kind"],
                      json.loads(geometry.read_text(encoding="utf-8")))
    column_dirs = np.full((n_cols, 3), np.nan)
    for side in ("left", "right"):
        column_dirs[eyes[side].column_index] = eyes[side].directions
    directions, _ = unit_directions(lobe, column_dirs)
    mapping = map_to_lattice(lobe, lattice, ensemble.types, directions, eyes)
    result["lattice_mapping"] = mapping.report
    net = lattice_network(lattice, ensemble, 0)
    lattice_frames = np.r_[np.full((GREY, 2, 721), 0.5), np.full((FLASH, 2, 721), 1.0)].astype(np.float32)
    run = lattice_hybrid(hybrid, [net, net], mapping, device="cuda").run(lattice_frames)
    steps, units = run.spikes.spike_times()
    result["flash"]["E3"] = per_superclass(steps, hybrid.spiking_neuron[units], superclass)

    column = graph["neuron_column"].astype(np.int64)
    photo = np.flatnonzero(np.array([isinstance(t, str) and (t == "R1-R6" or t.startswith(("R7", "R8")))
                                     for t in types]) & (column >= 0))
    weights = synaptic_weights(graph["csr_indptr"], graph["csr_indices"], graph["csr_count"], graph["neuron_sign"],
                               0.275)
    engine = LIFTorch(graph["csr_indptr"], graph["csr_indices"], weights, device="cuda")
    run = engine.run((GREY + FLASH) * 50, photoreceptor_drive(photo, column[photo], frames), seed=0)
    steps, neurons = run.spike_times()
    result["flash"]["E1"] = per_superclass(steps, neurons, superclass)
    result["flash"]["E1"]["optic_lobe_beyond_photoreceptors"] = int((~np.isin(neurons, photo)).sum())

    for gain in (25.0, 50.0, 100.0, 200.0, 400.0):
        h = build_hybrid(graph, lobe, bridge_gain_hz=gain)
        run = HybridTorch(h, device="cuda").run(frames)
        steps, units = run.spikes.spike_times()
        result["bridge_gain"][str(gain)] = per_superclass(steps, h.spiking_neuron[units], superclass)

    classes = np.array(graph.strings["class"], dtype=object)[graph["neuron_class"]]
    drive = Drive(activate={int(i): 150.0 for i in np.flatnonzero(classes == "gustatory")})
    args = (graph["csr_indptr"], graph["csr_indices"], graph["csr_count"], graph["neuron_sign"])
    for name, w, params in (("published", synaptic_weights(*args, 0.275), LIFParams()),
                            ("stabilised", stabilised_weights(*args, graph["neuron_type"], 0.275),
                             LIFParams(adaptation_mv=1.5))):
        eng = LIFTorch(graph["csr_indptr"], graph["csr_indices"], w, params, device="cuda")
        result["bistability"][name] = [int(eng.run(5000, drive, seed=s).spike_counts().sum()) for s in range(10)]
    result["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("flash", "bistability")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
