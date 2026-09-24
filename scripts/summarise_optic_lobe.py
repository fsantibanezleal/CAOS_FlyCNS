#!/usr/bin/env python3
"""Summarise ``measure_optic_lobe.py``'s results: per T4/T5 subtype, how often the transfer keeps what flyvis's own
network does, and how stable the transferred networks are. Prints the numbers the wiki quotes.

Usage: python scripts/summarise_optic_lobe.py RESULTS_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from flycns.compiled import read_compiled
from flycns.motion import FLYVIS_KNOWN_PREFERRED_DEG, KNOWN_PREFERRED_DEG, T4_T5, angular_distance_deg

SELECTIVE_DSI = 0.1
WITHIN_DEG = 45.0


def main() -> None:
    r = read_compiled(Path(sys.argv[1]), schema="flycns.optic-lobe-measure/1")
    types = r.manifest["release"]["lattice_types"]
    keys = r.manifest["release"]["summary_keys"]
    k_err, k_dsi = keys.index("mean_error_deg"), keys.index("median_dsi")
    n = len(r["models"])
    print(f"networks: {n}")
    rows = []
    for j, t in enumerate(T4_T5):
        pol = 1 if t.startswith("T4") else 0
        i = types.index(t)
        lat_ok = (r["lattice_dsi"][:, pol, i] > SELECTIVE_DSI) & (
            angular_distance_deg(r["lattice_pd_deg"][:, pol, i], FLYVIS_KNOWN_PREFERRED_DEG[t]) <= WITHIN_DEG)
        mal_err = r["malecns_summary"][:, :, j, k_err]             # (networks, eyes)
        mal_dsi = r["malecns_summary"][:, :, j, k_dsi]
        mal_ok = (mal_err <= WITHIN_DEG).all(axis=1)
        both = lat_ok & mal_ok
        rows.append((t, int(lat_ok.sum()), int(mal_ok.sum()), int(both.sum()),
                     float(np.median(mal_err[lat_ok])) if lat_ok.any() else float("nan"),
                     float(np.median(mal_dsi)), float(np.median(r["lattice_dsi"][:, pol, i]))))
    print("subtype | lattice selective with the known direction | MaleCNS within 45 deg on both eyes | both | "
          "MaleCNS median error where lattice selective | MaleCNS median DSI | lattice median DSI")
    for row in rows:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.0f} | {row[5]:.3f} | {row[6]:.3f}")
    for group in ("T4", "T5"):
        g = [row for row in rows if row[0].startswith(group)]
        lat, both = sum(x[1] for x in g), sum(x[3] for x in g)
        print(f"{group}: of {lat} (network, subtype) cases selective on the lattice, {both} ({both / max(lat, 1):.0%}) "
              f"are within 45 deg on both MaleCNS eyes; MaleCNS within 45 deg overall: {sum(x[2] for x in g)} of "
              f"{len(g) * n}")
    s = r["stability_max_abs_v"]                                    # (networks, lattice/MaleCNS, marks)
    for k, name in enumerate(("lattice", "MaleCNS")):
        v20 = s[:, k, -1]
        finite = np.isfinite(v20)
        print(f"{name}: finite at 20 s {int(finite.sum())} of {n}; below 1e3 {int((v20 < 1e3).sum())}; "
              f"grows more than twofold from 5 to 20 s {int((s[:, k, -1] > 2 * s[:, k, 1]).sum())}; "
              f"median max|V| {np.nanmedian(v20):.1f}")
    both_bounded = (s[:, 0, -1] < 1e3) & (s[:, 1, -1] < 1e3)
    print(f"bounded in both: {int(both_bounded.sum())}; bounded on the lattice only: "
          f"{int(((s[:, 0, -1] < 1e3) & ~(s[:, 1, -1] < 1e3)).sum())}; on MaleCNS only: "
          f"{int((~(s[:, 0, -1] < 1e3) & (s[:, 1, -1] < 1e3)).sum())}")
    print("known directions used (anatomical frame):", KNOWN_PREFERRED_DEG)


if __name__ == "__main__":
    main()
