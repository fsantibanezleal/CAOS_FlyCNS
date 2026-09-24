#!/usr/bin/env python3
"""Draw the T4 and T5 preferred directions measured on MaleCNS for all 50 flyvis networks, as two SVGs (light and
dark) for ``docs/models/04_optic_lobe.md``.

One polar panel per subtype. The dashed spoke is the known preferred direction; each dot is one network on one eye,
at the angle of the subtype's DSI-weighted mean preferred direction and at a radius equal to its median DSI (the
outer ring is 0.5). Dots are coloured by whether that network's own cell of the subtype, on flyvis's lattice, is
selective (DSI above 0.1) with the known direction.

Usage: python scripts/figures/direction_figure.py RESULTS_DIR OUT_DIR
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from flycns.compiled import read_compiled
from flycns.motion import FLYVIS_KNOWN_PREFERRED_DEG, KNOWN_PREFERRED_DEG, T4_T5, angular_distance_deg

PALETTES = {
    "dark": {"bg": "#0d1117", "text": "#c9d1d9", "muted": "#8b949e", "grid": "#30363d", "yes": "#58a6ff",
             "no": "#d29922", "known": "#3fb950"},
    "light": {"bg": "#ffffff", "text": "#1f2328", "muted": "#59636e", "grid": "#d0d7de", "yes": "#0969da",
              "no": "#9a6700", "known": "#1a7f37"},
}
W, H = 800, 470
R = 72
RMAX = 0.5


def main() -> None:
    r = read_compiled(Path(sys.argv[1]), schema="flycns.optic-lobe-measure/1")
    out = Path(sys.argv[2])
    types = r.manifest["release"]["lattice_types"]
    keys = r.manifest["release"]["summary_keys"]
    k_pd, k_dsi = keys.index("mean_preferred_deg"), keys.index("median_dsi")
    for name, pal in PALETTES.items():
        style = (f".bg{{fill:{pal['bg']}}} .h{{font:600 15px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['text']}}}"
                 f" .m{{font:12px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['muted']}}}"
                 f" .b{{font:13px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['text']}}}"
                 f" .grid{{stroke:{pal['grid']};stroke-width:1;fill:none}}"
                 f" .known{{stroke:{pal['known']};stroke-width:2;stroke-dasharray:5 4}}")
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
                 f'aria-labelledby="t d"><title id="t">T4 and T5 preferred directions on MaleCNS, 50 flyvis '
                 f'networks</title><desc id="d">Eight polar panels, one per T4 and T5 subtype. Each shows the known '
                 f'preferred direction as a dashed spoke and, as dots, the direction each of the 50 transferred '
                 f'networks prefers on each eye, further out the more selective it is.</desc>',
                 f'<style>{style}</style>', f'<rect class="bg" width="{W}" height="{H}"/>']
        for j, t in enumerate(T4_T5):
            cx = 100 + (j % 4) * 200
            cy = 110 + (j // 4) * 200
            pol = 1 if t.startswith("T4") else 0
            i = types.index(t)
            selective = (r["lattice_dsi"][:, pol, i] > 0.1) & (
                angular_distance_deg(r["lattice_pd_deg"][:, pol, i], FLYVIS_KNOWN_PREFERRED_DEG[t]) <= 45)
            parts.append(f'<text class="h" x="{cx}" y="{cy - R - 14}" text-anchor="middle">{t}</text>')
            for rr in (0.25, 0.5):
                parts.append(f'<circle class="grid" cx="{cx}" cy="{cy}" r="{R * rr / RMAX:.1f}"/>')
                if j == 0:
                    parts.append(f'<text class="m" x="{cx + 3}" y="{cy + R * rr / RMAX + 13:.1f}">DSI {rr}</text>')
            a = math.radians(KNOWN_PREFERRED_DEG[t])
            parts.append(f'<line class="known" x1="{cx}" y1="{cy}" x2="{cx + R * math.cos(a):.1f}" '
                         f'y2="{cy - R * math.sin(a):.1f}"/>')
            for e in (0, 1):
                pd = r["malecns_summary"][:, e, j, k_pd]
                dsi = np.minimum(r["malecns_summary"][:, e, j, k_dsi], RMAX)
                for m in np.argsort(selective):
                    ang = math.radians(float(pd[m]))
                    rad = R * float(dsi[m]) / RMAX
                    colour = pal["yes"] if selective[m] else pal["no"]
                    parts.append(f'<circle cx="{cx + rad * math.cos(ang):.1f}" cy="{cy - rad * math.sin(ang):.1f}" '
                                 f'r="2.6" fill="{colour}" fill-opacity="0.85"/>')
        y = H - 22
        parts.append(f'<line class="known" x1="24" y1="{y - 4}" x2="52" y2="{y - 4}"/>')
        parts.append(f'<text class="b" x="60" y="{y}">known direction</text>')
        parts.append(f'<circle cx="200" cy="{y - 4}" r="4" fill="{pal["yes"]}"/>')
        parts.append(f'<text class="b" x="210" y="{y}">network selective on flyvis\'s lattice</text>')
        parts.append(f'<circle cx="470" cy="{y - 4}" r="4" fill="{pal["no"]}"/>')
        parts.append(f'<text class="b" x="480" y="{y}">not selective there</text>')
        parts.append(f'<text class="m" x="{W - 24}" y="{y}" text-anchor="end">0 deg: front to back; 90: up</text>')
        parts.append("</svg>")
        (out / f"t4t5-directions-{name}.svg").write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print("written")


if __name__ == "__main__":
    main()
