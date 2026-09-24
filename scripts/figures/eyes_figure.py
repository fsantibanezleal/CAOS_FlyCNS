#!/usr/bin/env python3
"""Draw the right eye's columns in azimuth and elevation, before and after the dorsal-rim orientation, as two SVGs
(light and dark) for ``docs/models/01_eyes.md``. Every point is a column of MaleCNS v1.0; the colours are the column
kinds compiled from the photoreceptor subtypes.

Usage: python scripts/figures/eyes_figure.py COMPILED GEOMETRY_JSON OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from flycns.compiled import read_compiled
from flycns.eyes import build_eyes

PALETTES = {
    "dark": {"bg": "#0d1117", "text": "#c9d1d9", "muted": "#8b949e", "grid": "#30363d", "rim": "#f85149",
             "pale": "#58a6ff", "yellow": "#d29922", "unknown": "#484f58"},
    "light": {"bg": "#ffffff", "text": "#1f2328", "muted": "#59636e", "grid": "#d0d7de", "rim": "#d1242f",
              "pale": "#0969da", "yellow": "#9a6700", "unknown": "#afb8c1"},
}
W, H, PAD = 800, 400, 24
PANEL_W, PANEL_H = 360, 300


def panel(x0, y0, eye, kinds, title, pal):
    az, el = eye.azimuth_deg, eye.elevation_deg
    sx = lambda a: x0 + 40 + (a + 40) / 220 * (PANEL_W - 50)       # noqa: E731  azimuth -40..180
    sy = lambda e: y0 + 30 + (95 - e) / 190 * (PANEL_H - 60)       # noqa: E731  elevation 95..-95
    out = [f'<text class="h" x="{x0 + 40}" y="{y0 + 16}">{title}</text>']
    for a in (0, 45, 90, 135, 180):
        out.append(f'<line class="grid" x1="{sx(a):.1f}" y1="{sy(95):.1f}" x2="{sx(a):.1f}" y2="{sy(-95):.1f}"/>')
        out.append(f'<text class="m" x="{sx(a):.1f}" y="{sy(-95) + 16:.1f}" text-anchor="middle">{a}</text>')
    for e in (-90, -45, 0, 45, 90):
        out.append(f'<line class="grid" x1="{sx(-40):.1f}" y1="{sy(e):.1f}" x2="{sx(180):.1f}" y2="{sy(e):.1f}"/>')
        out.append(f'<text class="m" x="{sx(-40) - 6:.1f}" y="{sy(e) + 4:.1f}" text-anchor="end">{e}</text>')
    order = np.argsort(kinds == "dorsal_rim")                     # the rim drawn last, on top
    for i in order:
        k = kinds[i] if kinds[i] in ("pale", "yellow", "dorsal_rim") else "unknown"
        r = 3.2 if k == "dorsal_rim" else 2.2
        colour = pal["rim"] if k == "dorsal_rim" else pal[k]
        out.append(f'<circle cx="{sx(az[i]):.1f}" cy="{sy(el[i]):.1f}" r="{r}" fill="{colour}"/>')
    return out


def main() -> None:
    compiled, geometry, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    graph = read_compiled(compiled, verify=False)
    geo = json.loads(geometry.read_text(encoding="utf-8"))
    kinds_all = np.array(graph.strings["column_kind"], dtype=object)
    before = build_eyes(graph["column_side"], graph["column_hex"], graph["column_kind"], geo, dorsal_rim_code=None)
    after = build_eyes(graph["column_side"], graph["column_hex"], graph["column_kind"], geo)
    for name, pal in PALETTES.items():
        style = (f".bg{{fill:{pal['bg']}}} .h{{font:600 15px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['text']}}}"
                 f" .m{{font:12px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['muted']}}}"
                 f" .b{{font:13px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['text']}}}"
                 f" .grid{{stroke:{pal['grid']};stroke-width:1}}")
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
                 f'aria-labelledby="t d"><title id="t">The right eye of MaleCNS, before and after the dorsal-rim '
                 f'orientation</title><desc id="d">Each dot is one of the 892 columns of the right eye, placed at its '
                 f'modelled azimuth and elevation. Red dots are the dorsal-rim columns. Left: the lattice with the '
                 f'medulla\'s dorsal axis vertical, the rim runs diagonally from the front of the equator to the top '
                 f'back. Right: turned by 60 degrees, the rim lies along the top of the eye.</desc>',
                 f'<style>{style}</style>', f'<rect class="bg" width="{W}" height="{H}"/>']
        eye_b, eye_a = before["right"], after["right"]
        parts += panel(PAD, PAD, eye_b, kinds_all[eye_b.kinds], "medulla axis vertical (0.02 to 0.04)", pal)
        parts += panel(PAD + PANEL_W + 30, PAD, eye_a, kinds_all[eye_a.kinds], "turned by the dorsal rim (0.05)", pal)
        y = PAD + PANEL_H + 40
        x = PAD + 40
        for label, key in (("dorsal rim", "rim"), ("pale", "pale"), ("yellow", "yellow"), ("unknown", "unknown")):
            parts.append(f'<circle cx="{x}" cy="{y - 4}" r="4" fill="{pal[key]}"/>')
            parts.append(f'<text class="b" x="{x + 10}" y="{y}">{label}</text>')
            x += 110
        parts.append(f'<text class="m" x="{PAD + 490}" y="{y}">azimuth (deg, toward the back), elevation (deg)</text>')
        parts.append("</svg>")
        (out / f"eyes-orientation-{name}.svg").write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print("written")


if __name__ == "__main__":
    main()
