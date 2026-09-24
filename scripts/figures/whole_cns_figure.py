#!/usr/bin/env python3
"""Draw the whole-CNS measurement (``measure_whole_cns.py``) as two SVGs (light and dark) for the wiki: spikes per
superclass during a flash for the four engines (log scale), and the published and stabilised models' spike totals
over ten seeds.

Usage: python scripts/figures/whole_cns_figure.py RESULT_JSON OUT_DIR
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PALETTES = {
    "dark": {"bg": "#0d1117", "text": "#c9d1d9", "muted": "#8b949e", "grid": "#30363d",
             "E1": "#8b949e", "E2": "#58a6ff", "E3": "#3fb950", "E4": "#d29922", "pub": "#f85149", "stab": "#58a6ff"},
    "light": {"bg": "#ffffff", "text": "#1f2328", "muted": "#59636e", "grid": "#d0d7de",
              "E1": "#59636e", "E2": "#0969da", "E3": "#1a7f37", "E4": "#9a6700", "pub": "#d1242f", "stab": "#0969da"},
}
ROWS = (("visual_projection", "visual projection"), ("visual_centrifugal", "visual centrifugal"),
        ("cb_intrinsic", "central brain"), ("descending_neuron", "descending"), ("vnc_motor", "nerve-cord motor"))
W, H = 800, 440
X0, X1, LO, HI = 170, 500, 2, 6                          # bars: 10^2 to 10^6 spikes
YA0, YA1, VMIN, VMAX = 360, 80, 100_000, 600_000         # dots: spike totals


def sx(v: float) -> float:
    return X0 + (math.log10(max(v, 1)) - LO) / (HI - LO) * (X1 - X0)


def sy(v: float) -> float:
    return YA0 - (v - VMIN) / (VMAX - VMIN) * (YA0 - YA1)


def main() -> None:
    r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    for name, pal in PALETTES.items():
        style = (f".bg{{fill:{pal['bg']}}} .h{{font:600 15px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['text']}}}"
                 f" .m{{font:12px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['muted']}}}"
                 f" .b{{font:13px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{pal['text']}}}"
                 f" .grid{{stroke:{pal['grid']};stroke-width:1}}")
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
             f'aria-labelledby="t d"><title id="t">The four engines on the whole male CNS</title><desc id="d">Left: '
             f'spikes per superclass during a 300 ms full-field flash for engines E1 to E4, on a log scale; E1 has '
             f'none beyond the photoreceptors. Right: spikes in 500 ms under a strong gustatory drive over ten seeds, '
             f'spread over two states in the published model and gathered in one with the stabilisers.</desc>',
             f'<style>{style}</style>', f'<rect class="bg" width="{W}" height="{H}"/>',
             '<text class="h" x="24" y="30">Spikes during a 300 ms flash</text>']
        for d in range(LO, HI + 1):
            s.append(f'<line class="grid" x1="{sx(10 ** d):.1f}" y1="48" x2="{sx(10 ** d):.1f}" y2="372"/>')
            s.append(f'<text class="m" x="{sx(10 ** d):.1f}" y="388" text-anchor="middle">1e{d}</text>')
        y = 60
        for key, label in ROWS:
            s.append(f'<text class="b" x="{X0 - 8}" y="{y + 30}" text-anchor="end">{label}</text>')
            for k, eng in enumerate(("E1", "E2", "E3", "E4")):
                v = r["flash"][eng][key][1]
                yy = y + 6 + k * 12
                if v > 0:
                    s.append(f'<rect x="{X0}" y="{yy}" width="{max(sx(v) - X0, 1):.1f}" height="10" '
                             f'fill="{pal[eng]}"/>')
                else:
                    s.append(f'<text class="m" x="{X0 + 4}" y="{yy + 9}">0</text>')
            y += 62
        lx = 24
        for eng, label in (("E1", "E1 spiking everywhere"), ("E2", "E2 graded optic lobe"),
                           ("E3", "E3 flyvis per eye"), ("E4", "E4 stabilised")):
            s.append(f'<rect x="{lx}" y="{H - 30}" width="12" height="12" fill="{pal[eng]}"/>')
            s.append(f'<text class="b" x="{lx + 18}" y="{H - 20}">{label}</text>')
            lx += 175
        # right panel: bistability
        s.append('<text class="h" x="560" y="30">One state or two</text>')
        s.append('<text class="m" x="560" y="48">spikes in 500 ms, strong drive, 10 seeds</text>')
        for v in (200_000, 400_000, 600_000):
            s.append(f'<line class="grid" x1="600" y1="{sy(v):.1f}" x2="780" y2="{sy(v):.1f}"/>')
            s.append(f'<text class="m" x="594" y="{sy(v) + 4:.1f}" text-anchor="end">{v // 1000}k</text>')
        for k, (key, label, colour) in enumerate((("published", "published", pal["pub"]),
                                                  ("stabilised", "stabilised", pal["stab"]))):
            cx = 650 + k * 90
            for i, v in enumerate(r["bistability"][key]):
                s.append(f'<circle cx="{cx + (i % 5 - 2) * 7}" cy="{sy(v):.1f}" r="4" fill="{colour}"/>')
            s.append(f'<text class="b" x="{cx}" y="{YA0 + 22}" text-anchor="middle">{label}</text>')
        s.append("</svg>")
        (out / f"whole-cns-{name}.svg").write_text("\n".join(s) + "\n", encoding="utf-8", newline="\n")
    print("written")


if __name__ == "__main__":
    main()
