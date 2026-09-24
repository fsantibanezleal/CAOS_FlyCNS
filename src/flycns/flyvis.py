"""The flyvis pretrained ensemble, as data (Lappalainen et al., Nature 2024; flyvis 1.2.0, MIT).

The package carries the trained numbers of flyvis's 50 pretrained networks (``flow/0000``), extracted by
``scripts/extract_flyvis_ensemble.py`` in flyvis's own order: per cell type a resting potential and a time constant,
per pair of types a unitary strength, and the signs and mean synapse counts that training never changes. flyvis's
licence travels with them (``data/flyvis-1.2.0-ensemble/LICENSE-flyvis.txt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .compiled import Compiled, read_compiled

ENSEMBLE = "flyvis-1.2.0-ensemble"


@dataclass(frozen=True)
class FlyvisEnsemble:
    """The 50 networks' parameters and the orders that index them."""

    types: tuple[str, ...]          # 65 cell types, in flyvis's order
    pair_source: np.ndarray         # (604,) type index of each pair's presynaptic type
    pair_target: np.ndarray         # (604,) type index of each pair's postsynaptic type
    pair_sign: np.ndarray           # (604,) +1 or -1, fixed
    group_pair: np.ndarray          # (2355,) pair of each synapse-count group
    group_du: np.ndarray            # (2355,) columnar offset of the group
    group_dv: np.ndarray
    group_n_syn: np.ndarray         # (2355,) mean synapse count of the group, fixed
    bias: np.ndarray                # (50, 65) resting potential per network and type
    time_const_s: np.ndarray        # (50, 65) time constant per network and type, seconds
    strength: np.ndarray            # (50, 604) unitary strength per network and pair
    manifest: dict

    @property
    def n_models(self) -> int:
        return self.bias.shape[0]

    def type_index(self, name: str) -> int:
        return self.types.index(name)

    def pair_index(self) -> dict[tuple[str, str], int]:
        return {(self.types[s], self.types[t]): i
                for i, (s, t) in enumerate(zip(self.pair_source, self.pair_target, strict=True))}


def ensemble_directory() -> Path:
    return Path(str(resources.files("flycns") / "data" / ENSEMBLE))


def load_ensemble(directory: Path | None = None) -> FlyvisEnsemble:
    """Read the ensemble (the copy inside the package unless ``directory`` is given), every array hash-checked."""
    c: Compiled = read_compiled(directory or ensemble_directory())
    return FlyvisEnsemble(
        types=tuple(c.strings["type"]),
        pair_source=c["pair_source"].astype(np.int64),
        pair_target=c["pair_target"].astype(np.int64),
        pair_sign=c["pair_sign"].astype(np.int8),
        group_pair=c["group_pair"].astype(np.int64),
        group_du=c["group_du"].astype(np.int64),
        group_dv=c["group_dv"].astype(np.int64),
        group_n_syn=c["group_n_syn"].astype(np.float64),
        bias=c["bias"],
        time_const_s=c["time_const_s"],
        strength=c["strength"],
        manifest=c.manifest,
    )
