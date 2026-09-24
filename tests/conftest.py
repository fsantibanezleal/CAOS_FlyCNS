"""A synthetic MaleCNS-shaped release, small enough to compile in a test and built to exercise every rule.

Bodies (voxel coordinates; +x is the fly's left, so right-side bodies sit at small x):

- 100, 101, 102: right optic lobe columnar neurons (L1 at hex 5,5 and 5,6; L2 at hex 5,5), side from the instance
- 103: a left optic lobe L1 at hex 5,5
- 200: an R1-R6 terminal on the right, no soma (its centroid comes from synapse points), histamine; drives 100 most
- 201: an R7p on the right driving 101; 202: an R8y on the right also driving 101 (a pale/yellow conflict)
- 300: a GABAergic central neuron on the left, with a soma
- 301: a descending neuron, no soma but a to-soma point, transmitter only from the per-body prediction
- 302: a dopaminergic neuron; 303: a neuron absent from the transmitter table
- 304: a central neuron with no soma, no to-soma point and no synapse points (position source: none)
- 400: a nerve-cord motor neuron, glutamatergic, midline
- 900: a fragment without a superclass (dropped, with its edges)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flycns.release.base import SourceTable


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.reset_index(drop=True).to_feather(path)


def _table(key: str, path: Path) -> SourceTable:
    data = path.read_bytes()
    return SourceTable(key, path.name, f"file://{path.name}", len(data), hashlib.sha256(data).hexdigest())


@pytest.fixture
def synthetic_release(tmp_path: Path):
    none = None
    rows = [
        # bodyId, superclass, class, type, instance, somaSide, rootSide, soma, tosoma, hex1, hex2
        (100, "ol_intrinsic", None, "L1", "L1_R", None, None, None, None, 5.0, 5.0),
        (101, "ol_intrinsic", None, "L1", "L1_R", None, None, None, None, 5.0, 6.0),
        (102, "ol_intrinsic", None, "L2", "L2_R", None, None, None, None, 5.0, 5.0),
        (103, "ol_intrinsic", None, "L1", "L1_L", None, None, None, None, 5.0, 5.0),
        (200, "ol_sensory", "visual", "R1-R6", "R1-R6_R", None, None, None, None, np.nan, np.nan),
        (201, "ol_sensory", "visual", "R7p", "R7p_R", None, None, None, None, np.nan, np.nan),
        (202, "ol_sensory", "visual", "R8y", "R8y_R", None, None, None, None, np.nan, np.nan),
        (300, "cb_intrinsic", None, "CB0001", "CB0001_L", "L", None, [70000, 30000, 30000], None, np.nan, np.nan),
        (301, "descending_neuron", None, "DNp01", "DNp01_R", None, "R", None, [20000, 31000, 40000], np.nan, np.nan),
        (302, "cb_intrinsic", None, "PAM01", "PAM01_L", "L", None, [65000, 20000, 25000], None, np.nan, np.nan),
        (303, "cb_intrinsic", None, "CB0002", "CB0002_R", "R", None, [30000, 20000, 25000], None, np.nan, np.nan),
        (304, "cb_intrinsic", None, "CB0003", "CB0003_R", None, "R", None, None, np.nan, np.nan),
        (400, "vnc_motor", None, "MN1", "MN1_M", "M", None, [48000, 40000, 120000], None, np.nan, np.nan),
        (900, None, None, None, None, None, None, None, None, np.nan, np.nan),
    ]
    annotations = pd.DataFrame(rows, columns=["bodyId", "superclass", "class", "type", "instance", "somaSide",
                                              "rootSide", "somaLocation", "tosomaLocation", "assignedOlHex1",
                                              "assignedOlHex2"])
    annotations["somaLocation"] = [none if v is None else np.array(v, dtype=np.int64)
                                   for v in annotations["somaLocation"]]
    annotations["tosomaLocation"] = [none if v is None else np.array(v, dtype=np.int64)
                                     for v in annotations["tosomaLocation"]]
    transmitters = pd.DataFrame({
        "body": [100, 101, 102, 103, 200, 201, 202, 300, 301, 302, 304, 400, 900],
        "consensus_nt": ["glutamate", "glutamate", "acetylcholine", "glutamate", "histamine", "histamine",
                         "histamine", "gaba", "unclear", "dopamine", "acetylcholine", "glutamate", "gaba"],
        "predicted_nt": ["glutamate", "glutamate", "acetylcholine", "glutamate", "histamine", "histamine",
                         "histamine", "gaba", "acetylcholine", "dopamine", "acetylcholine", "glutamate", "gaba"],
    })
    weights = pd.DataFrame({
        "body_pre": [200, 200, 201, 202, 300, 300, 301, 301, 100, 900, 302],
        "body_post": [100, 101, 101, 101, 301, 301, 400, 900, 300, 300, 300],
        "weight": [30, 5, 20, 10, 7, 8, 70000, 4, 3, 50, 2],
    })
    synapses = pd.DataFrame({
        "x": np.array([20000, 22000, 21000, 60000], dtype=np.int32),
        "y": np.array([30000, 30000, 33000, 10000], dtype=np.int32),
        "z": np.array([10000, 10000, 13000, 10000], dtype=np.int32),
        "kind": ["PreSyn", "PreSyn", "PostSyn", "PreSyn"],
        "body": np.array([200, 200, 200, 900], dtype=np.int64),
    })
    names = {
        "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "transmitters": "body-neurotransmitters-male-cns-v1.0.feather",
        "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "synapses": "syn-points-male-cns-v1.0-minconf-0.5.feather",
    }
    frames = {"annotations": annotations, "transmitters": transmitters, "weights": weights, "synapses": synapses}
    tables = {}
    for key, frame in frames.items():
        path = tmp_path / names[key]
        _write(frame, path)
        tables[key] = _table(key, path)
    return tmp_path, tables
