"""The MaleCNS v1.0 release adapter.

MaleCNS v1.0 is the complete connectome of the male *Drosophila melanogaster* central nervous system: 166,700
proofread neurons in brain and ventral nerve cord, 11,710 cell types (Berg et al., "Sexual dimorphism in the complete
Drosophila male central nervous system connectome", Cell 189:5504-5526, 2026, doi:10.1016/j.cell.2026.08.015).
Data: CC BY 4.0, https://male-cns.janelia.org/download/.

Every rule applied here is a pure function below, so the synthetic tests exercise the same code as the compilation of
the real tables. The rules are specified in ``docs/design/features/malecns-compiler/design.md``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..compiled import write_compiled
from .base import SourceTable, check_table, iter_batches

RELEASE = {"name": "MaleCNS", "version": "v1.0", "citation": "Berg et al., Cell 189:5504-5526 (2026)",
           "doi": "10.1016/j.cell.2026.08.015", "license": "CC BY 4.0",
           "download_page": "https://male-cns.janelia.org/download/"}

_BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"

#: The official tables, locked. The first three hashes were checked against two independent public lock files
#: (DOOMFLY and mps-malecns-model); the synapse table against the bucket's own MD5.
TABLES = {
    "annotations": SourceTable("annotations", "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                               _BASE_URL + "body-annotations-male-cns-v1.0-minconf-0.5.feather", 14483314,
                               "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"),
    "transmitters": SourceTable("transmitters", "body-neurotransmitters-male-cns-v1.0.feather",
                                _BASE_URL + "body-neurotransmitters-male-cns-v1.0.feather", 43282834,
                                "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"),
    "weights": SourceTable("weights", "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                           _BASE_URL + "connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1051241946,
                           "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"),
    "synapses": SourceTable("synapses", "syn-points-male-cns-v1.0-minconf-0.5.feather",
                            _BASE_URL + "syn-points-male-cns-v1.0-minconf-0.5.feather", 13061489098,
                            "c16b1b63186c4d4f28939decea7444451f0f5f6f7ef1bb5dab5b2a7058f8f284"),
}

#: Voxel size of the release coordinates, micrometres. In these coordinates +x runs toward the fly's LEFT
#: (measured: neurons named ``_L`` have larger x than those named ``_R``).
VOXEL_UM = 0.008

TRANSMITTERS = ["unclear", "acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "octopamine", "serotonin"]
#: Shiu et al. 2024: GABA and glutamate inhibitory, dopamine, octopamine and serotonin excitatory. Histamine, absent
#: from FlyWire, is inhibitory: it is the photoreceptors' transmitter, acting through histamine-gated chloride channels.
SIGN_BY_TRANSMITTER = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
                       "dopamine": 1, "octopamine": 1, "serotonin": 1}
NT_SOURCES = ["consensus", "predicted", "none"]
SIDES = ["unknown", "left", "right", "midline"]
PARTITIONS = ["optic_lobe_left", "optic_lobe_right", "central_brain", "nerve_cord"]
POSITION_SOURCES = ["soma", "to_soma", "synapse_centroid", "none"]
COLUMN_KINDS = ["unknown", "pale", "yellow", "dorsal_rim"]
COUNT_MAX = 65535


# ---------------------------------------------------------------------------------------------------------- rules


def retained(annotations):
    """The neurons a simulation runs: rows with a superclass, sorted by body ID, each body once."""
    kept = annotations[annotations["superclass"].notna()].sort_values("bodyId", kind="stable")
    if kept["bodyId"].duplicated().any():
        duplicates = kept.loc[kept["bodyId"].duplicated(), "bodyId"].tolist()[:5]
        raise ValueError(f"body IDs repeated in the annotations: {duplicates}")
    return kept.reset_index(drop=True)


def resolve_transmitter(consensus, predicted) -> tuple[str, int]:
    """The transmitter label and which field supplied it (consensus first, then the per-body prediction)."""
    for source, value in ((0, consensus), (1, predicted)):
        if isinstance(value, str) and value in SIGN_BY_TRANSMITTER:
            return value, source
    return "unclear", 2


def sign_of(transmitter: str) -> int:
    """+1, -1 or 0 by the rule above; unclear or unknown transmitters carry no weight."""
    return SIGN_BY_TRANSMITTER.get(transmitter, 0)


def side_of(soma_side, root_side, instance) -> int:
    """Side code: the soma side, else the root side, else the ``_L`` / ``_R`` suffix of the instance name."""
    for value in (soma_side, root_side):
        if value == "L":
            return 1
        if value == "R":
            return 2
        if value in ("M", "C"):
            return 3
    if isinstance(instance, str):
        if instance.endswith("_L"):
            return 1
        if instance.endswith("_R"):
            return 2
    return 0


def side_from_x(x_um: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """Fill unknown sides from position: the midline lies halfway between the mean x of left and right neurons.

    +x is the fly's left in the release coordinates, so a neuron beyond the midline toward larger x is on the left.
    """
    sides = sides.copy()
    known_left = (sides == 1) & np.isfinite(x_um)
    known_right = (sides == 2) & np.isfinite(x_um)
    if not known_left.any() or not known_right.any():
        return sides
    midline = 0.5 * (x_um[known_left].mean() + x_um[known_right].mean())
    unknown = (sides == 0) & np.isfinite(x_um)
    sides[unknown & (x_um > midline)] = 1
    sides[unknown & (x_um <= midline)] = 2
    return sides


def partition_of(superclass: str, side: int) -> int:
    """Optic lobe of its side for optic-lobe neurons and photoreceptors, nerve cord for ``vnc*``, else central."""
    if superclass in ("ol_intrinsic", "ol_sensory"):
        if side == 1:
            return 0
        if side == 2:
            return 1
        return 2
    if superclass.startswith("vnc"):
        return 3
    return 2


def positions(soma, to_soma, centroid_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions in micrometres and their source: soma, else to-soma point, else synapse centroid, else none.

    ``soma`` and ``to_soma`` are sequences of ``[x, y, z]`` voxel triples or missing values; ``centroid_um`` holds the
    synapse centroid of each neuron in micrometres, NaN where the neuron has no synapse.
    """
    n = len(soma)
    out = np.full((n, 3), np.nan, dtype=np.float64)
    source = np.full(n, 3, dtype=np.uint8)
    for i in range(n):
        for code, value in ((0, soma[i]), (1, to_soma[i])):
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                triple = np.asarray(value, dtype=np.float64)
                if triple.shape == (3,) and np.all(np.isfinite(triple)):
                    out[i] = triple * VOXEL_UM
                    source[i] = code
                    break
        else:
            if np.all(np.isfinite(centroid_um[i])):
                out[i] = centroid_um[i]
                source[i] = 2
    return out.astype(np.float32), source


def lookup(sorted_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of each value in ``sorted_ids``, or -1 where the value is absent."""
    index = np.searchsorted(sorted_ids, values)
    index = np.minimum(index, len(sorted_ids) - 1)
    found = sorted_ids[index] == values
    return np.where(found, index, -1)


def build_csr(pre: np.ndarray, post: np.ndarray, weight: np.ndarray, n: int):
    """Signed-agnostic CSR by presynaptic index, duplicate pairs summed, counts saturated at 65,535.

    Returns ``(indptr int64[n+1], indices int32[E], counts uint16[E], n_saturated)``; targets ascend within a row.
    """
    key = pre.astype(np.int64) * n + post.astype(np.int64)
    unique, inverse = np.unique(key, return_inverse=True)
    summed = np.bincount(inverse, weights=weight.astype(np.float64), minlength=len(unique))
    saturated = int(np.count_nonzero(summed > COUNT_MAX))
    counts = np.minimum(np.rint(summed), COUNT_MAX).astype(np.uint16)
    rows = (unique // n).astype(np.int64)
    indices = (unique % n).astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr, indices, counts, saturated


def build_columns(sides: np.ndarray, hex1: np.ndarray, hex2: np.ndarray):
    """Column table from the hex assignments: sorted by (side, hex1, hex2); the column of every assigned neuron."""
    assigned = np.isfinite(hex1) & np.isfinite(hex2) & ((sides == 1) | (sides == 2))
    triples = np.stack([sides[assigned], hex1[assigned], hex2[assigned]], axis=1).astype(np.int64)
    table, inverse = np.unique(triples, axis=0, return_inverse=True)
    neuron_column = np.full(len(sides), -1, dtype=np.int32)
    neuron_column[np.flatnonzero(assigned)] = inverse.reshape(-1).astype(np.int32)
    return table[:, 0].astype(np.uint8), table[:, 1:].astype(np.int16), neuron_column


def is_photoreceptor(type_name) -> bool:
    return isinstance(type_name, str) and (type_name == "R1-R6" or type_name.startswith(("R7", "R8")))


def assign_photoreceptors(receptors: np.ndarray, indptr, indices, counts, neuron_column, sides,
                          column_side: np.ndarray) -> np.ndarray:
    """Give each photoreceptor the column whose neurons receive most of its synapses (same side only).

    Photoreceptor terminals carry no hex assignment in the release. R1-R6 terminals synapse onto the lamina neurons
    of one cartridge, R7 and R8 onto the medulla neurons of one column, so the column that receives the largest summed
    synapse count identifies the ommatidium the terminal belongs to. Ties go to the lowest column index.
    """
    out = neuron_column.copy()
    for p in receptors:
        targets = indices[indptr[p]:indptr[p + 1]]
        weights = counts[indptr[p]:indptr[p + 1]].astype(np.int64)
        columns = neuron_column[targets]
        keep = columns >= 0
        keep[keep] = column_side[columns[keep]] == sides[p]
        if not keep.any():
            continue
        totals = np.bincount(columns[keep], weights=weights[keep])
        out[p] = int(np.argmax(totals))
    return out


def column_kinds(n_columns: int, neuron_column: np.ndarray, types: list) -> tuple[np.ndarray, int]:
    """Pale, yellow or dorsal-rim from the column's R7/R8 subtypes (``p``, ``y``, ``d``); majority wins.

    Returns the kind codes and the number of columns whose photoreceptors disagreed.
    """
    votes = np.zeros((n_columns, 4), dtype=np.int64)
    for i, name in enumerate(types):
        column = neuron_column[i]
        if column < 0 or not isinstance(name, str) or not name.startswith(("R7", "R8")) or name == "R7R8_unclear":
            continue
        suffix = name[2:]
        code = {"p": 1, "y": 2, "d": 3}.get(suffix)
        if code is not None:
            votes[column, code] += 1
    kinds = np.zeros(n_columns, dtype=np.uint8)
    has = votes[:, 1:].sum(axis=1) > 0
    kinds[has] = (1 + np.argmax(votes[has, 1:], axis=1)).astype(np.uint8)
    conflicts = int(np.count_nonzero((votes[:, 1:] > 0).sum(axis=1) > 1))
    return kinds, conflicts


# ------------------------------------------------------------------------------------------------------ compile


def _string_table(values) -> tuple[list[str], np.ndarray]:
    labels = ["" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in values]
    table = sorted(set(labels))
    position = {label: i for i, label in enumerate(table)}
    return table, np.array([position[label] for label in labels], dtype=np.uint32)


def synapse_centroids(path: Path, need_ids: np.ndarray, progress: Callable[[str], None] | None = None) -> np.ndarray:
    """Mean position (micrometres) of all synapse points of each body in ``need_ids`` (sorted); NaN if none."""
    sums = np.zeros((len(need_ids), 3), dtype=np.float64)
    hits = np.zeros(len(need_ids), dtype=np.float64)
    if len(need_ids) == 0:
        return np.full((0, 3), np.nan)
    for b, batch in enumerate(iter_batches(path, ["x", "y", "z", "body"])):
        index = lookup(need_ids, batch["body"])
        valid = index >= 0
        if valid.any():
            idx = index[valid]
            hits += np.bincount(idx, minlength=len(need_ids))
            for axis, name in enumerate(("x", "y", "z")):
                sums[:, axis] += np.bincount(idx, weights=batch[name][valid].astype(np.float64),
                                             minlength=len(need_ids))
        if progress and b % 500 == 0:
            progress(f"synapse points: batch {b}")
    with np.errstate(invalid="ignore"):
        centroid = sums / hits[:, None]
    return centroid * VOXEL_UM


def stream_edges(path: Path, ids: np.ndarray, progress: Callable[[str], None] | None = None):
    """The connections whose two ends are retained neurons, as index arrays plus weights."""
    pres, posts, weights = [], [], []
    for b, batch in enumerate(iter_batches(path, ["body_pre", "body_post", "weight"])):
        pre = lookup(ids, batch["body_pre"])
        post = lookup(ids, batch["body_post"])
        keep = (pre >= 0) & (post >= 0)
        if keep.any():
            pres.append(pre[keep].astype(np.int32))
            posts.append(post[keep].astype(np.int32))
            weights.append(batch["weight"][keep].astype(np.int64))
        if progress and b % 500 == 0:
            progress(f"weights: batch {b}")
    if not pres:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64)
    return np.concatenate(pres), np.concatenate(posts), np.concatenate(weights)


def compile_malecns_v1(tables_dir: Path, out_dir: Path, *, tables: dict[str, SourceTable] | None = None,
                       verify_hash: bool = True, synapse_centroids_for_missing: bool = True,
                       progress: Callable[[str], None] | None = None) -> dict:
    """Compile the MaleCNS v1.0 tables in ``tables_dir`` into the compiled directory ``out_dir``.

    ``tables`` replaces the locked table descriptions (the synthetic tests use it); the real compilation uses
    :data:`TABLES`. Returns the manifest.
    """
    import pandas as pd

    tables = tables or TABLES
    say = progress or (lambda message: None)
    started = time.time()
    paths = {}
    for key in ("annotations", "transmitters", "weights"):
        paths[key] = check_table(tables_dir, tables[key], verify_hash=verify_hash)
    if synapse_centroids_for_missing:
        paths["synapses"] = check_table(tables_dir, tables["synapses"], verify_hash=verify_hash)
    say("tables verified")

    annotations = retained(pd.read_feather(paths["annotations"]))
    ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    n = len(ids)
    say(f"retained neurons: {n}")

    transmitters = pd.read_feather(paths["transmitters"], columns=["body", "consensus_nt", "predicted_nt"])
    transmitters = transmitters.drop_duplicates("body").set_index("body")
    consensus = transmitters["consensus_nt"].reindex(ids).astype(object).to_numpy()
    predicted = transmitters["predicted_nt"].reindex(ids).astype(object).to_numpy()
    resolved = [resolve_transmitter(c, p) for c, p in zip(consensus, predicted, strict=True)]
    nt_labels = [label for label, _ in resolved]
    neuron_nt = np.array([TRANSMITTERS.index(label) for label in nt_labels], dtype=np.uint8)
    nt_source = np.array([source for _, source in resolved], dtype=np.uint8)
    neuron_sign = np.array([sign_of(label) for label in nt_labels], dtype=np.int8)

    soma = annotations["somaLocation"].to_numpy(dtype=object)
    to_soma = annotations["tosomaLocation"].to_numpy(dtype=object)
    missing = np.array([
        (s is None or (isinstance(s, float) and np.isnan(s))) and (t is None or (isinstance(t, float) and np.isnan(t)))
        for s, t in zip(soma, to_soma, strict=True)
    ])
    centroid = np.full((n, 3), np.nan)
    if synapse_centroids_for_missing and missing.any():
        centroid[missing] = synapse_centroids(paths["synapses"], ids[missing], progress=say)
    position, position_source = positions(soma, to_soma, centroid)
    say("positions assigned")

    sides = np.array([side_of(s, r, i) for s, r, i in zip(annotations["somaSide"], annotations["rootSide"],
                                                             annotations["instance"], strict=True)], dtype=np.uint8)
    sides = side_from_x(position[:, 0].astype(np.float64), sides)
    superclasses = annotations["superclass"].astype(str).to_list()
    partition = np.array([partition_of(sc, int(sd)) for sc, sd in zip(superclasses, sides, strict=True)],
                         dtype=np.uint8)

    pre, post, weight = stream_edges(paths["weights"], ids, progress=say)
    indptr, indices, counts, saturated = build_csr(pre, post, weight, n)
    say(f"edges: {len(indices)}")

    hex1 = annotations["assignedOlHex1"].to_numpy(dtype=np.float64)
    hex2 = annotations["assignedOlHex2"].to_numpy(dtype=np.float64)
    column_side, column_hex, neuron_column = build_columns(sides, hex1, hex2)
    types = annotations["type"].to_list()
    receptors = np.array([i for i, t in enumerate(types) if is_photoreceptor(t)], dtype=np.int64)
    neuron_column = assign_photoreceptors(receptors, indptr, indices, counts, neuron_column, sides, column_side)
    kinds, conflicts = column_kinds(len(column_side), neuron_column, types)

    type_table, type_index = _string_table(types)
    class_table, class_index = _string_table(annotations["class"].to_list())
    superclass_table, superclass_index = _string_table(superclasses)

    arrays = {
        "neuron_body_id": ids,
        "neuron_type": type_index,
        "neuron_class": class_index,
        "neuron_superclass": superclass_index.astype(np.uint8),
        "neuron_side": sides,
        "neuron_partition": partition,
        "neuron_nt": neuron_nt,
        "neuron_nt_source": nt_source,
        "neuron_sign": neuron_sign,
        "neuron_position_um": position,
        "neuron_position_source": position_source,
        "neuron_column": neuron_column,
        "csr_indptr": indptr,
        "csr_indices": indices,
        "csr_count": counts,
        "column_side": column_side,
        "column_hex": column_hex,
        "column_kind": kinds,
    }
    assigned_receptors = int(np.count_nonzero(neuron_column[receptors] >= 0)) if len(receptors) else 0
    counts_meta = {
        "neurons": n,
        "edges": int(len(indices)),
        "synapses_in_edges": int(counts.astype(np.int64).sum()),
        "saturated_counts": saturated,
        "self_edges": int(np.count_nonzero(pre == post)),
        "position_source": {POSITION_SOURCES[k]: int(np.count_nonzero(position_source == k)) for k in range(4)},
        "transmitter": {TRANSMITTERS[k]: int(np.count_nonzero(neuron_nt == k)) for k in range(len(TRANSMITTERS))},
        "transmitter_source": {NT_SOURCES[k]: int(np.count_nonzero(nt_source == k)) for k in range(3)},
        "sign": {"excitatory": int(np.count_nonzero(neuron_sign > 0)), "inhibitory": int(np.count_nonzero(neuron_sign < 0)),
                 "none": int(np.count_nonzero(neuron_sign == 0))},
        "side": {SIDES[k]: int(np.count_nonzero(sides == k)) for k in range(4)},
        "partition": {PARTITIONS[k]: int(np.count_nonzero(partition == k)) for k in range(4)},
        "columns": {"left": int(np.count_nonzero(column_side == 1)), "right": int(np.count_nonzero(column_side == 2))},
        "column_kind": {COLUMN_KINDS[k]: int(np.count_nonzero(kinds == k)) for k in range(4)},
        "column_kind_conflicts": conflicts,
        "photoreceptors": int(len(receptors)),
        "photoreceptors_assigned_to_a_column": assigned_receptors,
    }
    meta = {
        "release": RELEASE,
        "sources": [tables[key].describe() for key in sorted(paths)],
        "counts": counts_meta,
        "strings": {
            "type": type_table, "class": class_table, "superclass": superclass_table, "transmitter": TRANSMITTERS,
            "nt_source": NT_SOURCES, "side": SIDES, "partition": PARTITIONS, "position_source": POSITION_SOURCES,
            "column_kind": COLUMN_KINDS,
        },
    }
    say(f"writing the compiled directory ({time.time() - started:.0f} s so far)")
    return write_compiled(out_dir, arrays, meta)
