"""The graded optic lobes of MaleCNS: flyvis's trained numbers on the release's own neurons and synapses.

flyvis (Lappalainen et al., Nature 2024) trained a resting potential and a time constant per cell type and a unitary
strength per synapse for each pair of types, on one averaged column. Here the same numbers drive the neuron-level
wiring of both MaleCNS optic lobes: every optic-lobe intrinsic neuron, every photoreceptor and every neuron of a type
flyvis models becomes a graded unit, and every synapse between them keeps its count. The rules, stated with their
measurements in ``docs/design/features/graded/design.md``:

1. **Classes.** A unit's transfer class is its MaleCNS type, except that photoreceptors pool into R1-R6, R7 and R8
   (the release does not tell R1 from R6, and its R7 and R8 subtypes are one flyvis type each; the few R7R8_unclear
   stand for both).
2. **Mapping.** A class maps to the flyvis types it stands for: the same name for 49 types, and the merges of
   ``CLASS_TO_FLYVIS`` for the rest. Classes flyvis does not model are unmapped.
3. **Neurons.** A mapped class takes the mean resting potential and time constant of its flyvis types; an unmapped
   class takes flyvis's initial resting potential (0.5) and the network's median trained time constant.
4. **Connections of a mapped pair.** Each synapse carries flyvis's trained strength for the pair, with flyvis's
   sign (where a class stands for several presynaptic flyvis types, R1-R6, their count-weighted mean). MaleCNS and
   flyvis count synapses on the same scale: over the pairs both have, the median ratio of MaleCNS's synapses per
   target to flyvis's, weighted by flyvis's drive, is 1.08. But flyvis's seven-column reconstructions under-count
   lateral connections, and trained strengths on the release's denser recurrence drive the network to infinity
   (TmY4 onto TmY4, 7 times denser in MaleCNS, within 100 ms; with a cap on each pair's average, T5d within 700 ms,
   through the neurons that receive several times their class's average). So each neuron's drive from each
   presynaptic class is capped at flyvis's: where a neuron receives ``c`` synapses from a class for which flyvis's
   column has ``N``, and ``c > N``, those synapses are scaled by ``N / c``. No neuron is driven harder than its
   flyvis counterpart by any class; where MaleCNS gives fewer synapses, the connection keeps flyvis's strength and is
   weaker, as the release has it.
5. **CT1 compartments.** flyvis models CT1 as one medulla (M10) and one lobula (Lo1) compartment per column,
   electrically separate; MaleCNS has one CT1 per side. Each CT1 connection moves to the compartment of its
   partner's column (the partner's own, else the column that feeds it most), of the kind flyvis pairs the partner's
   type with. Without the split, T5 cells lose their local inhibition and most of their direction selectivity.
6. **Every other connection** (an unmapped class on either side, or a mapped pair flyvis has no connection for) gets
   flyvis's initialisation scale: each neuron receives from each such presynaptic class a total drive of
   ``rho x K`` (``rho = 0.01``, ``K = 2``, flyvis's median number of columnar offsets per pair), spread over that
   class's synapses onto it, with the sign of each presynaptic neuron's transmitter.
7. **Stand-in photoreceptors.** Where a column's reconstructed photoreceptors deliver fewer synapses onto a columnar
   neuron of that column than the median of complete columns (six R1-R6, one R7, one R8), one stand-in unit per
   column and group supplies the difference, sees the column's light, and is flagged.
8. **Light** reaches every photoreceptor with a column, real or stand-in, as the intensity of its column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compiled import Compiled
from .dynamics.graded import GradedNetwork
from .flyvis import FlyvisEnsemble

PHOTORECEPTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "R1-R6": ("R1-R6",),
    "R7": ("R7p", "R7y", "R7d", "R7_unclear"),
    "R8": ("R8p", "R8y", "R8d", "R8_unclear"),
    "R7R8": ("R7R8_unclear",),
}
FULL_COLUMN = {"R1-R6": 6, "R7": 1, "R8": 1}
CT1_COMPARTMENTS = ("CT1(M10)", "CT1(Lo1)")

CLASS_TO_FLYVIS: dict[str, tuple[str, ...]] = {
    "R1-R6": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "R7": ("R7",),
    "R8": ("R8",),
    "R7R8": ("R7", "R8"),
    "TmY9a": ("TmY9",),
    "TmY9b": ("TmY9",),
}


@dataclass(frozen=True)
class TransferRules:
    default_bias: float = 0.5        # flyvis's initial resting potential (the mean of its initial distribution)
    default_rho: float = 0.01        # flyvis's initial strength scale: rho / <n> per synapse
    default_offsets: float = 2.0     # flyvis's median number of columnar offsets per pair of types
    transfer: str = "capped"        # "capped": at most flyvis's drive per class and neuron; "exact": exactly
    stand_ins: bool = True
    ct1_compartments: bool = True    # split each CT1 into flyvis's per-column M10 and Lo1 compartments
    include_unmapped: bool = True    # False keeps only the classes flyvis models


@dataclass
class OpticLobe:
    """The graded optic lobes: the network, and for every unit where it came from."""

    network: GradedNetwork
    neuron: np.ndarray               # compiled neuron of each unit; -1 for a stand-in
    unit_class: np.ndarray           # index into ``classes``
    classes: tuple[str, ...]
    column: np.ndarray               # compiled column of each unit; -1 where it has none
    side: np.ndarray                 # "left", "right", "midline" or "unknown"
    stand_in: np.ndarray             # bool
    mapped: np.ndarray               # bool: the unit's class maps to flyvis types
    transferred: np.ndarray          # bool per connection: flyvis's strength (rule 4), else the default (rule 6)
    model: int
    report: dict = field(default_factory=dict)

    def units_of(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.unit_class == self.classes.index(name))


def _class_of(type_name) -> str | None:
    if not isinstance(type_name, str) or not type_name:
        return None
    for group, members in PHOTORECEPTOR_GROUPS.items():
        if type_name in members:
            return group
    return type_name


def class_mapping(ensemble: FlyvisEnsemble, classes) -> dict[str, tuple[str, ...]]:
    """Which flyvis types each class stands for (rule 2)."""
    flyvis_types = set(ensemble.types)
    present = set(classes)
    mapping = {c: (c,) for c in present if c in flyvis_types}
    mapping.update({c: t for c, t in CLASS_TO_FLYVIS.items() if c in present})
    return mapping


def _stand_ins(unit_class, column, src, tar, count, class_index, n_columns):
    """Rule 7: per group, the synapses complete columns deliver onto each columnar class, and what is missing."""
    extra, report = [], {}
    for group, full in FULL_COLUMN.items():
        g = class_index[group]
        is_g = unit_class == g
        per_column = np.bincount(column[is_g & (column >= 0)], minlength=n_columns)
        complete = per_column == full
        same = is_g[src] & (column[src] >= 0) & (column[tar] == column[src])
        received = np.bincount(tar[same], weights=count[same], minlength=len(unit_class))
        deficit = np.zeros(len(unit_class))
        medians = {}
        classes_of = {v: k for k, v in class_index.items()}
        for cls in np.unique(unit_class[tar[same]]):
            of_class = (unit_class == cls) & (column >= 0)
            ref = received[of_class & complete[np.maximum(column, 0)]]
            if len(ref) == 0 or np.median(ref) <= 0:
                continue
            m = float(np.median(ref))
            medians[classes_of[cls]] = m
            deficit[of_class] = np.maximum(0.0, m - received[of_class])
        needing = np.flatnonzero(deficit > 0)
        columns = np.unique(column[needing])
        for col in columns:
            targets = needing[column[needing] == col]
            extra.append((g, int(col), targets, deficit[targets]))
        report[group] = {"columns_complete": int(complete.sum()), "stand_ins": int(len(columns)),
                         "median_synapses_onto": {k: round(v, 2) for k, v in sorted(medians.items())}}
    return extra, report


def home_columns(column: np.ndarray, src: np.ndarray, tar: np.ndarray, count: np.ndarray, n_columns: int,
                 passes: int = 3) -> np.ndarray:
    """A column for every unit: its own where the release gives one, else the column that delivers it the most
    synapses (through presynaptic partners that have a column), over ``passes`` rounds. -1 where none reaches."""
    home = column.copy()
    for _ in range(passes):
        use = (home[src] >= 0) & (home[tar] < 0)
        if not use.any():
            break
        key = tar[use] * n_columns + home[src[use]]
        uniq, inverse = np.unique(key, return_inverse=True)
        total = np.bincount(inverse, weights=count[use])
        unit, col = uniq // n_columns, uniq % n_columns
        order = np.lexsort((-total, unit))
        first = order[np.r_[True, unit[order][1:] != unit[order][:-1]]]
        home[unit[first]] = col[first]
    return home


def _split_ct1(src, tar, count, unit_class, column, neuron, stand_in, class_index, mapping, ensemble, n_columns):
    """flyvis models CT1 as one medulla (M10) and one lobula (Lo1) compartment per column, electrically separate;
    MaleCNS has one CT1 per side. Each CT1 connection moves to the compartment of its partner's column (the partner's
    own column, else the column that feeds it most), of the kind flyvis pairs the partner's type with (M10 or Lo1;
    M10 where flyvis pairs it with neither). Connections whose partner reaches no column stay on the whole CT1, which
    keeps its unmapped defaults."""
    ct1 = class_index["CT1"]
    home = home_columns(column, src, tar, count, n_columns)
    fv = {(ensemble.types[s], ensemble.types[t])
          for s, t in zip(ensemble.pair_source, ensemble.pair_target, strict=True)}
    names = {v: k for k, v in class_index.items()}

    def kind(partner_class: int, incoming: bool) -> str:
        weight = {c: 0 for c in CT1_COMPARTMENTS}
        for t in mapping.get(names[partner_class], ()):
            for c in CT1_COMPARTMENTS:
                if ((t, c) if incoming else (c, t)) in fv:
                    weight[c] += 1
        return "CT1(Lo1)" if weight["CT1(Lo1)"] > weight["CT1(M10)"] else "CT1(M10)"

    kind_in = {c: kind(c, True) for c in range(len(class_index))}
    kind_out = {c: kind(c, False) for c in range(len(class_index))}
    compartments: dict[tuple[int, str, int], int] = {}
    new_class, new_column, new_neuron = [], [], []
    first = len(unit_class)
    moved = 0
    src, tar = src.copy(), tar.copy()
    for e in np.flatnonzero((unit_class[tar] == ct1) | (unit_class[src] == ct1)):
        if unit_class[tar[e]] == ct1 and unit_class[src[e]] != ct1:
            whole, partner, k = tar[e], src[e], kind_in[unit_class[src[e]]]
        elif unit_class[src[e]] == ct1 and unit_class[tar[e]] != ct1:
            whole, partner, k = src[e], tar[e], kind_out[unit_class[tar[e]]]
        else:
            continue
        col = home[partner]
        if col < 0:
            continue
        key = (int(whole), k, int(col))
        if key not in compartments:
            compartments[key] = first + len(new_class)
            new_class.append(class_index[k])
            new_column.append(col)
            new_neuron.append(neuron[whole])
        if tar[e] == whole:
            tar[e] = compartments[key]
        else:
            src[e] = compartments[key]
        moved += 1
    ct1_edges = int(np.count_nonzero((unit_class[np.minimum(tar, first - 1)] == ct1) & (tar < first)
                                     | (unit_class[np.minimum(src, first - 1)] == ct1) & (src < first)))
    total = ct1_edges + moved
    report = {"compartments": len(new_class), "connections_moved": moved,
              "connections_left_on_whole_ct1": total - moved,
              "M10": sum(1 for c in new_class if c == class_index["CT1(M10)"]),
              "Lo1": sum(1 for c in new_class if c == class_index["CT1(Lo1)"])}
    return (src, tar, np.concatenate([unit_class, np.array(new_class, dtype=np.int64)]),
            np.concatenate([column, np.array(new_column, dtype=np.int64)]),
            np.concatenate([neuron, np.array(new_neuron, dtype=np.int64)]),
            np.concatenate([stand_in, np.zeros(len(new_class), dtype=bool)]), report)


def build_optic_lobe(graph: Compiled, ensemble: FlyvisEnsemble, model: int = 0,
                     rules: TransferRules | None = None) -> OpticLobe:
    """Both optic lobes of a compiled MaleCNS as graded units, with the parameters of flyvis network ``model``."""
    rules = rules or TransferRules()
    n_all = graph.n_neurons
    names = np.array(graph.strings["type"], dtype=object)[graph["neuron_type"]]
    superclass = np.array(graph.strings["superclass"], dtype=object)[graph["neuron_superclass"]]
    cls_all = np.array([_class_of(t) for t in names], dtype=object)
    known = cls_all != None  # noqa: E711
    mapping = class_mapping(ensemble, {str(c) for c in cls_all[known]})
    photoreceptor = np.isin(cls_all, list(PHOTORECEPTOR_GROUPS))
    in_mapping = np.isin(cls_all, list(mapping))
    chosen = known & ((superclass == "ol_intrinsic") | photoreceptor | in_mapping)
    if not rules.include_unmapped:
        chosen &= in_mapping

    units = np.flatnonzero(chosen)
    local = np.full(n_all, -1, dtype=np.int64)
    local[units] = np.arange(len(units))
    indptr = graph["csr_indptr"].astype(np.int64)
    pre = np.repeat(np.arange(n_all, dtype=np.int64), np.diff(indptr))
    post = graph["csr_indices"].astype(np.int64)
    keep = chosen[pre] & chosen[post]
    src, tar = local[pre[keep]], local[post[keep]]
    count = graph["csr_count"][keep].astype(np.float64)
    pre_sign = graph["neuron_sign"][pre[keep]].astype(np.float64)

    classes = sorted({str(c) for c in cls_all[units]} | set(FULL_COLUMN) | set(CT1_COMPARTMENTS))
    mapping.update({c: (c,) for c in CT1_COMPARTMENTS})
    class_index = {c: i for i, c in enumerate(classes)}
    unit_class = np.array([class_index[str(c)] for c in cls_all[units]], dtype=np.int64)
    column = graph["neuron_column"][units].astype(np.int64)
    n_columns = len(graph["column_side"])
    stand_in = np.zeros(len(units), dtype=bool)
    neuron = units.copy()

    standin_report = {}
    if rules.stand_ins:
        extra, standin_report = _stand_ins(unit_class, column, src, tar, count, class_index, n_columns)
        if extra:
            first = len(unit_class)
            ids = np.arange(first, first + len(extra))
            src = np.concatenate([src, *[np.full(len(t), i) for i, (_, _, t, _) in zip(ids, extra, strict=True)]])
            tar = np.concatenate([tar, *[t for _, _, t, _ in extra]])
            count = np.concatenate([count, *[d for _, _, _, d in extra]])
            pre_sign = np.concatenate([pre_sign, *[np.full(len(t), -1.0) for _, _, t, _ in extra]])  # histamine
            unit_class = np.concatenate([unit_class, np.array([g for g, _, _, _ in extra])])
            column = np.concatenate([column, np.array([c for _, c, _, _ in extra])])
            stand_in = np.concatenate([stand_in, np.ones(len(extra), dtype=bool)])
            neuron = np.concatenate([neuron, np.full(len(extra), -1)])

    ct1_report = {}
    if rules.ct1_compartments and "CT1" in class_index:
        (src, tar, unit_class, column, neuron, stand_in, ct1_report) = _split_ct1(
            src, tar, count, unit_class, column, neuron, stand_in, class_index, mapping, ensemble, n_columns)

    # rule 3: neurons
    type_index = {t: i for i, t in enumerate(ensemble.types)}
    bias_m = ensemble.bias[model].astype(np.float64)
    tau_m = ensemble.time_const_s[model].astype(np.float64)
    default_tau = float(np.median(tau_m))
    class_bias = np.full(len(classes), rules.default_bias)
    class_tau = np.full(len(classes), default_tau)
    class_mapped = np.zeros(len(classes), dtype=bool)
    for c, i in class_index.items():
        if c in mapping:
            idx = [type_index[t] for t in mapping[c]]
            class_bias[i], class_tau[i], class_mapped[i] = bias_m[idx].mean(), tau_m[idx].mean(), True

    # rules 4 and 6: connections
    n_classes = len(classes)
    pair_key = unit_class[src] * n_classes + unit_class[tar]
    synapses = np.bincount(pair_key, weights=count, minlength=n_classes * n_classes)
    class_size = np.bincount(unit_class, minlength=n_classes).astype(np.float64)
    n_tot = np.bincount(ensemble.group_pair, weights=ensemble.group_n_syn, minlength=len(ensemble.pair_sign))
    alpha = ensemble.strength[model].astype(np.float64)
    fv_pair = {(ensemble.types[s], ensemble.types[t]): i
               for i, (s, t) in enumerate(zip(ensemble.pair_source, ensemble.pair_target, strict=True))}
    per_synapse = np.zeros(n_classes * n_classes)
    flyvis_count = np.zeros(n_classes * n_classes)
    transferred = np.zeros(n_classes * n_classes, dtype=bool)
    covered = set()
    ratios = []
    for key in np.flatnonzero(synapses > 0):
        s_cls, t_cls = classes[key // n_classes], classes[key % n_classes]
        if s_cls not in mapping or t_cls not in mapping:
            continue
        found = [(fv_pair[(s, t)], t) for t in mapping[t_cls] for s in mapping[s_cls] if (s, t) in fv_pair]
        if not found:
            continue
        covered.update(i for i, _ in found)
        signed = np.array([ensemble.pair_sign[i] * n_tot[i] * alpha[i] for i, _ in found])
        counts_f = np.array([n_tot[i] for i, _ in found])
        per_target = synapses[key] / class_size[key % n_classes]
        drive_per_target = signed.sum() / len(mapping[t_cls])
        flyvis_per_target = counts_f.sum() / len(mapping[t_cls])
        ratios.append((per_target / flyvis_per_target, abs(drive_per_target)))
        per_synapse[key] = signed.sum() / counts_f.sum()             # flyvis's count-weighted signed strength
        flyvis_count[key] = flyvis_per_target
        transferred[key] = True
    # what each neuron receives from each presynaptic class
    received_key = tar * n_classes + unit_class[src]
    received = np.bincount(received_key, weights=count, minlength=len(unit_class) * n_classes)[received_key]
    t_edges = transferred[pair_key]
    scale = flyvis_count[pair_key] / np.maximum(received, 1e-12)
    cap = np.minimum(1.0, scale) if rules.transfer == "capped" else scale
    weight = np.where(t_edges, count * per_synapse[pair_key] * cap,
                      pre_sign * count * rules.default_rho * rules.default_offsets / np.maximum(received, 1e-12))
    capped = t_edges & (cap < 1.0)

    # rule 8: light
    groups = [class_index[g] for g in PHOTORECEPTOR_GROUPS if g in class_index]
    lit = np.flatnonzero(np.isin(unit_class, groups) & (column >= 0))
    network = GradedNetwork(bias=class_bias[unit_class], time_const_s=class_tau[unit_class], source=src, target=tar,
                            weight=weight, input_neuron=lit, input_column=column[lit], n_columns=n_columns)

    flyvis_drive = np.abs(ensemble.pair_sign * n_tot * alpha)
    disagree = t_edges & (pre_sign != 0) & (np.sign(pre_sign) != np.sign(weight))
    r = np.array(ratios)
    order = np.argsort(r[:, 0])
    cumulative = np.cumsum(r[order, 1]) / r[:, 1].sum()
    report = {
        "model": model,
        "units": int(len(unit_class)), "real_units": int(len(np.unique(neuron[neuron >= 0]))),
        "stand_ins": int(stand_in.sum()),
        "photoreceptors_lit": int(len(lit)), "classes": n_classes, "classes_mapped": int(class_mapped.sum()),
        "units_mapped": int(class_mapped[unit_class].sum()),
        "flyvis_types_without_malecns_class": sorted(set(ensemble.types)
                                                     - {t for c, ts in mapping.items() for t in ts}),
        "connections": int(len(weight)), "synapses": float(count.sum()),
        "connections_transferred": int(t_edges.sum()),
        "synapses_transferred_fraction": float(count[t_edges].sum() / count.sum()),
        "flyvis_pairs_covered": len(covered),
        "flyvis_drive_covered_fraction": float(flyvis_drive[sorted(covered)].sum() / flyvis_drive.sum()),
        "count_ratio_median_weighted_by_drive": float(r[order, 0][np.searchsorted(cumulative, 0.5)]),
        "sign_disagreements_fraction_of_transferred": float(disagree.sum() / max(t_edges.sum(), 1)),
        "transferred_synapses_capped_fraction": float(count[capped].sum() / max(count[t_edges].sum(), 1)),
        "default_tau_s": default_tau, "stand_in_rule": standin_report, "ct1_compartments": ct1_report,
    }
    side_names = np.array(graph.strings["side"], dtype=object)
    side = np.where(neuron >= 0, side_names[graph["neuron_side"][np.maximum(neuron, 0)]],
                    np.where(graph["column_side"][np.maximum(column, 0)] == 1, "left", "right"))
    return OpticLobe(network=network, neuron=neuron, unit_class=unit_class, classes=tuple(classes), column=column,
                     side=side.astype(object),
                     stand_in=stand_in, mapped=class_mapped[unit_class], transferred=t_edges, model=model,
                     report=report)


def unit_directions(lobe: OpticLobe, column_directions: np.ndarray, passes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A viewing direction for every unit: its column's for the neurons the release places in a column, and for the
    rest (T4, T5, Tm3, T2, TmY cells and others the release gives no column) the synapse-weighted mean direction of
    their presynaptic partners, over ``passes`` rounds so that second-order cells inherit from first-order ones.
    Returns (directions (units, 3), order): order 0 for a column of its own, k for a direction derived in round k,
    -1 for none."""
    n = len(lobe.unit_class)
    directions = np.full((n, 3), np.nan)
    order = np.full(n, -1, dtype=np.int64)
    has = lobe.column >= 0
    directions[has] = column_directions[lobe.column[has]]
    order[has] = 0
    src, tar = lobe.network.source, lobe.network.target
    count = np.abs(lobe.network.weight) > 0
    for k in range(1, passes + 1):
        known = order >= 0
        use = known[src] & ~known[tar] & count
        if not use.any():
            break
        sums = np.zeros((n, 3))
        for axis in range(3):
            sums[:, axis] = np.bincount(tar[use], weights=directions[src[use], axis], minlength=n)
        norm = np.linalg.norm(sums, axis=1)
        new = (~known) & (norm > 0)
        directions[new] = sums[new] / norm[new, None]
        order[new] = k
    return directions, order


def azimuth_elevation(directions: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of the eyes' convention: azimuth from straight ahead toward the eye's own side, elevation up."""
    lateral = -1.0 if side == "right" else 1.0
    az = np.degrees(np.arctan2(lateral * directions[:, 1], directions[:, 0]))
    el = np.degrees(np.arcsin(np.clip(directions[:, 2], -1, 1)))
    return az, el
