"""Null graphs: the same number of connections, wired so that what makes the real graph special is removed.

Each null answers "is it the wiring?" by keeping some of the real graph's structure and destroying the rest:

- **N1, degree-preserving rewiring**: every neuron keeps its in-degree and out-degree, every connection its count and
  its presynaptic neuron (hence its sign), and every partition block (optic lobe left/right, central brain, nerve cord,
  in each pre/post combination) keeps its number of connections. Only which neuron meets which changes.
- **N2, size-matched random graph**: each block keeps its number of connections; the pairs are drawn uniformly at
  random (no self connections, no repeats) and the counts are the block's real counts, shuffled.
- **N3, sign shuffle**: the wiring and counts unchanged; the neuron signs permuted among the neurons that have one.

All three are seeded and deterministic.
"""

from __future__ import annotations

import numpy as np


def _edges(indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pre = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
    return pre, indices.astype(np.int64)


def _to_csr(pre: np.ndarray, post: np.ndarray, counts: np.ndarray, n: int):
    order = np.lexsort((post, pre))
    pre, post, counts = pre[order], post[order], counts[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    return indptr, post.astype(np.int32), counts


def _blocks(pre: np.ndarray, post: np.ndarray, partition: np.ndarray) -> np.ndarray:
    return partition[pre].astype(np.int64) * 256 + partition[post].astype(np.int64)


def _count_in(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_keys, keys, "right") - np.searchsorted(sorted_keys, keys, "left")


def rewire_degree_preserving(indptr, indices, counts, partition, seed: int = 0, max_candidates: int = 1024,
                             patience: int = 20):
    """N1. The targets of each block's connections are permuted; the collisions this creates (a self connection, or a
    pair that already exists) are then repaired by swaps. A swap exchanges the targets of a colliding connection and a
    random connection of the same block, and is accepted only when both new pairs are self-free and absent from the
    current graph and from every other swap accepted in the same round, so a repair never creates a collision. Rounds
    repeat, with more candidate partners per collision as fewer remain, until none is left or ``patience`` rounds at
    ``max_candidates`` accept nothing. Returns ``(indptr, indices, counts, unresolved)``: collisions still left are
    dropped and counted, so the degrees are exact whenever ``unresolved`` is 0.
    """
    rng = np.random.default_rng(seed)
    n = len(indptr) - 1
    pre, post = _edges(np.asarray(indptr), np.asarray(indices))
    counts = np.asarray(counts).copy()
    block = _blocks(pre, post, np.asarray(partition))
    _, block_id = np.unique(block, return_inverse=True)
    by_block = np.argsort(block_id, kind="stable")
    sizes = np.bincount(block_id)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    new_post = post.copy()
    for b in range(len(sizes)):
        members = by_block[starts[b]:starts[b] + sizes[b]]
        new_post[members] = post[members][rng.permutation(len(members))]
    keys = pre * n + new_post
    base = np.sort(keys)                                  # the pairs after the permutation, with multiplicity
    added = np.zeros(0, dtype=np.int64)
    removed = np.zeros(0, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    dup = np.zeros(len(keys), dtype=bool)
    dup[order[1:]] = keys[order][1:] == keys[order][:-1]
    pending = np.flatnonzero(dup | (pre == new_post))
    is_pending = np.zeros(len(keys), dtype=bool)
    is_pending[pending] = True
    idle = 0
    while len(pending) and idle < patience:
        tries = int(np.clip(65536 // len(pending), 1, max_candidates))
        e = np.repeat(pending, tries)
        bid = block_id[e]
        f = by_block[starts[bid] + (rng.random(len(e)) * sizes[bid]).astype(np.int64)]
        u, x, w, y = pre[e], new_post[e], pre[f], new_post[f]
        k1, k2 = u * n + y, w * n + x
        ok = ~is_pending[f] & (u != y) & (w != x)
        for k in (k1, k2):
            ok &= (_count_in(base, k) - _count_in(removed, k) + _count_in(added, k)) == 0
        rows = np.flatnonzero(ok)
        rows = rows[np.unique(e[rows], return_index=True)[1]]            # the first valid candidate of each collision
        rows = rows[np.sort(np.unique(f[rows], return_index=True)[1])]   # a partner serves one swap
        created = np.concatenate([k1[rows], k2[rows]])
        _, inverse, multiplicity = np.unique(created, return_inverse=True, return_counts=True)
        clash = multiplicity[inverse] > 1                                  # two swaps creating the same pair
        rows = rows[~(clash[:len(rows)] | clash[len(rows):])]
        if len(rows) == 0:
            idle += tries == max_candidates
            continue
        idle = 0
        ea, fa = e[rows], f[rows]
        removed = np.sort(np.concatenate([removed, keys[ea], keys[fa]]))
        added = np.sort(np.concatenate([added, k1[rows], k2[rows]]))
        new_post[ea], new_post[fa] = y[rows], x[rows]
        keys[ea], keys[fa] = k1[rows], k2[rows]
        is_pending[ea] = False
        pending = np.flatnonzero(is_pending)
    keep = ~is_pending
    indptr2, indices2, counts2 = _to_csr(pre[keep], new_post[keep], counts[keep], n)
    return indptr2, indices2, counts2, int(is_pending.sum())


def random_size_matched(indptr, indices, counts, partition, seed: int = 0):
    """N2. For each block, as many connections as the real one, between uniformly random distinct pairs of the block's
    presynaptic and postsynaptic partitions, carrying the block's real counts in random order. Pairs are drawn
    independently and kept in order of first appearance until the block is full, which gives a uniformly random set of
    distinct pairs."""
    rng = np.random.default_rng(seed)
    n = len(indptr) - 1
    partition = np.asarray(partition)
    pre, post = _edges(np.asarray(indptr), np.asarray(indices))
    counts = np.asarray(counts)
    block = _blocks(pre, post, partition)
    out_pre, out_post, out_counts = [], [], []
    for b in np.unique(block):
        members = np.flatnonzero(block == b)
        sources = np.flatnonzero(partition == b // 256)
        targets = np.flatnonzero(partition == b % 256)
        want = len(members)
        keys = np.zeros(0, dtype=np.int64)
        while len(keys) < want:
            k = (want - len(keys)) * 2 + 16
            s = sources[rng.integers(len(sources), size=k)]
            t = targets[rng.integers(len(targets), size=k)]
            drawn = np.concatenate([keys, (s * n + t)[s != t]])
            keys = drawn[np.sort(np.unique(drawn, return_index=True)[1])][:want]
        out_pre.append(keys // n)
        out_post.append(keys % n)
        out_counts.append(counts[members][rng.permutation(want)])
    return _to_csr(np.concatenate(out_pre), np.concatenate(out_post), np.concatenate(out_counts), n)


def shuffle_signs(signs: np.ndarray, seed: int = 0) -> np.ndarray:
    """N3. The signs of the neurons that have one, permuted among them; neurons without a sign keep 0."""
    rng = np.random.default_rng(seed)
    signs = np.asarray(signs).copy()
    signed = np.flatnonzero(signs != 0)
    signs[signed] = signs[signed][rng.permutation(len(signed))]
    return signs
