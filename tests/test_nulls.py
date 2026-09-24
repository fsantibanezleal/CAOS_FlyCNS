"""The null graphs keep what each one promises, and nothing else of the real graph.

The synthetic graphs have heavy-tailed degrees, so the N1 permutation creates many collisions around their hubs and
the repair is exercised: with a Pareto tail of 2 (hubs reach about 80% of the graph) every collision is repaired;
with a tail of 1.2 (hubs reach every neuron) some collisions admit no swap at all, and they must be dropped and
counted. The MaleCNS run needs the compiled release and is skipped (and listed by ``-rs``) without it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flycns.nulls import random_size_matched, rewire_degree_preserving, shuffle_signs

COMPILED = Path(os.environ.get("FLYCNS_MALECNS_COMPILED", "E:/_Datos/destello/compiled/malecns-v1.0"))


def graph(seed: int = 0, n: int = 400, edges: int = 12_000, tail: float = 2.0):
    rng = np.random.default_rng(seed)
    partition = rng.integers(0, 4, size=n).astype(np.int8)
    weight = rng.pareto(tail, size=n) + 1.0
    weight /= weight.sum()
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < edges:
        a, b = rng.choice(n, size=2, p=weight)
        if a != b:
            pairs.add((int(a), int(b)))
    pre, post = np.array(sorted(pairs)).T
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    counts = rng.integers(1, 60, size=edges).astype(np.uint16)
    signs = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n, p=[0.35, 0.05, 0.6])
    return indptr, post.astype(np.int32), counts, partition, signs


def edges_of(indptr, indices):
    pre = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    return pre, np.asarray(indices, dtype=np.int64)


def block_counts(pre, post, partition):
    return np.bincount(partition[pre].astype(np.int64) * 4 + partition[post], minlength=16)


def assert_simple(pre, post):
    assert np.all(pre != post)
    assert len(np.unique(pre * 1_000_003 + post)) == len(pre)


def test_null_graphs_keep_what_they_promise():
    indptr, indices, counts, partition, signs = graph()
    n = len(indptr) - 1
    pre, post = edges_of(indptr, indices)
    real_pairs = set(zip(pre.tolist(), post.tolist(), strict=True))

    # N1: degrees, block sizes and each neuron's outgoing counts kept; the pairs changed
    i1, x1, c1, unresolved = rewire_degree_preserving(indptr, indices, counts, partition, seed=5)
    assert unresolved == 0
    p1, q1 = edges_of(i1, x1)
    assert_simple(p1, q1)
    assert np.array_equal(np.diff(i1), np.diff(indptr))
    assert np.array_equal(np.bincount(q1, minlength=n), np.bincount(post, minlength=n))
    assert np.array_equal(block_counts(p1, q1, partition), block_counts(pre, post, partition))
    for neuron in range(n):
        kept = sorted(c1[i1[neuron]:i1[neuron + 1]].tolist())
        assert kept == sorted(counts[indptr[neuron]:indptr[neuron + 1]].tolist())
    assert len(real_pairs & set(zip(p1.tolist(), q1.tolist(), strict=True))) < 0.5 * len(real_pairs)
    again = rewire_degree_preserving(indptr, indices, counts, partition, seed=5)
    assert all(np.array_equal(u, v) for u, v in zip(again[:3], (i1, x1, c1), strict=True))
    assert not np.array_equal(rewire_degree_preserving(indptr, indices, counts, partition, seed=6)[1], x1)

    # N1 where hubs reach every neuron: the collisions no swap can repair are dropped and counted, never kept
    hi, hx, hc, hpart, _ = graph(tail=1.2)
    ji, jx, jc, dropped = rewire_degree_preserving(hi, hx, hc, hpart, seed=5)
    assert 0 < dropped < 0.01 * len(hx)
    assert len(jx) == len(hx) - dropped
    assert_simple(*edges_of(ji, jx))
    assert np.all(np.diff(ji) <= np.diff(hi))
    assert np.all(np.bincount(jx, minlength=len(hi) - 1) <= np.bincount(hx, minlength=len(hi) - 1))
    assert (np.diff(hi) - np.diff(ji)).sum() == dropped

    # N2: block sizes and each block's counts kept; degrees not
    i2, x2, c2 = random_size_matched(indptr, indices, counts, partition, seed=5)
    p2, q2 = edges_of(i2, x2)
    assert_simple(p2, q2)
    assert np.array_equal(block_counts(p2, q2, partition), block_counts(pre, post, partition))
    real_block = partition[pre].astype(np.int64) * 4 + partition[post]
    null_block = partition[p2].astype(np.int64) * 4 + partition[q2]
    for b in np.unique(real_block):
        assert sorted(c2[null_block == b].tolist()) == sorted(counts[real_block == b].tolist())
    assert not np.array_equal(np.diff(i2), np.diff(indptr))
    assert len(real_pairs & set(zip(p2.tolist(), q2.tolist(), strict=True))) < 0.2 * len(real_pairs)

    # N3: the wiring untouched (the function never sees it), the multiset of signs kept, unsigned neurons unsigned
    s3 = shuffle_signs(signs, seed=5)
    assert sorted(s3.tolist()) == sorted(signs.tolist())
    assert np.array_equal(s3 == 0, signs == 0)
    assert np.mean(s3 != signs) > 0.2


@pytest.mark.data
@pytest.mark.skipif(not (COMPILED / "manifest.json").is_file(), reason=f"no compiled MaleCNS at {COMPILED}")
def test_degree_preserving_null_of_malecns_is_exact():
    from flycns.compiled import read_compiled

    c = read_compiled(COMPILED, verify=False)
    indptr, indices, counts = c["csr_indptr"], c["csr_indices"], c["csr_count"]
    i1, x1, c1, unresolved = rewire_degree_preserving(indptr, indices, counts, c["neuron_partition"], seed=0)
    assert unresolved == 0
    assert len(x1) == len(indices) == 25_582_938
    assert np.array_equal(np.diff(i1), np.diff(indptr))
    n = len(indptr) - 1
    assert np.array_equal(np.bincount(x1, minlength=n), np.bincount(indices, minlength=n))
    assert int(c1.astype(np.int64).sum()) == int(counts.astype(np.int64).sum())
