"""Counter-based randomness: the same (seed, neuron, step) gives the same number in Python, TypeScript and WGSL.

The generator is MurmurHash3_x86_32 (Austin Appleby's public-domain hash) of the 8-byte little-endian key
(neuron, step), with the run's seed as the hash seed. Every operation is a 32-bit unsigned multiply, add, xor, shift
or rotation, which JavaScript (``Math.imul``, ``>>> 0``), WGSL (``u32`` arithmetic wraps) and NumPy (``uint32`` wraps)
compute identically, and any MurmurHash3 implementation is an oracle for it. A Poisson event of rate ``r`` at step
size ``dt`` occurs when the hash is below ``floor(r * dt * 2**32)``, so an event stream needs no generator state, can
be evaluated for any neuron at any step in any order, and reproduces bit for bit across implementations.
"""

from __future__ import annotations

import numpy as np

_C1 = np.uint32(0xCC9E2D51)
_C2 = np.uint32(0x1B873593)
_ROUND = np.uint32(0xE6546B64)
_FIVE = np.uint32(5)
_MIX_A = np.uint32(0x85EBCA6B)
_MIX_B = np.uint32(0xC2B2AE35)
_KEY_BYTES = np.uint32(8)


def _rotl(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def fmix32(h: np.ndarray) -> np.ndarray:
    """MurmurHash3's 32-bit finaliser, on uint32 arrays."""
    h = np.asarray(h, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = h ^ (h >> np.uint32(16))
        h = h * _MIX_A
        h = h ^ (h >> np.uint32(13))
        h = h * _MIX_B
        h = h ^ (h >> np.uint32(16))
    return h


def _block(h: np.ndarray, k: np.ndarray) -> np.ndarray:
    """One 4-byte block of MurmurHash3_x86_32's body."""
    k = _rotl(k * _C1, 15) * _C2
    return _rotl(h ^ k, 13) * _FIVE + _ROUND


def hash3(seed: int, neuron: np.ndarray, step: int | np.ndarray) -> np.ndarray:
    """MurmurHash3_x86_32 of the little-endian bytes of (neuron, step), seeded with ``seed``; a uint32 array."""
    neuron = np.asarray(neuron, dtype=np.uint32)
    step = np.asarray(step, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = np.full(np.broadcast_shapes(neuron.shape, step.shape), seed & 0xFFFFFFFF, dtype=np.uint32)
        h = _block(h, neuron)
        h = _block(h, step)
        return fmix32(h ^ _KEY_BYTES)


def event_threshold(rate_hz: float, dt_s: float) -> int:
    """The uint32 threshold below which a hash counts as an event of probability ``rate * dt``."""
    p = min(max(rate_hz * dt_s, 0.0), 1.0)
    return min(int(p * 2**32), 2**32 - 1)


def poisson_events(seed: int, neurons: np.ndarray, step: int, threshold: int | np.ndarray) -> np.ndarray:
    """Boolean mask: which of ``neurons`` receive an event at ``step``."""
    return hash3(seed, neurons, step) < np.asarray(threshold, dtype=np.uint64)
