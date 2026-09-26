/**
 * Counter-based randomness, the same numbers as `flycns.rng`: MurmurHash3_x86_32 of the 8-byte little-endian key
 * (neuron, step), seeded with the run's seed. Every operation is a 32-bit unsigned multiply (`Math.imul`), add, xor,
 * shift or rotation, so the result equals NumPy's uint32 arithmetic and the WGSL kernel's u32 arithmetic bit for bit.
 */

const C1 = 0xcc9e2d51;
const C2 = 0x1b873593;
const ROUND = 0xe6546b64;
const MIX_A = 0x85ebca6b;
const MIX_B = 0xc2b2ae35;
const KEY_BYTES = 8;

function rotl(x: number, r: number): number {
  return ((x << r) | (x >>> (32 - r))) >>> 0;
}

function block(h: number, k: number): number {
  k = Math.imul(k, C1) >>> 0;
  k = rotl(k, 15);
  k = Math.imul(k, C2) >>> 0;
  h = (h ^ k) >>> 0;
  h = rotl(h, 13);
  return (Math.imul(h, 5) + ROUND) >>> 0;
}

/** MurmurHash3's 32-bit finaliser. */
export function fmix32(h: number): number {
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, MIX_A) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, MIX_B) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

/** MurmurHash3_x86_32 of (neuron, step) seeded with `seed`: an unsigned 32-bit integer. */
export function hash3(seed: number, neuron: number, step: number): number {
  let h = seed >>> 0;
  h = block(h, neuron >>> 0);
  h = block(h, step >>> 0);
  return fmix32((h ^ KEY_BYTES) >>> 0);
}

/**
 * The threshold below which a hash counts as an event of probability `rate * dt`; `rateHz * dtS` is computed here
 * as the reference's `event_threshold` computes it (the caller passes `dtS` already divided).
 */
export function eventThreshold(rateHz: number, dtS: number): number {
  const p = Math.min(Math.max(rateHz * dtS, 0), 1);
  return Math.min(Math.floor(p * 4294967296), 4294967295);
}

/** Whether `neuron` receives an event at `step` under `threshold`. */
export function poissonEvent(seed: number, neuron: number, step: number, threshold: number): boolean {
  return hash3(seed, neuron, step) < threshold;
}
