import { test } from "node:test";
import assert from "node:assert/strict";
import { eventThreshold, hash3 } from "../src/rng.js";
import { readCompiled } from "../src/compiled.js";
import { fsSource, parityDir } from "./files.js";

// (seed, neuron, step) -> hash, pinned in tests/test_lif.py and checked there against the mmh3 package
const VECTORS: [[number, number, number], number][] = [
  [[0, 0, 0], 1669671676],
  [[1, 2, 3], 2920678231],
  [[0xdeadbeef, 166699, 1000000], 2110861114],
  [[0xffffffff, 0xffffffff, 0xffffffff], 3338479333],
  [[42, 12345, 67890], 80117027],
];

test("the hash matches the pinned vectors and the reference on the committed keys", async () => {
  for (const [[seed, neuron, step], expected] of VECTORS) assert.equal(hash3(seed, neuron, step), expected);
  const vectors = await readCompiled(fsSource(parityDir("hash-vectors")), { schema: "flycns.vectors/1" });
  const seed = vectors.get("seed") as Uint32Array;
  const neuron = vectors.get("neuron") as Uint32Array;
  const step = vectors.get("step") as Uint32Array;
  const hash = vectors.get("hash") as Uint32Array;
  assert.ok(seed.length >= 10000);
  for (let i = 0; i < seed.length; i++) {
    assert.equal(hash3(seed[i] as number, neuron[i] as number, step[i] as number), hash[i], `key ${i}`);
  }
  // thresholds as the reference computes them: probability floored onto 2^32, saturating at 2^32 - 1
  const threshold = vectors.numbers("threshold");
  const rate = vectors.float64("rate_hz");
  for (let i = 0; i < rate.length; i++) assert.equal(eventThreshold(rate[i] as number, 0.1 / 1000.0), threshold[i]);
});
