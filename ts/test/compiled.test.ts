import { test } from "node:test";
import assert from "node:assert/strict";
import { CompiledError, readCompiled } from "../src/compiled.js";
import { SCENARIO_SCHEMA } from "../src/bundle.js";
import { fsSource, parityDir } from "./files.js";

test("the loader checks every array against its manifest", async () => {
  const source = fsSource(parityDir("lif-poisson"));
  const c = await readCompiled(source, { schema: SCENARIO_SCHEMA });
  // every allowed dtype lands in its typed array; int64 comes back as BigInt64Array and converts on request
  assert.ok(c.get("lif_weight_mv") instanceof Float64Array);
  assert.ok(c.get("lif_indices") instanceof Int32Array);
  assert.ok(c.get("lif_indptr") instanceof BigInt64Array);
  assert.equal(c.numbers("lif_indptr").length, 61);
  assert.equal(c.int32("lif_indptr")[60], c.get("lif_indices").length);
  assert.deepEqual(c.shape("drive_modulated_rate_hz"), [20, 3]);
  // a wrong schema
  await assert.rejects(readCompiled(source, { schema: "flycns.compiled/1" }), CompiledError);
  // a flipped byte in one array
  const tampered = async (name: string) => {
    const bytes = new Uint8Array(await source(name));
    if (name === "lif_indices.bin") bytes[0] ^= 1;
    return bytes;
  };
  await assert.rejects(readCompiled(tampered, { schema: SCENARIO_SCHEMA }), (error: Error) => {
    assert.ok(error instanceof CompiledError && error.message.includes("lif_indices"));
    return true;
  });
  // the same bytes pass without verification (partition-first loading may skip the digest, never the length)
  await readCompiled(tampered, { schema: SCENARIO_SCHEMA, verify: false });
  // a byte length that does not match the declared shape
  const truncated = async (name: string) => {
    const bytes = await source(name);
    return name === "lif_weight_mv.bin" ? bytes.subarray(0, bytes.length - 8) : bytes;
  };
  await assert.rejects(readCompiled(truncated, { schema: SCENARIO_SCHEMA, verify: false }), (error: Error) => {
    assert.ok(error instanceof CompiledError && error.message.includes("lif_weight_mv"));
    return true;
  });
  // a missing file
  const missing = async (name: string) => {
    if (name === "drive_silenced.bin") throw new Error("gone");
    return source(name);
  };
  await assert.rejects(readCompiled(missing, { schema: SCENARIO_SCHEMA }), CompiledError);
  // `only` reads a subset, still verified
  const part = await readCompiled(source, { schema: SCENARIO_SCHEMA, only: ["lif_indptr"] });
  assert.ok(part.has("lif_indptr") && !part.has("lif_indices"));
});
