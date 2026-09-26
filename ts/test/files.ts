/** Test helpers: file sources over the repository's parity fixtures, and array comparisons. */

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { FileSource } from "../src/compiled.js";

export const PARITY = new URL("../../parity/", import.meta.url);

export function parityDir(name: string): string {
  return decodeURIComponent(new URL(name + "/", PARITY).pathname).replace(/^\/([A-Za-z]:)/, "$1");
}

export function fsSource(directory: string): FileSource {
  return async (name) => new Uint8Array(readFileSync(join(directory, name)));
}

export function scenarioNames(): string[] {
  const root = decodeURIComponent(PARITY.pathname).replace(/^\/([A-Za-z]:)/, "$1");
  return readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory() && d.name !== "hash-vectors")
    .map((d) => d.name)
    .sort();
}

export function assertSameNumbers(actual: ArrayLike<number>, expected: ArrayLike<number>, what: string): void {
  assert.equal(actual.length, expected.length, `${what}: length ${actual.length} vs ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    if (!Object.is(actual[i], expected[i]) && actual[i] !== expected[i]) {
      assert.fail(`${what}: differs at ${i}: ${actual[i]} vs ${expected[i]}`);
    }
  }
}

export function maxAbsDifference(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  assert.equal(actual.length, expected.length);
  let worst = 0;
  for (let i = 0; i < actual.length; i++) worst = Math.max(worst, Math.abs((actual[i] as number) - (expected[i] as number)));
  return worst;
}
