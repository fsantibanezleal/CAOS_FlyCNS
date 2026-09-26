import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { lifFromBundle, loadScenario } from "../src/bundle.js";
import { createWorkerHandler, type Files, type WorkerResponse } from "../src/worker.js";
import { assertSameNumbers, fsSource, parityDir } from "./files.js";

function files(directory: string): Files {
  const out: Files = {};
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (entry.isFile()) out[entry.name] = new Uint8Array(readFileSync(join(directory, entry.name)));
  }
  return out;
}

test("the worker protocol gives the engines spikes", async () => {
  const responses: WorkerResponse[] = [];
  const handle = createWorkerHandler((r) => responses.push(r));
  const directory = parityDir("lif-poisson");
  await handle({ type: "load", scenario: files(directory) });
  assert.deepEqual(responses[0], { type: "loaded", engine: "lif", n: 60 });
  await handle({ type: "run" });
  const done = responses[1];
  assert.ok(done && done.type === "done");
  const scenario = await loadScenario(fsSource(directory));
  const direct = lifFromBundle(scenario.bundle).run(scenario.run.steps, scenario.drive, scenario.run.seed, scenario.traceNeurons);
  assertSameNumbers(done.run.neuronIndex, direct.neuronIndex, "spikes");
  assertSameNumbers(done.run.tracesMv, direct.tracesMv, "traces");
  // stepping a started run in batches gives the same spikes
  await handle({ type: "start", drive: scenario.drive, seed: scenario.run.seed });
  await handle({ type: "step", count: 1200 });
  await handle({ type: "step", count: 800 });
  await handle({ type: "finish" });
  const batches = responses.filter((r) => r.type === "spikes");
  assert.equal(batches.length, 2);
  const last = responses[responses.length - 1];
  assert.ok(last && last.type === "done");
  assertSameNumbers(last.run.neuronIndex, direct.neuronIndex, "stepped spikes");
  // a hybrid scenario answers with per-frame spikes and the graded activity
  responses.length = 0;
  await handle({ type: "load", scenario: files(parityDir("hybrid-toy")) });
  await handle({ type: "run" });
  const frames = responses.filter((r) => r.type === "spikes");
  const hybridDone = responses[responses.length - 1];
  assert.equal(frames.length, 100);
  assert.ok(hybridDone && hybridDone.type === "done" && hybridDone.hybrid && hybridDone.hybrid.graded.length === 800);
  // errors come back as messages, never as exceptions
  await handle({ type: "step", count: 1 });
  assert.equal(responses[responses.length - 1]?.type, "error");
});
