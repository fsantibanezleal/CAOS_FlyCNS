#!/usr/bin/env node
/**
 * Measure the browser engines on the whole CNS against the NumPy reference (SDD section 7): the TypeScript CPU engine
 * (the worker fallback) and the WebGPU engine run the scenario written by scripts/write_whole_cns_scenario.py, and
 * each is compared with the reference's output by the active-neuron Jaccard index and the correlation of per-neuron
 * spike counts over the neurons active in either run; the CPU engine is also checked spike for spike. Under the
 * strong drive (strong/), five GPU runs (seeds 0 to 4) are compared with five reference runs (seeds 5 to 9) by the
 * correlation of mean rates, which must reach the 5th percentile of the reference against itself; one CPU run
 * (seed 0) must give the reference's seed-0 rates exactly.
 *
 * Needs the package built (npm run build) and, for the GPU, Dawn installed locally: npm install --no-save webgpu.
 *
 * Usage: node scripts/measure_browser.mjs SCENARIO_DIR OUT_JSON
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { LIFGpu, lifFromBundle, loadScenario, readExpected, requestDevice, VERSION } from "../dist/index.js";

const [scenarioDir, outJson] = process.argv.slice(2);
if (!scenarioDir || !outJson) {
  console.error("usage: node scripts/measure_browser.mjs SCENARIO_DIR OUT_JSON");
  process.exit(2);
}
const source = (dir) => async (name) => new Uint8Array(readFileSync(join(dir, name)));

function counts(run) {
  const c = new Int32Array(run.nNeurons);
  for (const i of run.neuronIndex) c[i] += 1;
  return c;
}

function compare(a, b) {
  let both = 0;
  let either = 0;
  const x = [];
  const y = [];
  for (let i = 0; i < a.length; i++) {
    const ia = a[i] > 0;
    const ib = b[i] > 0;
    if (ia && ib) both += 1;
    if (ia || ib) {
      either += 1;
      x.push(a[i]);
      y.push(b[i]);
    }
  }
  const mean = (v) => v.reduce((s, t) => s + t, 0) / v.length;
  const mx = mean(x);
  const my = mean(y);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let j = 0; j < x.length; j++) {
    sxy += (x[j] - mx) * (y[j] - my);
    sxx += (x[j] - mx) ** 2;
    syy += (y[j] - my) ** 2;
  }
  return { jaccard: both / Math.max(either, 1), correlation: sxy / Math.sqrt(sxx * syy) };
}

/** As tests/test_lif.py: the correlation of mean rates over the neurons active in either set of runs. */
function meanRateCorrelation(xs, ys) {
  const n = xs[0].length;
  const mean = (rows) => {
    const m = new Float64Array(n);
    for (const r of rows) for (let i = 0; i < n; i++) m[i] += r[i] / rows.length;
    return m;
  };
  const mx = mean(xs);
  const my = mean(ys);
  const a = [];
  const b = [];
  for (let i = 0; i < n; i++) {
    if (mx[i] > 0 || my[i] > 0) {
      a.push(mx[i]);
      b.push(my[i]);
    }
  }
  const avg = (v) => v.reduce((s, t) => s + t, 0) / v.length;
  const ma = avg(a);
  const mb = avg(b);
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let j = 0; j < a.length; j++) {
    sab += (a[j] - ma) * (b[j] - mb);
    saa += (a[j] - ma) ** 2;
    sbb += (b[j] - mb) ** 2;
  }
  return sab / Math.sqrt(saa * sbb);
}

function ratesHz(run) {
  const c = counts(run);
  const seconds = (run.steps * run.dtMs) / 1000;
  return Float64Array.from(c, (x) => x / seconds);
}

function firstDifference(run, tickIndptr, neuronIndex) {
  for (let k = 0; k < run.steps; k++) {
    const a0 = run.tickIndptr[k];
    const a1 = run.tickIndptr[k + 1];
    const b0 = Number(tickIndptr[k]);
    const b1 = Number(tickIndptr[k + 1]);
    if (a1 - a0 !== b1 - b0) return { step: k, spikesBefore: a0 };
    for (let j = 0; j < a1 - a0; j++) if (run.neuronIndex[a0 + j] !== neuronIndex[b0 + j]) return { step: k, spikesBefore: a0 };
  }
  return null;
}

const t0 = performance.now();
const scenario = await loadScenario(source(scenarioDir));
const expected = await readExpected(source(join(scenarioDir, "expected")));
const loadSeconds = (performance.now() - t0) / 1000;
const refTick = expected.numbers("tick_indptr");
const refNeuron = expected.int32("neuron_index");
const reference = { nNeurons: scenario.bundle.release.n_neurons, neuronIndex: refNeuron };
const refCounts = counts(reference);
const { steps, seed } = scenario.run;
const result = {
  version: VERSION,
  scenario: { steps, seed, neurons: reference.nNeurons, drive: scenario.bundle.release.drive_name },
  load_seconds: +loadSeconds.toFixed(1),
  reference: { spikes: refNeuron.length, active: refCounts.filter((c) => c > 0).length },
};

// the TypeScript CPU engine, float64: the reference's operations in the reference's order
{
  const engine = lifFromBundle(scenario.bundle);
  const t = performance.now();
  const run = engine.run(steps, scenario.drive, seed, scenario.traceNeurons);
  const seconds = (performance.now() - t) / 1000;
  const diff = firstDifference(run, refTick, refNeuron);
  result.cpu = {
    seconds: +seconds.toFixed(1),
    spikes: run.neuronIndex.length,
    identical: diff === null,
    first_difference: diff,
    ...compare(refCounts, counts(run)),
  };
  console.log("cpu", JSON.stringify(result.cpu));
}

const strongData = await readExpected(source(join(scenarioDir, "strong")));
const strongRelease = strongData.release;
const strongDrive = { activate: new Map(Array.from(strongData.numbers("gustatory"), (i) => [i, strongRelease.rate_hz])) };
const referenceRates = strongData.float32("reference_rates_hz");
const n = reference.nNeurons;
const refRow = (s) => Float64Array.from(referenceRates.subarray(s * n, (s + 1) * n));
result.strong = {
  steps: strongRelease.steps,
  neurons_driven: strongDrive.activate.size,
  reference_spikes: strongRelease.reference_spikes,
  split_p5: strongRelease.split_p5,
  split_median: strongRelease.split_median,
};
{
  const engine = lifFromBundle(scenario.bundle);
  const t = performance.now();
  const run = engine.run(strongRelease.steps, strongDrive, 0);
  const cpuRates = ratesHz(run);
  const ref0 = refRow(0);
  // the reference's rates were stored as float32; compare the counts they stand for
  const seconds = (strongRelease.steps * run.dtMs) / 1000;
  let same = true;
  for (let i = 0; i < n; i++) if (Math.round(ref0[i] * seconds) !== Math.round(cpuRates[i] * seconds)) same = false;
  result.strong.cpu_seed0 = { seconds: +((performance.now() - t) / 1000).toFixed(1), spikes: run.neuronIndex.length, same_counts_as_reference: same };
  console.log("cpu strong", JSON.stringify(result.strong.cpu_seed0));
}

// the WebGPU engine, float32 with fixed-point delivery
{
  let webgpu;
  try {
    webgpu = await import("webgpu");
  } catch {
    webgpu = undefined;
  }
  if (!webgpu) {
    result.gpu = { run: false, reason: "the webgpu package is not installed (npm install --no-save webgpu)" };
  } else {
    Object.assign(globalThis, webgpu.globals);
    const dawn = webgpu.create([]);
    globalThis.__flycnsDawn = dawn; // held for the life of the process
    const adapter = await dawn.requestAdapter({ powerPreference: "high-performance" });
    const device = await requestDevice(dawn);
    const t = performance.now();
    const engine = LIFGpu.fromBundle(device, scenario.bundle);
    const setup = (performance.now() - t) / 1000;
    const t2 = performance.now();
    const run = await engine.run(steps, scenario.drive, seed, scenario.traceNeurons, { batchSteps: 500 });
    const seconds = (performance.now() - t2) / 1000;
    const diff = firstDifference(run, refTick, refNeuron);
    result.gpu = {
      adapter: `${adapter.info.vendor} ${adapter.info.architecture} (${adapter.info.description})`,
      fixed_point_scale: engine.scale,
      setup_seconds: +setup.toFixed(1),
      seconds: +seconds.toFixed(1),
      spikes: run.neuronIndex.length,
      identical: diff === null,
      first_difference: diff,
      ...compare(refCounts, counts(run)),
    };
    console.log("gpu", JSON.stringify(result.gpu));
    const trials = [];
    const spikes = [];
    const t3 = performance.now();
    for (let s = 0; s < 5; s++) {
      const r = await engine.run(strongRelease.steps, strongDrive, s, new Int32Array(0), { batchSteps: 250 });
      trials.push(ratesHz(r));
      spikes.push(r.neuronIndex.length);
    }
    const correlation = meanRateCorrelation([5, 6, 7, 8, 9].map(refRow), trials);
    result.strong.gpu = {
      seeds: [0, 1, 2, 3, 4],
      spikes,
      seconds: +((performance.now() - t3) / 1000).toFixed(1),
      correlation_with_reference_seeds_5_to_9: correlation,
      passes: correlation >= strongRelease.split_p5,
    };
    console.log("gpu strong", JSON.stringify(result.strong.gpu));
    engine.destroy();
  }
}

writeFileSync(outJson, JSON.stringify(result, null, 1) + "\n");
console.log(JSON.stringify(result, null, 1));
process.exit(0);
