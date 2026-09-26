import { test } from "node:test";
import assert from "node:assert/strict";
import { gradedFromBundle, hybridFromBundle, lifFromBundle, loadScenario } from "../src/bundle.js";
import { readExpected } from "../src/recording.js";
import { assertSameNumbers, fsSource, parityDir, scenarioNames } from "./files.js";

const names = scenarioNames();

test("the cpu lif engine reproduces the reference on every lif scenario", async () => {
  const lif = names.filter((n) => n.startsWith("lif-"));
  assert.ok(lif.length >= 5, `lif scenarios: ${lif}`);
  for (const name of lif) {
    const scenario = await loadScenario(fsSource(parityDir(name)));
    const expected = await readExpected(fsSource(parityDir(name + "/expected")));
    const engine = lifFromBundle(scenario.bundle);
    const run = engine.run(scenario.run.steps, scenario.drive, scenario.run.seed, scenario.traceNeurons);
    assertSameNumbers(run.tickIndptr, expected.numbers("tick_indptr"), `${name} tick_indptr`);
    assertSameNumbers(run.neuronIndex, expected.int32("neuron_index"), `${name} neuron_index`);
    assertSameNumbers(run.tracesMv, expected.float32("traces_mv"), `${name} traces`);
    assert.ok(run.neuronIndex.length > 50, `${name} is active`);
  }
});

test("the cpu graded engine reproduces the reference", async () => {
  const scenario = await loadScenario(fsSource(parityDir("graded-random")));
  const expected = await readExpected(fsSource(parityDir("graded-random/expected")));
  const engine = gradedFromBundle(scenario.bundle);
  const out = engine.run(scenario.intensity as Float64Array, scenario.run.steps);
  assertSameNumbers(out.activity, expected.float32("activity"), "activity");
  assertSameNumbers(out.final, expected.float64("final_state"), "final state");
  assert.ok(expected.float32("activity").some((x) => Math.abs(x) > 0.1));
});

test("the cpu hybrid reproduces the reference with both sources", async () => {
  for (const name of ["hybrid-toy", "hybrid-lattice-toy"]) {
    const scenario = await loadScenario(fsSource(parityDir(name)));
    const expected = await readExpected(fsSource(parityDir(name + "/expected")));
    const engine = hybridFromBundle(scenario.bundle);
    assert.equal(engine.source.constructor.name, name === "hybrid-toy" ? "OwnSource" : "LatticeSource");
    const out = engine.run(scenario.intensity as Float64Array, scenario.run.steps, {
      drive: scenario.drive,
      seed: scenario.run.seed,
      preSteps: scenario.run.pre_steps as number,
      grey: scenario.run.grey,
      recordGraded: scenario.recordGraded,
      traceSpiking: scenario.traceNeurons,
    });
    assertSameNumbers(out.spikes.tickIndptr, expected.numbers("tick_indptr"), `${name} tick_indptr`);
    assertSameNumbers(out.spikes.neuronIndex, expected.int32("neuron_index"), `${name} neuron_index`);
    assertSameNumbers(out.spikes.tracesMv, expected.float32("traces_mv"), `${name} traces`);
    assertSameNumbers(out.graded, expected.float32("graded"), `${name} graded`);
    assertSameNumbers(out.greyRelease, expected.float32("grey_release"), `${name} grey release`);
    assert.ok(out.spikes.neuronIndex.length > 10);
  }
});
