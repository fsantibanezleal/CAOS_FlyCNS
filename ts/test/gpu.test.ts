import { test } from "node:test";
import assert from "node:assert/strict";
import { loadScenario } from "../src/bundle.js";
import { readExpected } from "../src/recording.js";
import { fixedPointScale, quantise } from "../src/gpu/device.js";
import { GradedGpu } from "../src/gpu/graded-gpu.js";
import { HybridGpu } from "../src/gpu/hybrid-gpu.js";
import { LIFGpu } from "../src/gpu/lif-gpu.js";
import { NO_GPU, assertSameNumbers, fsSource, gpuDevice, maxAbsDifference, parityDir, scenarioNames } from "./files.js";

/** SDD section 7: graded activity on the GPU within 1e-4 of the reference. */
const GRADED_TOLERANCE = 1e-4;
/** Traced LIF voltages (mV): float32 state against the float64 reference rounded to float32. */
const TRACE_TOLERANCE_MV = 1e-3;

test("the gpu engines reproduce the reference on the scenarios", async (t) => {
  const device = await gpuDevice();
  if (!device) {
    t.skip(NO_GPU);
    return;
  }
  const names = scenarioNames();
  const lif = names.filter((n) => n.startsWith("lif-"));
  assert.ok(lif.length >= 5, `lif scenarios: ${lif}`);
  for (const name of lif) {
    const scenario = await loadScenario(fsSource(parityDir(name)));
    const expected = await readExpected(fsSource(parityDir(name + "/expected")));
    const engine = LIFGpu.fromBundle(device, scenario.bundle);
    // an odd batch length, so batch boundaries fall inside the delay and the events
    const run = await engine.run(scenario.run.steps, scenario.drive, scenario.run.seed, scenario.traceNeurons, {
      batchSteps: 97,
    });
    engine.destroy();
    assertSameNumbers(run.tickIndptr, expected.numbers("tick_indptr"), `${name} tick_indptr`);
    assertSameNumbers(run.neuronIndex, expected.int32("neuron_index"), `${name} neuron_index`);
    const worst = maxAbsDifference(run.tracesMv, expected.float32("traces_mv"));
    assert.ok(worst <= TRACE_TOLERANCE_MV, `${name} traces differ by ${worst} mV`);
  }

  for (const name of ["hybrid-toy", "hybrid-lattice-toy"]) {
    const scenario = await loadScenario(fsSource(parityDir(name)));
    const expected = await readExpected(fsSource(parityDir(name + "/expected")));
    const engine = HybridGpu.fromBundle(device, scenario.bundle);
    assert.equal(engine.graded === undefined, name === "hybrid-lattice-toy");
    const out = await engine.run(scenario.intensity as Float64Array, scenario.run.steps, {
      drive: scenario.drive,
      seed: scenario.run.seed,
      preSteps: scenario.run.pre_steps as number,
      grey: scenario.run.grey,
      recordGraded: scenario.recordGraded,
      traceSpiking: scenario.traceNeurons,
      framesPerBatch: 3,
    });
    engine.destroy();
    assertSameNumbers(out.spikes.tickIndptr, expected.numbers("tick_indptr"), `${name} tick_indptr`);
    assertSameNumbers(out.spikes.neuronIndex, expected.int32("neuron_index"), `${name} neuron_index`);
    const graded = maxAbsDifference(out.graded, expected.float32("graded"));
    assert.ok(graded <= GRADED_TOLERANCE, `${name} graded activity differs by ${graded}`);
    const grey = maxAbsDifference(out.greyRelease, expected.float32("grey_release"));
    assert.ok(grey <= GRADED_TOLERANCE, `${name} grey release differs by ${grey}`);
    assert.ok(out.spikes.neuronIndex.length > 10);
  }

  const scenario = await loadScenario(fsSource(parityDir("graded-random")));
  const expected = await readExpected(fsSource(parityDir("graded-random/expected")));
  const graded = GradedGpu.fromBundle(device, scenario.bundle);
  const out = await graded.run(scenario.intensity as Float64Array, scenario.run.steps);
  graded.destroy();
  const activity = maxAbsDifference(out.activity, expected.float32("activity"));
  assert.ok(activity <= GRADED_TOLERANCE, `graded activity differs by ${activity}`);
  const final = maxAbsDifference(out.final, expected.float64("final_state"));
  assert.ok(final <= GRADED_TOLERANCE, `graded final state differs by ${final}`);
});

test("the fixed point scale bounds every neuron's delivery within the int32 range", async () => {
  for (const name of scenarioNames().filter((n) => n.startsWith("lif-"))) {
    const scenario = await loadScenario(fsSource(parityDir(name)));
    const net = scenario.bundle.lifNetwork();
    const n = net.indptr.length - 1;
    const scale = fixedPointScale(net.indices, net.weightsMv, n);
    assert.equal(Math.log2(scale) % 1, 0, `${name}: the scale ${scale} is a power of two`);
    const q = quantise(net.weightsMv, scale);
    const incoming = new Float64Array(n);
    for (let e = 0; e < q.length; e++) {
      const target = net.indices[e] as number;
      incoming[target] = (incoming[target] as number) + Math.abs(q[e] as number);
      assert.ok(Math.abs((q[e] as number) / scale - (net.weightsMv[e] as number)) <= 0.5 / scale);
    }
    assert.ok(Math.max(...incoming) <= 2 ** 31 - 1, `${name}: a neuron's largest possible input overflows int32`);
  }
});
