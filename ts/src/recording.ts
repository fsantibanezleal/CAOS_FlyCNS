/**
 * Recordings (`flycns.record`) and expected outputs of parity scenarios (`flycns.bundle.write_expected`), read with
 * the same verification as the compiled graph.
 */

import { Compiled, type FileSource, readCompiled } from "./compiled.js";

export const RECORDING_SCHEMA = "flycns.recording/1";
export const EXPECTED_SCHEMA = "flycns.expected/1";

/** The result of a spiking run: spikes as a CSR over steps, and the requested voltage traces. */
export interface Run {
  nNeurons: number;
  steps: number;
  dtMs: number;
  /** length steps + 1: the spikes of step k are neuronIndex[tickIndptr[k] .. tickIndptr[k + 1]) */
  tickIndptr: Float64Array;
  neuronIndex: Int32Array;
  traceNeurons: Int32Array;
  /** row-major (steps, traceNeurons.length) */
  tracesMv: Float32Array;
}

export function spikeCounts(run: Run): Int32Array {
  const counts = new Int32Array(run.nNeurons);
  for (let i = 0; i < run.neuronIndex.length; i++) counts[run.neuronIndex[i] as number]++;
  return counts;
}

export function ratesHz(run: Run): Float64Array {
  const counts = spikeCounts(run);
  const seconds = (run.steps * run.dtMs) / 1000;
  return Float64Array.from(counts, (c) => c / seconds);
}

/** (step of each spike, neuron of each spike), ordered by step then neuron. */
export function spikeTimes(run: Run): { step: Int32Array; neuron: Int32Array } {
  const step = new Int32Array(run.neuronIndex.length);
  for (let k = 0; k < run.steps; k++) {
    step.fill(k, run.tickIndptr[k] as number, run.tickIndptr[k + 1] as number);
  }
  return { step, neuron: run.neuronIndex };
}

export async function readRecording(source: FileSource): Promise<Run> {
  const c = await readCompiled(source, { schema: RECORDING_SCHEMA });
  const counts = c.counts as { neurons: number; steps: number; dt_ms: number };
  return {
    nNeurons: counts.neurons,
    steps: counts.steps,
    dtMs: counts.dt_ms,
    tickIndptr: c.numbers("tick_indptr"),
    neuronIndex: c.int32("neuron_index"),
    traceNeurons: c.int32("trace_neurons"),
    tracesMv: c.float32("traces_mv"),
  };
}

export async function readExpected(source: FileSource): Promise<Compiled> {
  return readCompiled(source, { schema: EXPECTED_SCHEMA });
}
