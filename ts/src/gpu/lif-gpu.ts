/**
 * The published LIF model on WebGPU, in float32, with the delivery summed in int32 fixed point. One step is the CPU
 * engine's step in the same order, as seven kernels in one compute pass (integrate, deliver, apply, events when the
 * step has any, reset, trace when neurons are traced, tick); a batch of steps is one command buffer, and the spikes
 * and traces of a batch are read back once. Results are compared with the reference by tolerance (SDD section 7),
 * never claimed bit-identical: float32 and float64 part at the first threshold the two decide differently.
 */

import type { Bundle } from "../bundle.js";
import type { Drive, LIFConstants, LIFNetwork } from "../lif.js";
import type { Run } from "../recording.js";
import { eventThreshold } from "../rng.js";
import {
  bindGroup,
  fixedPointScale,
  groups,
  pipeline,
  quantise,
  readBuffers,
  storageBuffer,
  uniformBuffer,
} from "./device.js";
import { LIF_WGSL } from "./shaders.js";

const MUTE = 1;
const ACT = 8;
const ADAPTING = 1;
const HAS_GINPUT = 2;
const HAS_RATE = 4;
const HAS_MOD = 8;
const NEURON_WORDS = 6;
/** The last spike of a neuron that never spiked: far enough back that it is free from step 0. */
const LAST_NEVER = -1_000_000_000;

type Kernel = keyof typeof LIF_WGSL;

export interface LIFGpuOptions {
  /** steps per command buffer; spikes and traces are read back once per batch (default 500, 50 ms) */
  batchSteps?: number;
  /** an extra input to g, one float per neuron, filled by the caller's kernels before each step (the hybrid's bridge) */
  gInput?: GPUBuffer;
  /** keep the filtered spike rate the hybrid's feedback reads: rate = rate x decay, plus amount on a spike */
  rate?: { decay: number; amount: number };
  /** the most spikes one batch may hold (default n x batchSteps, at most 4,194,304) */
  spikeCapacity?: number;
}

/** One run's state on the device and the spikes and traces read back so far. */
export class LIFGpuRun {
  /** the absolute step the next encoded step will take */
  k = 0;
  /** steps encoded in the batch not yet collected */
  kb = 0;
  readonly spikes: Int32Array[] = [];
  readonly traces: Float32Array[] = [];

  constructor(
    readonly buffers: Record<string, GPUBuffer>,
    readonly binds: Record<Kernel, GPUBindGroup>,
    readonly eventSteps: Set<number>,
    readonly traceNeurons: Int32Array,
    readonly batchSteps: number,
    readonly spikeCapacity: number,
    private readonly owned: GPUBuffer[],
  ) {}

  /** The filtered spike rate, one float per neuron (zeros unless the run keeps it). */
  get rate(): GPUBuffer {
    return this.buffers.rate as GPUBuffer;
  }

  destroy(): void {
    for (const b of this.owned) b.destroy();
  }
}

/** A run of the CPU engines' shape from per-step spike lists and traces. */
export function assembleRun(
  nNeurons: number,
  dtMs: number,
  spikes: Int32Array[],
  traces: Float32Array[],
  traceNeurons: Int32Array,
): Run {
  const steps = spikes.length;
  const tickIndptr = new Float64Array(steps + 1);
  let total = 0;
  for (let k = 0; k < steps; k++) {
    tickIndptr[k] = total;
    total += (spikes[k] as Int32Array).length;
  }
  tickIndptr[steps] = total;
  const neuronIndex = new Int32Array(total);
  let at = 0;
  for (const s of spikes) {
    neuronIndex.set(s, at);
    at += s.length;
  }
  const nTrace = traceNeurons.length;
  const tracesMv = new Float32Array(traces.length * nTrace);
  for (let k = 0; k < traces.length; k++) tracesMv.set(traces[k] as Float32Array, k * nTrace);
  return { nNeurons, steps, dtMs, tickIndptr, neuronIndex, traceNeurons, tracesMv };
}

export class LIFGpu {
  readonly n: number;
  /** the fixed-point scale of the delivery (a power of two) */
  readonly scale: number;
  private readonly pipelines: Record<Kernel, GPUComputePipeline>;
  private readonly csr: { indptr: GPUBuffer; indices: GPUBuffer; weights: GPUBuffer };

  constructor(
    readonly device: GPUDevice,
    network: LIFNetwork,
    readonly constants: LIFConstants,
  ) {
    this.n = network.indptr.length - 1;
    this.scale = fixedPointScale(network.indices, network.weightsMv, this.n);
    const indices = network.indices;
    this.csr = {
      indptr: storageBuffer(device, Uint32Array.from(network.indptr), "lif indptr"),
      indices: storageBuffer(device, new Uint32Array(indices.buffer, indices.byteOffset, indices.length), "lif indices"),
      weights: storageBuffer(device, quantise(network.weightsMv, this.scale), "lif weights"),
    };
    const entries = Object.entries(LIF_WGSL) as [Kernel, string][];
    this.pipelines = Object.fromEntries(entries.map(([k, code]) => [k, pipeline(device, code, `lif ${k}`)])) as Record<
      Kernel,
      GPUComputePipeline
    >;
  }

  static fromBundle(device: GPUDevice, bundle: Bundle): LIFGpu {
    const constants = bundle.release.lif;
    if (!constants) throw new Error("the bundle has no LIF constants");
    return new LIFGpu(device, bundle.lifNetwork(), constants);
  }

  /** The state of a run under `drive`, on the device; nothing is dispatched yet. */
  start(
    drive: Drive = {},
    seed = 0,
    traceNeurons: Int32Array = new Int32Array(0),
    options: LIFGpuOptions = {},
  ): LIFGpuRun {
    const { device, n } = this;
    const c = this.constants;
    const batchSteps = options.batchSteps ?? 500;
    const slots = c.delay_steps + 1;

    const state = new ArrayBuffer(n * NEURON_WORDS * 4);
    const f32 = new Float32Array(state);
    const i32 = new Int32Array(state);
    const u32 = new Uint32Array(state);
    for (let i = 0; i < n; i++) {
      f32[NEURON_WORDS * i] = c.v0_mv;
      i32[NEURON_WORDS * i + 3] = LAST_NEVER;
      i32[NEURON_WORDS * i + 4] = c.refractory_steps;
    }
    // activated neurons: thresholds as the reference computes them, and no refractory period
    const actThr = new Uint32Array(n);
    const dtS = c.dt_ms / 1000.0;
    for (const [i, rate] of drive.activate ?? new Map<number, number>()) {
      u32[NEURON_WORDS * i + 5] = (u32[NEURON_WORDS * i + 5] as number) | ACT;
      i32[NEURON_WORDS * i + 4] = 0;
      actThr[i] = eventThreshold(rate, dtS);
    }
    if (drive.silenced) {
      for (let j = 0; j < drive.silenced.length; j++) {
        const i = drive.silenced[j] as number;
        u32[NEURON_WORDS * i + 5] = (u32[NEURON_WORDS * i + 5] as number) | MUTE;
      }
    }
    // modulated Poisson input: each neuron's slot, and the thresholds per frame as the reference computes them
    const modSlot = new Int32Array(n).fill(-1);
    let modThr = new Uint32Array(0);
    const modulated = drive.modulated;
    if (modulated) {
      modulated.neurons.forEach((i, j) => {
        if (modSlot[i] !== -1) throw new Error(`neuron ${i} appears twice in the modulated drive`);
        modSlot[i] = j;
        i32[NEURON_WORDS * i + 4] = 0;
      });
      modThr = Uint32Array.from(modulated.rates, (r) => {
        const p = Math.min(Math.max((r * c.dt_ms) / 1000.0, 0), 1);
        return Math.min(Math.floor(p * 4294967296), 4294967295);
      });
    }
    // fixed events, per step in their order
    const events = drive.events ?? new Map<number, { neurons: ArrayLike<number>; dvMv: ArrayLike<number> }>();
    const eventSteps = new Set<number>(events.keys());
    const lastStep = eventSteps.size ? Math.max(...eventSteps) : -1;
    const eventIndptr = new Uint32Array(lastStep + 2);
    let eventCount = 0;
    for (let k = 0; k <= lastStep; k++) {
      eventIndptr[k] = eventCount;
      eventCount += events.get(k)?.neurons.length ?? 0;
    }
    if (lastStep >= 0) eventIndptr[lastStep + 1] = eventCount;
    const eventData = new ArrayBuffer(eventCount * 8);
    const eventNeuron = new Uint32Array(eventData);
    const eventDv = new Float32Array(eventData);
    let at = 0;
    for (let k = 0; k <= lastStep; k++) {
      const e = events.get(k);
      if (!e) continue;
      for (let j = 0; j < e.neurons.length; j++) {
        eventNeuron[2 * at] = e.neurons[j] as number;
        eventDv[2 * at + 1] = e.dvMv[j] as number;
        at++;
      }
    }

    const spikeCapacity = Math.max(1, options.spikeCapacity ?? Math.min(n * batchSteps, 1 << 22));
    let flags = 0;
    if (c.adaptation_mv !== 0) flags |= ADAPTING;
    if (options.gInput) flags |= HAS_GINPUT;
    if (options.rate) flags |= HAS_RATE;
    if (modulated) flags |= HAS_MOD;
    const params = new ArrayBuffer(24 * 4);
    const pu = new Uint32Array(params);
    const pf = new Float32Array(params);
    pu[0] = n;
    pu[1] = c.delay_steps;
    pu[2] = slots;
    pu[3] = seed >>> 0;
    pf[4] = c.v0_mv;
    pf[5] = c.v_rst_mv;
    pf[6] = c.v_th_mv;
    pf[7] = c.decay_a;
    pf[8] = c.decay_b;
    pf[9] = c.decay_c;
    pf[10] = c.w_event_mv;
    pf[11] = c.adaptation_mv;
    pf[12] = c.adaptation_decay;
    pf[13] = 1 / this.scale;
    pf[14] = options.rate?.decay ?? 1;
    pf[15] = options.rate?.amount ?? 0;
    pu[16] = modulated ? modulated.neurons.length : 0;
    pu[17] = modulated ? modulated.stepsPerFrame : 1;
    pu[18] = modulated ? modulated.frames : 1;
    pu[19] = traceNeurons.length;
    pu[20] = flags;
    pu[21] = spikeCapacity;

    const buffers: Record<string, GPUBuffer> = {
      params: uniformBuffer(device, params, "lif params"),
      clock: storageBuffer(device, new Uint32Array(2), "lif clock"),
      neurons: storageBuffer(device, u32, "lif neurons"),
      ring: storageBuffer(device, slots * n * 4, "lif ring"),
      acc: storageBuffer(device, n * 4, "lif accumulators"),
      actThr: storageBuffer(device, actThr, "lif activation thresholds"),
      modSlot: storageBuffer(device, modSlot, "lif modulated slots"),
      modThr: storageBuffer(device, modThr, "lif modulated thresholds"),
      eventIndptr: storageBuffer(device, eventIndptr, "lif event indptr"),
      events: storageBuffer(device, eventNeuron, "lif events"),
      spikes: storageBuffer(device, spikeCapacity * 8, "lif spikes"),
      counter: storageBuffer(device, 4, "lif spike counter"),
      rate: storageBuffer(device, options.rate ? n * 4 : 4, "lif rate"),
      traceNeurons: storageBuffer(device, new Uint32Array(traceNeurons), "lif trace neurons"),
      traces: storageBuffer(device, batchSteps * traceNeurons.length * 4, "lif traces"),
    };
    const owned = Object.values(buffers);
    buffers.gInput = options.gInput ?? storageBuffer(device, 4, "lif no extra input");
    if (!options.gInput) owned.push(buffers.gInput);
    const b = buffers as Record<string, GPUBuffer> & typeof buffers;
    const p = this.pipelines;
    const bind = (k: Kernel, list: GPUBuffer[]) => bindGroup(device, p[k], list, `lif ${k}`);
    const binds: Record<Kernel, GPUBindGroup> = {
      integrate: bind("integrate", [b.params, b.clock, b.neurons, b.ring] as GPUBuffer[]),
      deliver: bind("deliver", [b.params, b.clock, b.ring, this.csr.indptr, this.csr.indices, this.csr.weights, b.acc] as GPUBuffer[]),
      apply: bind("apply", [b.params, b.clock, b.neurons, b.acc, b.gInput, b.actThr, b.modSlot, b.modThr] as GPUBuffer[]),
      events: bind("events", [b.clock, b.neurons, b.eventIndptr, b.events] as GPUBuffer[]),
      reset: bind("reset", [b.params, b.clock, b.neurons, b.spikes, b.counter, b.rate] as GPUBuffer[]),
      trace: bind("trace", [b.params, b.clock, b.neurons, b.traceNeurons, b.traces] as GPUBuffer[]),
      tick: bind("tick", [b.clock] as GPUBuffer[]),
    };
    return new LIFGpuRun(buffers, binds, eventSteps, traceNeurons, batchSteps, spikeCapacity, owned);
  }

  /** Encode `steps` steps of `run` into `encoder`, within the current batch. */
  encode(run: LIFGpuRun, encoder: GPUCommandEncoder, steps: number): void {
    if (run.kb + steps > run.batchSteps) {
      throw new Error(`a batch holds ${run.batchSteps} steps; ${run.kb} are encoded and ${steps} more were asked`);
    }
    const pass = encoder.beginComputePass({ label: "lif steps" });
    const perNeuron = groups(this.n);
    const perTrace = groups(run.traceNeurons.length);
    const dispatch = (k: Kernel, count: number) => {
      pass.setPipeline(this.pipelines[k]);
      pass.setBindGroup(0, run.binds[k]);
      pass.dispatchWorkgroups(count);
    };
    for (let s = 0; s < steps; s++) {
      dispatch("integrate", perNeuron);
      dispatch("deliver", perNeuron);
      dispatch("apply", perNeuron);
      if (run.eventSteps.has(run.k)) dispatch("events", 1);
      dispatch("reset", perNeuron);
      if (run.traceNeurons.length) dispatch("trace", perTrace);
      dispatch("tick", 1);
      run.k += 1;
      run.kb += 1;
    }
    pass.end();
  }

  /** Read the batch's spikes (sorted per step) and traces back into the run, and start a new batch. */
  async collect(run: LIFGpuRun): Promise<void> {
    const { device } = this;
    const nTrace = run.traceNeurons.length;
    const b = run.buffers;
    const [countBytes, traceBytes] = await readBuffers(device, [
      { buffer: b.counter as GPUBuffer, byteLength: 4 },
      { buffer: b.traces as GPUBuffer, byteLength: run.kb * nTrace * 4 },
    ]);
    const count = new Uint32Array(countBytes as ArrayBuffer)[0] as number;
    if (count > run.spikeCapacity) {
      throw new Error(
        `a batch fired ${count} spikes, over its capacity of ${run.spikeCapacity}; use shorter batches or a larger spikeCapacity`,
      );
    }
    const buckets: number[][] = Array.from({ length: run.kb }, () => []);
    if (count > 0) {
      const [list] = await readBuffers(device, [{ buffer: b.spikes as GPUBuffer, byteLength: count * 8 }]);
      const pairs = new Uint32Array(list as ArrayBuffer);
      for (let j = 0; j < count; j++) (buckets[pairs[2 * j] as number] as number[]).push(pairs[2 * j + 1] as number);
    }
    for (const bucket of buckets) run.spikes.push(Int32Array.from(bucket).sort());
    if (nTrace) {
      const all = new Float32Array(traceBytes as ArrayBuffer);
      for (let kb = 0; kb < run.kb; kb++) run.traces.push(all.slice(kb * nTrace, (kb + 1) * nTrace));
    }
    device.queue.writeBuffer(b.counter as GPUBuffer, 0, new Uint32Array([0]));
    device.queue.writeBuffer(b.clock as GPUBuffer, 4, new Uint32Array([0]));
    run.kb = 0;
  }

  finish(run: LIFGpuRun): Run {
    return assembleRun(this.n, this.constants.dt_ms, run.spikes, run.traces, run.traceNeurons);
  }

  /** `steps` steps under `drive`: the CPU engine's `run`, on the device. */
  async run(
    steps: number,
    drive: Drive = {},
    seed = 0,
    traceNeurons: Int32Array = new Int32Array(0),
    options: LIFGpuOptions = {},
  ): Promise<Run> {
    const run = this.start(drive, seed, traceNeurons, options);
    try {
      let done = 0;
      while (done < steps) {
        const count = Math.min(run.batchSteps, steps - done);
        const encoder = this.device.createCommandEncoder({ label: "lif batch" });
        this.encode(run, encoder, count);
        this.device.queue.submit([encoder.finish()]);
        await this.collect(run);
        done += count;
      }
      return this.finish(run);
    } finally {
      run.destroy();
    }
  }

  destroy(): void {
    this.csr.indptr.destroy();
    this.csr.indices.destroy();
    this.csr.weights.destroy();
  }
}
