/**
 * The published whole-brain leaky integrate-and-fire model (Shiu et al. 2024) on the CPU, in float64, performing the
 * NumPy reference's operations in the reference's order (`flycns.dynamics.lif.LIFReference`), so that the same
 * bundle, drive and seed give the same spikes and traces bit for bit. See `docs/design/features/browser/design.md`
 * for the rules that make this hold; every constant of the step comes from the bundle, never from `Math.exp`.
 */

import { hash3, eventThreshold } from "./rng.js";
import type { Run } from "./recording.js";

/** The published constants and what the step derives from them (`flycns.bundle.lif_constants`). */
export interface LIFConstants {
  v0_mv: number;
  v_rst_mv: number;
  v_th_mv: number;
  t_mbr_ms: number;
  tau_ms: number;
  t_rfc_ms: number;
  t_dly_ms: number;
  w_syn_mv: number;
  f_poi: number;
  dt_ms: number;
  adaptation_mv: number;
  tau_adaptation_ms: number;
  delay_steps: number;
  refractory_steps: number;
  decay_a: number;
  decay_b: number;
  decay_c: number;
  adaptation_decay: number;
  w_event_mv: number;
}

/** Poisson input whose rate changes over time: `rates` is row-major (frames, neurons.length), in Hz. */
export interface ModulatedDrive {
  neurons: Int32Array;
  rates: Float64Array;
  frames: number;
  stepsPerFrame: number;
}

/** What drives a run (`flycns.dynamics.lif.Drive`). Events are (neurons, dv mV) per step, summed in order. */
export interface Drive {
  activate?: Map<number, number>;
  events?: Map<number, { neurons: ArrayLike<number>; dvMv: ArrayLike<number> }>;
  silenced?: ArrayLike<number>;
  modulated?: ModulatedDrive;
}

export interface LIFNetwork {
  /** length n + 1 */
  indptr: Float64Array;
  indices: Int32Array;
  weightsMv: Float64Array;
}

export class LIFState {
  k = 0;
  readonly v: Float64Array;
  readonly g: Float64Array;
  readonly last: Float64Array;
  readonly refractory: Int32Array;
  readonly mute: Uint8Array;
  readonly adaptation: Float64Array;
  readonly free: Uint8Array;
  readonly delivery: Float64Array;
  readonly act: Int32Array;
  readonly thresholds: Float64Array;
  readonly modNeurons: Int32Array;
  readonly modThresholds: Float64Array;
  readonly modFrames: number;
  readonly modSteps: number;
  readonly events: Map<number, { neurons: ArrayLike<number>; dvMv: ArrayLike<number> }>;
  readonly pending: Int32Array[] = [];
  readonly traceNeurons: Int32Array;
  readonly spikes: Int32Array[] = [];
  readonly traces: Float32Array[] = [];
  readonly seed: number;

  constructor(n: number, c: LIFConstants, drive: Drive, seed: number, traceNeurons: Int32Array) {
    this.seed = seed >>> 0;
    this.v = new Float64Array(n).fill(c.v0_mv);
    this.g = new Float64Array(n);
    this.last = new Float64Array(n).fill(-1e9);
    this.refractory = new Int32Array(n).fill(c.refractory_steps);
    this.mute = new Uint8Array(n);
    this.adaptation = new Float64Array(n);
    this.free = new Uint8Array(n);
    this.delivery = new Float64Array(n);
    const activate = drive.activate ?? new Map<number, number>();
    this.act = Int32Array.from([...activate.keys()].sort((a, b) => a - b));
    const dtS = c.dt_ms / 1000.0; // the reference: event_threshold(rate, dt_ms / 1000.0)
    this.thresholds = Float64Array.from(this.act, (i) => eventThreshold(activate.get(i) as number, dtS));
    for (const i of this.act) this.refractory[i] = 0; // activated neurons have no refractory period
    const modulated = drive.modulated;
    if (modulated) {
      this.modNeurons = modulated.neurons;
      this.modFrames = modulated.frames;
      this.modSteps = modulated.stepsPerFrame;
      // the reference: clip(rates * dt_ms / 1000, 0, 1), which is (rate * dt_ms) / 1000 in that order
      this.modThresholds = Float64Array.from(modulated.rates, (r) => {
        const p = Math.min(Math.max((r * c.dt_ms) / 1000.0, 0), 1);
        return Math.min(Math.floor(p * 4294967296), 4294967295);
      });
      for (const i of this.modNeurons) this.refractory[i] = 0;
    } else {
      this.modNeurons = new Int32Array(0);
      this.modFrames = 1;
      this.modSteps = 1;
      this.modThresholds = new Float64Array(0);
    }
    if (drive.silenced) for (let j = 0; j < drive.silenced.length; j++) this.mute[drive.silenced[j] as number] = 1;
    this.events = drive.events ?? new Map();
    for (let d = 0; d < c.delay_steps; d++) this.pending.push(new Int32Array(0));
    this.traceNeurons = traceNeurons;
  }
}

export class LIFEngine {
  readonly n: number;

  constructor(
    readonly network: LIFNetwork,
    readonly constants: LIFConstants,
  ) {
    this.n = network.indptr.length - 1;
  }

  start(drive: Drive = {}, seed = 0, traceNeurons: Int32Array = new Int32Array(0)): LIFState {
    return new LIFState(this.n, this.constants, drive, seed, traceNeurons);
  }

  /** One step; returns the neurons that spiked, ascending. `gInput` is an extra input to g (the hybrid's bridge). */
  advance(s: LIFState, gInput?: Float64Array): Int32Array {
    const c = this.constants;
    const { v, g, last, refractory, free, delivery, adaptation } = s;
    const { indptr, indices, weightsMv } = this.network;
    const n = this.n;
    const k = s.k;
    const a = c.decay_a;
    const b = c.decay_b;
    const cc = c.decay_c;
    const adapting = c.adaptation_mv !== 0;
    // 1. groups: exact integration of v and g for neurons that are not refractory
    // 2. thresholds: a neuron that spikes is refractory from this step on
    const spiked: number[] = [];
    for (let i = 0; i < n; i++) {
      const isFree = k - (last[i] as number) >= (refractory[i] as number);
      if (isFree) {
        const rest = adapting ? c.v0_mv - (adaptation[i] as number) : c.v0_mv;
        v[i] = rest + ((v[i] as number) - rest) * a + (g[i] as number) * cc;
        g[i] = (g[i] as number) * b;
        if ((v[i] as number) > c.v_th_mv) {
          last[i] = k;
          free[i] = 0;
          spiked.push(i);
          continue;
        }
      }
      free[i] = isFree ? 1 : 0;
    }
    const spike = Int32Array.from(spiked);
    // 3. synapses: spikes emitted delay_steps ago add their weights to g, accumulated in the reference's order
    // (spiking neurons ascending, each row in CSR order) and then added to the neurons that are not refractory;
    // a silenced neuron's spikes reach no one
    const reaching = spike.filter((i) => s.mute[i] === 0);
    s.pending.push(reaching);
    const due = s.pending.shift() as Int32Array;
    if (due.length) {
      delivery.fill(0);
      for (let j = 0; j < due.length; j++) {
        const source = due[j] as number;
        const end = indptr[source + 1] as number;
        for (let e = indptr[source] as number; e < end; e++) {
          const t = indices[e] as number;
          delivery[t] = (delivery[t] as number) + (weightsMv[e] as number);
        }
      }
      for (let i = 0; i < n; i++) if (free[i]) g[i] = (g[i] as number) + (delivery[i] as number);
    }
    if (gInput) for (let i = 0; i < n; i++) if (free[i]) g[i] = (g[i] as number) + (gInput[i] as number);
    // Poisson and fixed events add to v, only for neurons that are not refractory
    const w = c.w_event_mv;
    for (let j = 0; j < s.act.length; j++) {
      const i = s.act[j] as number;
      if (hash3(s.seed, i, k) < (s.thresholds[j] as number) && free[i]) v[i] = (v[i] as number) + w;
    }
    if (s.modNeurons.length) {
      const frame = Math.min(Math.floor(k / s.modSteps), s.modFrames - 1);
      const m = s.modNeurons.length;
      for (let j = 0; j < m; j++) {
        const i = s.modNeurons[j] as number;
        if (hash3(s.seed, i, k) < (s.modThresholds[frame * m + j] as number) && free[i]) {
          v[i] = (v[i] as number) + w;
        }
      }
    }
    const events = s.events.get(k);
    if (events) {
      for (let j = 0; j < events.neurons.length; j++) {
        const i = events.neurons[j] as number;
        if (free[i]) v[i] = (v[i] as number) + (events.dvMv[j] as number);
      }
    }
    // 4. resets
    for (let j = 0; j < spike.length; j++) {
      v[spike[j] as number] = c.v_rst_mv;
      g[spike[j] as number] = 0;
    }
    if (adapting) {
      for (let i = 0; i < n; i++) adaptation[i] = (adaptation[i] as number) * c.adaptation_decay;
      for (let j = 0; j < spike.length; j++) {
        adaptation[spike[j] as number] = (adaptation[spike[j] as number] as number) + c.adaptation_mv;
      }
    }
    s.spikes.push(spike);
    if (s.traceNeurons.length) {
      s.traces.push(Float32Array.from(s.traceNeurons, (i) => v[i] as number));
    }
    s.k = k + 1;
    return spike;
  }

  finish(s: LIFState): Run {
    const tickIndptr = new Float64Array(s.k + 1);
    let total = 0;
    for (let k = 0; k < s.k; k++) {
      tickIndptr[k] = total;
      total += (s.spikes[k] as Int32Array).length;
    }
    tickIndptr[s.k] = total;
    const neuronIndex = new Int32Array(total);
    let at = 0;
    for (const spike of s.spikes) {
      neuronIndex.set(spike, at);
      at += spike.length;
    }
    const nTrace = s.traceNeurons.length;
    const tracesMv = new Float32Array(s.k * nTrace);
    for (let k = 0; k < s.traces.length; k++) tracesMv.set(s.traces[k] as Float32Array, k * nTrace);
    return {
      nNeurons: this.n,
      steps: s.k,
      dtMs: this.constants.dt_ms,
      tickIndptr,
      neuronIndex,
      traceNeurons: s.traceNeurons,
      tracesMv,
    };
  }

  run(steps: number, drive: Drive = {}, seed = 0, traceNeurons: Int32Array = new Int32Array(0)): Run {
    const s = this.start(drive, seed, traceNeurons);
    for (let k = 0; k < steps; k++) this.advance(s);
    return this.finish(s);
  }
}
