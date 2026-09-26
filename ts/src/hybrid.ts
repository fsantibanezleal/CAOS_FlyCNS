/**
 * The whole CNS as one system (`flycns.dynamics.hybrid`): the graded optic lobes every graded step (5 ms), the
 * published LIF for everything else every 0.1 ms, coupled through the bridge (release above grey, times beta, into
 * g) and the feedback (filtered rates over beta into the graded input). The CPU engine performs the reference's
 * operations in its order; every constant comes from the bundle.
 */

import { GradedEngine, type GradedNetwork } from "./graded.js";
import { type Drive, LIFEngine, type LIFConstants, type LIFNetwork } from "./lif.js";
import type { Run } from "./recording.js";

export interface HybridConstants {
  bridge_gain_hz: number;
  dt_graded_s: number;
  steps_per_graded: number;
  dt_lif_s: number;
  rate_decay: number;
  rate_amount: number;
}

export interface HybridNetwork {
  lif: LIFNetwork;
  graded: GradedNetwork;
  bridgeSource: Int32Array;
  bridgeTarget: Int32Array;
  bridgeWeight: Float64Array;
  feedbackSource: Int32Array;
  feedbackTarget: Int32Array;
  feedbackWeight: Float64Array;
  /** compiled neuron of each spiking unit (for the consumer; the engine never reads it) */
  spikingNeuron?: Float64Array;
}

/** E3's map: per mapped unit, the lattice nodes averaged; `weight` is float32 as the reference stores it. */
export interface LatticeMap {
  unit: Int32Array;
  side: Uint8Array;
  indptr: Float64Array;
  node: Int32Array;
  weight: Float32Array;
}

export interface HybridRun {
  spikes: Run;
  gradedUnits: Int32Array;
  /** row-major (frames, gradedUnits.length): voltages after each graded step */
  graded: Float32Array;
  greyRelease: Float32Array;
}

/** What the optic lobe's activity comes from: its own dynamics (E2, E4) or flyvis's lattices carried onto it (E3). */
export interface Source {
  grey(preSteps: number, grey: number): Float64Array[];
  step(state: Float64Array[], frame: Float64Array, feedback: Float64Array): Float64Array[];
  units(state: Float64Array[]): Float64Array;
  /** how many intensities one frame holds */
  readonly frameSize: number;
}

export class OwnSource implements Source {
  readonly frameSize: number;
  private readonly current: Float64Array;

  constructor(readonly engine: GradedEngine) {
    this.frameSize = engine.net.nColumns;
    this.current = new Float64Array(engine.n);
  }

  grey(preSteps: number, grey: number): Float64Array[] {
    return [this.engine.steadyState(preSteps, grey)];
  }

  step(state: Float64Array[], frame: Float64Array, feedback: Float64Array): Float64Array[] {
    const current = this.engine.columnCurrent(frame, this.current);
    for (let i = 0; i < current.length; i++) current[i] = (current[i] as number) + (feedback[i] as number);
    return [this.engine.step(state[0] as Float64Array, current)];
  }

  units(state: Float64Array[]): Float64Array {
    return state[0] as Float64Array;
  }
}

export class LatticeSource implements Source {
  readonly frameSize: number;
  private readonly parts: { unit: Int32Array; node: Int32Array; weight: Float32Array }[];

  constructor(
    readonly engines: GradedEngine[],
    map: LatticeMap,
    readonly nUnits: number,
  ) {
    this.frameSize = engines.reduce((sum, e) => sum + e.net.nColumns, 0);
    // the reference repeats each unit over its nodes, then splits the repeats by side, keeping their order
    const unitRep: number[] = [];
    const sideRep: number[] = [];
    for (let u = 0; u < map.unit.length; u++) {
      const count = (map.indptr[u + 1] as number) - (map.indptr[u] as number);
      for (let r = 0; r < count; r++) {
        unitRep.push(map.unit[u] as number);
        sideRep.push(map.side[u] as number);
      }
    }
    this.parts = engines.map((_, s) => {
      const keep = sideRep.map((side, r) => (side === s ? r : -1)).filter((r) => r >= 0);
      return {
        unit: Int32Array.from(keep, (r) => unitRep[r] as number),
        node: Int32Array.from(keep, (r) => map.node[r] as number),
        weight: Float32Array.from(keep, (r) => map.weight[r] as number),
      };
    });
  }

  grey(preSteps: number, grey: number): Float64Array[] {
    return this.engines.map((e) => e.steadyState(preSteps, grey));
  }

  /** Feedback from the spiking CNS does not enter flyvis's lattice. */
  step(state: Float64Array[], frame: Float64Array): Float64Array[] {
    let at = 0;
    return this.engines.map((e, s) => {
      const slice = frame.subarray(at, at + e.net.nColumns);
      at += e.net.nColumns;
      return e.step(state[s] as Float64Array, e.columnCurrent(slice));
    });
  }

  units(state: Float64Array[]): Float64Array {
    const out = new Float64Array(this.nUnits);
    this.parts.forEach((part, s) => {
      const v = state[s] as Float64Array;
      for (let r = 0; r < part.unit.length; r++) {
        const u = part.unit[r] as number;
        out[u] = (out[u] as number) + (v[part.node[r] as number] as number) * (part.weight[r] as number);
      }
    });
    return out;
  }
}

export class HybridEngine {
  readonly lif: LIFEngine;
  readonly graded: GradedEngine;
  source: Source;

  constructor(
    readonly h: HybridNetwork,
    readonly lifConstants: LIFConstants,
    readonly constants: HybridConstants,
    source?: Source,
  ) {
    this.graded = new GradedEngine(h.graded, constants.dt_graded_s);
    this.lif = new LIFEngine(h.lif, lifConstants);
    this.source = source ?? new OwnSource(this.graded);
  }

  get nSpiking(): number {
    return this.lif.n;
  }

  private bridge(deviation: Float64Array, out: Float64Array): Float64Array {
    const { bridgeSource, bridgeTarget, bridgeWeight } = this.h;
    out.fill(0);
    for (let e = 0; e < bridgeSource.length; e++) {
      const t = bridgeTarget[e] as number;
      out[t] = (out[t] as number) + (bridgeWeight[e] as number) * (deviation[bridgeSource[e] as number] as number);
    }
    const dt = this.constants.dt_lif_s;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] as number) * dt;
    return out;
  }

  private feedback(rate: Float64Array, out: Float64Array): Float64Array {
    const { feedbackSource, feedbackTarget, feedbackWeight } = this.h;
    out.fill(0);
    for (let e = 0; e < feedbackSource.length; e++) {
      const t = feedbackTarget[e] as number;
      out[t] = (out[t] as number) + (feedbackWeight[e] as number) * (rate[feedbackSource[e] as number] as number);
    }
    const gain = this.constants.bridge_gain_hz;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] as number) / gain;
    return out;
  }

  /**
   * Step through `frames` frames (row-major, frames x source.frameSize), one per graded step, from the grey steady
   * state of `preSteps` graded steps. `recordGraded` names the graded units recorded (all when omitted).
   */
  run(
    intensity: Float64Array,
    frames: number,
    options: {
      drive?: Drive;
      seed?: number;
      preSteps: number;
      grey?: number;
      recordGraded?: Int32Array;
      traceSpiking?: Int32Array;
      onFrame?: (frame: number, spikes: Int32Array[], units: Float64Array) => void;
    },
  ): HybridRun {
    const c = this.constants;
    const source = this.source;
    const grey = options.grey ?? 0.5;
    let gState = source.grey(options.preSteps, grey);
    let v = source.units(gState);
    const greyRelease = Float64Array.from(v, (x) => Math.max(x, 0.0));
    const record = options.recordGraded ?? Int32Array.from({ length: this.graded.n }, (_, i) => i);
    const state = this.lif.start(options.drive ?? {}, options.seed ?? 0, options.traceSpiking ?? new Int32Array(0));
    const nSpiking = this.nSpiking;
    let rate = new Float64Array(nSpiking);
    const fb = new Float64Array(this.graded.n);
    const gInput = new Float64Array(nSpiking);
    const deviation = new Float64Array(this.graded.n);
    const out = new Float32Array(frames * record.length);
    const size = source.frameSize;
    for (let f = 0; f < frames; f++) {
      const frame = intensity.subarray(f * size, (f + 1) * size);
      this.feedback(rate, fb);
      gState = source.step(gState, frame, fb);
      v = source.units(gState);
      for (let i = 0; i < deviation.length; i++) deviation[i] = Math.max(v[i] as number, 0.0) - (greyRelease[i] as number);
      this.bridge(deviation, gInput);
      const frameSpikes: Int32Array[] = [];
      for (let sub = 0; sub < c.steps_per_graded; sub++) {
        const spiked = this.lif.advance(state, gInput);
        const next = new Float64Array(nSpiking);
        for (let i = 0; i < nSpiking; i++) next[i] = (rate[i] as number) * c.rate_decay;
        for (let j = 0; j < spiked.length; j++) {
          next[spiked[j] as number] = (next[spiked[j] as number] as number) + c.rate_amount;
        }
        rate = next;
        if (options.onFrame) frameSpikes.push(spiked);
      }
      for (let j = 0; j < record.length; j++) out[f * record.length + j] = v[record[j] as number] as number;
      options.onFrame?.(f, frameSpikes, v);
    }
    return {
      spikes: this.lif.finish(state),
      gradedUnits: record,
      graded: out,
      greyRelease: Float32Array.from(greyRelease),
    };
  }
}
