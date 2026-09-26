/**
 * The whole CNS as one system on WebGPU: the graded optic lobe (its own dynamics, or flyvis's lattices carried onto it)
 * every graded step, the published LIF for everything else every 0.1 ms, coupled through the bridge and the feedback.
 * One graded frame is, in one command encoder: the feedback gathered from the filtered rates, the graded step, the
 * deviation from grey and the bridge gathered into the LIF's extra input, the record of the graded units, then the
 * frame's LIF steps. Several frames share a command buffer; spikes are read back once per buffer.
 */

import type { Bundle } from "../bundle.js";
import type { GradedNetwork } from "../graded.js";
import type { HybridConstants, HybridNetwork, HybridRun, LatticeMap } from "../hybrid.js";
import type { Drive, LIFConstants } from "../lif.js";
import { bindGroup, byTarget, groups, pipeline, readBuffers, storageBuffer, uniformBuffer } from "./device.js";
import { GradedGpu } from "./graded-gpu.js";
import { LIFGpu, type LIFGpuRun } from "./lif-gpu.js";
import { COUPLING_WGSL } from "./shaders.js";

const MODE_SUM = 0;
const MODE_TIMES = 1;
const MODE_OVER = 2;

interface Gather {
  params: GPUBuffer;
  ranges: GPUBuffer;
  edges: GPUBuffer;
  n: number;
}

export class HybridGpu {
  readonly lif: LIFGpu;
  /** the optic lobe's own dynamics (E2, E4); undefined when the lattices are the source (E3) */
  readonly graded?: GradedGpu;
  /** flyvis's lattices, one per eye, and the map from their nodes onto the lobe's units (E3) */
  readonly lattices: GradedGpu[] = [];
  readonly nGraded: number;
  readonly nSpiking: number;
  /** the source's units after each graded step */
  readonly units: GPUBuffer;
  private readonly frameSize: number;
  private readonly fb: GPUBuffer;
  private readonly grey: GPUBuffer;
  private readonly dev: GPUBuffer;
  private readonly gInput: GPUBuffer;
  private readonly bridge: Gather;
  private readonly feedback: Gather;
  private readonly latticeParts: Gather[] = [];
  private readonly releaseGrey: GPUBuffer;
  private readonly releaseDeviation: GPUBuffer;
  private readonly gatherPipeline: GPUComputePipeline;
  private readonly releasePipeline: GPUComputePipeline;
  private readonly owned: GPUBuffer[] = [];

  constructor(
    readonly device: GPUDevice,
    readonly h: HybridNetwork,
    lifConstants: LIFConstants,
    readonly constants: HybridConstants,
    lattice?: { networks: GradedNetwork[]; map: LatticeMap },
  ) {
    this.lif = new LIFGpu(device, h.lif, lifConstants);
    this.nSpiking = this.lif.n;
    this.nGraded = h.graded.bias.length;
    const buffer = (bytes: number, label: string) => {
      const b = storageBuffer(device, bytes, label);
      this.owned.push(b);
      return b;
    };
    this.fb = buffer(this.nGraded * 4, "hybrid feedback");
    this.grey = buffer(this.nGraded * 4, "hybrid grey release");
    this.dev = buffer(this.nGraded * 4, "hybrid deviation");
    this.gInput = buffer(this.nSpiking * 4, "hybrid bridge input");
    this.gatherPipeline = pipeline(device, COUPLING_WGSL.gather, "hybrid gather");
    this.releasePipeline = pipeline(device, COUPLING_WGSL.release, "hybrid release");
    const gather = (
      source: ArrayLike<number>,
      target: ArrayLike<number>,
      weight: ArrayLike<number>,
      n: number,
      mode: number,
      scale: number,
      accumulate: boolean,
      label: string,
    ): Gather => {
      const csr = byTarget(source, target, weight, n);
      const p = new ArrayBuffer(16);
      new Uint32Array(p, 0, 2).set([n, mode]);
      new Float32Array(p, 8, 1)[0] = scale;
      new Uint32Array(p, 12, 1)[0] = accumulate ? 1 : 0;
      const g = {
        params: uniformBuffer(device, p, `${label} params`),
        ranges: storageBuffer(device, csr.ranges, `${label} ranges`),
        edges: storageBuffer(device, new Uint32Array(csr.edges), `${label} edges`),
        n,
      };
      this.owned.push(g.params, g.ranges, g.edges);
      return g;
    };
    this.bridge = gather(h.bridgeSource, h.bridgeTarget, h.bridgeWeight, this.nSpiking, MODE_TIMES, constants.dt_lif_s, false, "bridge");
    this.feedback = gather(
      h.feedbackSource,
      h.feedbackTarget,
      h.feedbackWeight,
      this.nGraded,
      MODE_OVER,
      constants.bridge_gain_hz,
      false,
      "feedback",
    );
    const release = (mode: number) => {
      const p = new ArrayBuffer(16);
      new Uint32Array(p).set([this.nGraded, mode, 0, 0]);
      const b = uniformBuffer(device, p, "release params");
      this.owned.push(b);
      return b;
    };
    this.releaseGrey = release(0);
    this.releaseDeviation = release(1);

    if (lattice) {
      this.lattices = lattice.networks.map((net) => new GradedGpu(device, net, constants.dt_graded_s));
      this.frameSize = this.lattices.reduce((sum, e) => sum + e.nColumns, 0);
      this.units = buffer(this.nGraded * 4, "hybrid units");
      // the CPU source repeats each unit over its nodes, then splits the repeats by side, keeping their order
      const map = lattice.map;
      const unitRep: number[] = [];
      const sideRep: number[] = [];
      for (let u = 0; u < map.unit.length; u++) {
        const count = (map.indptr[u + 1] as number) - (map.indptr[u] as number);
        for (let r = 0; r < count; r++) {
          unitRep.push(map.unit[u] as number);
          sideRep.push(map.side[u] as number);
        }
      }
      this.lattices.forEach((_, s) => {
        const keep = sideRep.map((side, r) => (side === s ? r : -1)).filter((r) => r >= 0);
        this.latticeParts.push(
          gather(
            keep.map((r) => map.node[r] as number),
            keep.map((r) => unitRep[r] as number),
            keep.map((r) => map.weight[r] as number),
            this.nGraded,
            MODE_SUM,
            1,
            s > 0,
            `lattice ${s}`,
          ),
        );
      });
    } else {
      this.graded = new GradedGpu(device, h.graded, constants.dt_graded_s, this.fb);
      this.frameSize = this.graded.nColumns;
      this.units = this.graded.v;
    }
  }

  static fromBundle(device: GPUDevice, bundle: Bundle): HybridGpu {
    const { lif, hybrid, graded } = bundle.release;
    if (!lif || !hybrid || !graded) throw new Error("the bundle has no hybrid constants");
    const lattice =
      bundle.release.source === "lattice" ? { networks: bundle.latticeNetworks(), map: bundle.latticeMap() } : undefined;
    return new HybridGpu(device, bundle.hybridNetwork(), lif, hybrid, lattice);
  }

  private bindGather(g: Gather, x: GPUBuffer, out: GPUBuffer, label: string): GPUBindGroup {
    return bindGroup(this.device, this.gatherPipeline, [g.params, g.ranges, g.edges, x, out], label);
  }

  private dispatch(pass: GPUComputePassEncoder, p: GPUComputePipeline, bind: GPUBindGroup, count: number): void {
    pass.setPipeline(p);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups(count));
  }

  /** The lattices' units, gathered side by side as one running sum per unit. */
  private encodeLatticeUnits(encoder: GPUCommandEncoder, binds: GPUBindGroup[]): void {
    const pass = encoder.beginComputePass({ label: "lattice units" });
    binds.forEach((bind) => this.dispatch(pass, this.gatherPipeline, bind, this.nGraded));
    pass.end();
  }

  /**
   * Step through `frames` frames (row-major, frames x frameSize), one per graded step, from the grey steady state of
   * `preSteps` graded steps: the CPU engine's `run`, on the device.
   */
  async run(
    intensity: Float64Array,
    frames: number,
    options: {
      drive?: Drive;
      seed?: number;
      preSteps: number;
      grey?: number;
      recordGraded?: Int32Array;
      traceSpiking?: Int32Array;
      /** graded frames per command buffer (default: as many as fit in 1,000 LIF steps) */
      framesPerBatch?: number;
    },
  ): Promise<HybridRun> {
    const { device } = this;
    const c = this.constants;
    const grey = options.grey ?? 0.5;
    const record = options.recordGraded ?? Int32Array.from({ length: this.nGraded }, (_, i) => i);
    const latticeBinds = this.latticeParts.map((part, s) =>
      this.bindGather(part, (this.lattices[s] as GradedGpu).v, this.units, `lattice ${s} units`),
    );

    // the grey steady state and its release
    if (this.graded) {
      await this.graded.steadyState(options.preSteps, grey);
    } else {
      for (const e of this.lattices) await e.steadyState(options.preSteps, grey);
    }
    {
      const encoder = device.createCommandEncoder({ label: "hybrid grey" });
      if (!this.graded) this.encodeLatticeUnits(encoder, latticeBinds);
      const pass = encoder.beginComputePass({ label: "grey release" });
      const bind = bindGroup(device, this.releasePipeline, [this.releaseGrey, this.units, this.dev, this.grey], "grey release");
      this.dispatch(pass, this.releasePipeline, bind, this.nGraded);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [greyBytes] = await readBuffers(device, [{ buffer: this.grey, byteLength: this.nGraded * 4 }]);

    const framesPerBatch = Math.max(1, options.framesPerBatch ?? Math.floor(1000 / c.steps_per_graded));
    const run: LIFGpuRun = this.lif.start(options.drive ?? {}, options.seed ?? 0, options.traceSpiking ?? new Int32Array(0), {
      batchSteps: framesPerBatch * c.steps_per_graded,
      gInput: this.gInput,
      rate: { decay: c.rate_decay, amount: c.rate_amount },
    });
    const framesBuffer = storageBuffer(device, Float32Array.from(intensity), "hybrid frames");
    const recordBuffer = storageBuffer(device, new Uint32Array(record), "hybrid record list");
    const recorded = storageBuffer(device, record.length * 4, "hybrid recorded");
    const gradedOut = storageBuffer(device, frames * record.length * 4, "hybrid graded");
    const temporary = [framesBuffer, recordBuffer, recorded, gradedOut];
    try {
      const feedbackBind = this.bindGather(this.feedback, run.rate, this.fb, "feedback");
      const deviationBind = bindGroup(
        device,
        this.releasePipeline,
        [this.releaseDeviation, this.units, this.grey, this.dev],
        "deviation",
      );
      const bridgeBind = this.bindGather(this.bridge, this.dev, this.gInput, "bridge");
      const recordEngine = this.graded ?? (this.lattices[0] as GradedGpu);
      const recordBind = recordEngine.recorder(recordBuffer, this.units, recorded);
      for (let done = 0; done < frames; done += framesPerBatch) {
        const encoder = device.createCommandEncoder({ label: "hybrid frames" });
        for (let f = done; f < Math.min(frames, done + framesPerBatch); f++) {
          const frameAt = f * this.frameSize * 4;
          if (this.graded) {
            const pass = encoder.beginComputePass({ label: "feedback" });
            this.dispatch(pass, this.gatherPipeline, feedbackBind, this.nGraded);
            pass.end();
            if (this.frameSize) encoder.copyBufferToBuffer(framesBuffer, frameAt, this.graded.intensity, 0, this.frameSize * 4);
            this.graded.encodeStep(encoder, true);
          } else {
            let at = 0;
            for (const e of this.lattices) {
              if (e.nColumns) encoder.copyBufferToBuffer(framesBuffer, frameAt + at * 4, e.intensity, 0, e.nColumns * 4);
              at += e.nColumns;
              e.encodeStep(encoder);
            }
            this.encodeLatticeUnits(encoder, latticeBinds);
          }
          const pass = encoder.beginComputePass({ label: "bridge" });
          this.dispatch(pass, this.releasePipeline, deviationBind, this.nGraded);
          this.dispatch(pass, this.gatherPipeline, bridgeBind, this.nSpiking);
          pass.end();
          recordEngine.encodeRecord(encoder, recordBind, record.length);
          encoder.copyBufferToBuffer(recorded, 0, gradedOut, f * record.length * 4, record.length * 4);
          this.lif.encode(run, encoder, c.steps_per_graded);
        }
        device.queue.submit([encoder.finish()]);
        await this.lif.collect(run);
      }
      const [gradedBytes] = await readBuffers(device, [{ buffer: gradedOut, byteLength: frames * record.length * 4 }]);
      return {
        spikes: this.lif.finish(run),
        gradedUnits: record,
        graded: new Float32Array(gradedBytes as ArrayBuffer),
        greyRelease: new Float32Array(greyBytes as ArrayBuffer),
      };
    } finally {
      run.destroy();
      for (const b of temporary) b.destroy();
    }
  }

  destroy(): void {
    this.lif.destroy();
    this.graded?.destroy();
    for (const e of this.lattices) e.destroy();
    for (const b of this.owned) if (b !== this.graded?.v) b.destroy();
  }
}
