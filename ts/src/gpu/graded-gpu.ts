/**
 * The graded optic lobe on WebGPU, in float32. Each neuron gathers its incoming connections in their original order
 * (a CSR by target built once on the CPU), which is the order the CPU engine sums them in, so the step needs no
 * atomics and no bound on the activity; the result is compared with the reference by tolerance.
 */

import type { Bundle } from "../bundle.js";
import type { GradedNetwork } from "../graded.js";
import { bindGroup, byTarget, groups, pipeline, readBuffers, storageBuffer, uniformBuffer } from "./device.js";
import { GRADED_WGSL } from "./shaders.js";

const NODE_WORDS = 6;

/** Per neuron, its input columns in their original order: [start, end) into `columns`. */
function columnsByNeuron(inputNeuron: Int32Array, inputColumn: Int32Array, n: number) {
  const start = new Uint32Array(n + 1);
  for (let j = 0; j < inputNeuron.length; j++) {
    const i = inputNeuron[j] as number;
    start[i + 1] = (start[i + 1] as number) + 1;
  }
  for (let i = 0; i < n; i++) start[i + 1] = (start[i + 1] as number) + (start[i] as number);
  const next = start.slice(0, n);
  const columns = new Uint32Array(inputNeuron.length);
  for (let j = 0; j < inputNeuron.length; j++) {
    const i = inputNeuron[j] as number;
    columns[next[i] as number] = inputColumn[j] as number;
    next[i] = (next[i] as number) + 1;
  }
  return { start, columns };
}

export class GradedGpu {
  readonly n: number;
  readonly nColumns: number;
  /** the current state, one float per neuron */
  readonly v: GPUBuffer;
  /** this step's light, one float per column */
  readonly intensity: GPUBuffer;
  private readonly next: GPUBuffer;
  private readonly buffers: GPUBuffer[];
  private readonly step: GPUComputePipeline;
  private readonly record: GPUComputePipeline;
  private readonly plain: GPUBindGroup;
  private readonly withFeedback?: GPUBindGroup;

  /** `feedback`, when given, is added to every neuron's input on the steps that ask for it (the hybrid's feedback). */
  constructor(
    readonly device: GPUDevice,
    readonly net: GradedNetwork,
    readonly dt: number,
    feedback?: GPUBuffer,
  ) {
    const n = net.bias.length;
    this.n = n;
    this.nColumns = net.nColumns;
    const incoming = byTarget(net.source, net.target, net.weight, n);
    const cols = columnsByNeuron(net.inputNeuron, net.inputColumn, n);
    const nodes = new ArrayBuffer(n * NODE_WORDS * 4);
    const nf = new Float32Array(nodes);
    const nu = new Uint32Array(nodes);
    for (let i = 0; i < n; i++) {
      nf[NODE_WORDS * i] = net.bias[i] as number;
      nf[NODE_WORDS * i + 1] = 1.0 / Math.max(net.timeConstS[i] as number, dt);
      nu[NODE_WORDS * i + 2] = incoming.ranges[2 * i] as number;
      nu[NODE_WORDS * i + 3] = incoming.ranges[2 * i + 1] as number;
      nu[NODE_WORDS * i + 4] = cols.start[i] as number;
      nu[NODE_WORDS * i + 5] = cols.start[i + 1] as number;
    }
    const params = (hasFeedback: number) => {
      const p = new ArrayBuffer(16);
      new Uint32Array(p, 0, 2).set([n, hasFeedback]);
      new Float32Array(p, 8, 1)[0] = dt;
      return uniformBuffer(device, p, "graded params");
    };
    const nodeBuffer = storageBuffer(device, nu, "graded nodes");
    const edgeBuffer = storageBuffer(device, new Uint32Array(incoming.edges), "graded edges");
    const colBuffer = storageBuffer(device, cols.columns, "graded columns");
    this.intensity = storageBuffer(device, net.nColumns * 4, "graded intensity");
    this.v = storageBuffer(device, Float32Array.from(net.bias), "graded state");
    this.next = storageBuffer(device, n * 4, "graded next state");
    const noFeedback = storageBuffer(device, 4, "graded no feedback");
    const plainParams = params(0);
    const feedbackParams = params(1);
    this.buffers = [nodeBuffer, edgeBuffer, colBuffer, this.intensity, this.v, this.next, noFeedback, plainParams, feedbackParams];
    this.step = pipeline(device, GRADED_WGSL.step, "graded step");
    this.record = pipeline(device, GRADED_WGSL.record, "graded record");
    const common = [nodeBuffer, edgeBuffer, colBuffer, this.intensity];
    this.plain = bindGroup(device, this.step, [plainParams, ...common, noFeedback, this.v, this.next], "graded step");
    if (feedback) {
      this.withFeedback = bindGroup(
        device,
        this.step,
        [feedbackParams, ...common, feedback, this.v, this.next],
        "graded step with feedback",
      );
    }
  }

  static fromBundle(device: GPUDevice, bundle: Bundle): GradedGpu {
    const graded = bundle.release.graded;
    if (!graded) throw new Error("the bundle has no graded constants");
    return new GradedGpu(device, bundle.gradedNetwork(), graded.dt_s);
  }

  /** Encode one step: the new state is computed into a second buffer, then copied over the current one. */
  encodeStep(encoder: GPUCommandEncoder, feedback = false): void {
    const bind = feedback ? this.withFeedback : this.plain;
    if (!bind) throw new Error("this graded engine was built without a feedback buffer");
    const pass = encoder.beginComputePass({ label: "graded step" });
    pass.setPipeline(this.step);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups(this.n));
    pass.end();
    encoder.copyBufferToBuffer(this.next, 0, this.v, 0, this.n * 4);
  }

  /** A bind group that records the state of the neurons in `record` into `out`. */
  recorder(record: GPUBuffer, values: GPUBuffer, out: GPUBuffer): GPUBindGroup {
    return bindGroup(this.device, this.record, [record, values, out], "graded record");
  }

  encodeRecord(encoder: GPUCommandEncoder, bind: GPUBindGroup, count: number): void {
    const pass = encoder.beginComputePass({ label: "graded record" });
    pass.setPipeline(this.record);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups(count));
    pass.end();
  }

  setState(values: ArrayLike<number>): void {
    this.device.queue.writeBuffer(this.v, 0, Float32Array.from(values));
  }

  setIntensity(values: ArrayLike<number>): void {
    this.device.queue.writeBuffer(this.intensity, 0, Float32Array.from(values));
  }

  async read(): Promise<Float32Array> {
    const [bytes] = await readBuffers(this.device, [{ buffer: this.v, byteLength: this.n * 4 }]);
    return new Float32Array(bytes as ArrayBuffer);
  }

  /** `preSteps` steps of uniform intensity `grey` from the resting potentials (or `initial`); the state stays here. */
  async steadyState(preSteps: number, grey = 0.5, initial?: ArrayLike<number>): Promise<void> {
    this.setState(initial ?? this.net.bias);
    this.setIntensity(new Float32Array(this.nColumns).fill(grey));
    const chunk = 256;
    for (let done = 0; done < preSteps; done += chunk) {
      const encoder = this.device.createCommandEncoder({ label: "graded lead-in" });
      for (let k = done; k < Math.min(preSteps, done + chunk); k++) this.encodeStep(encoder);
      this.device.queue.submit([encoder.finish()]);
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Step through `frames` frames of per-column intensity (row-major, frames x nColumns): the CPU engine's `run`.
   * Returns the final state and the state after each step of the neurons in `record` (all when omitted).
   */
  async run(
    intensity: Float64Array,
    frames: number,
    initial?: ArrayLike<number>,
    recordNeurons?: Int32Array,
  ): Promise<{ final: Float32Array; activity: Float32Array }> {
    const { device, n } = this;
    const record = recordNeurons ?? Int32Array.from({ length: n }, (_, i) => i);
    this.setState(initial ?? this.net.bias);
    const framesBuffer = storageBuffer(device, Float32Array.from(intensity), "graded frames");
    const recordBuffer = storageBuffer(device, new Uint32Array(record), "graded record list");
    const recorded = storageBuffer(device, record.length * 4, "graded recorded");
    const activity = storageBuffer(device, frames * record.length * 4, "graded activity");
    const bind = this.recorder(recordBuffer, this.v, recorded);
    const frameBytes = this.nColumns * 4;
    const chunk = 256;
    for (let done = 0; done < frames; done += chunk) {
      const encoder = device.createCommandEncoder({ label: "graded frames" });
      for (let f = done; f < Math.min(frames, done + chunk); f++) {
        if (frameBytes) encoder.copyBufferToBuffer(framesBuffer, f * frameBytes, this.intensity, 0, frameBytes);
        this.encodeStep(encoder);
        this.encodeRecord(encoder, bind, record.length);
        encoder.copyBufferToBuffer(recorded, 0, activity, f * record.length * 4, record.length * 4);
      }
      device.queue.submit([encoder.finish()]);
    }
    const [finalBytes, activityBytes] = await readBuffers(device, [
      { buffer: this.v, byteLength: n * 4 },
      { buffer: activity, byteLength: frames * record.length * 4 },
    ]);
    for (const b of [framesBuffer, recordBuffer, recorded, activity]) b.destroy();
    return { final: new Float32Array(finalBytes as ArrayBuffer), activity: new Float32Array(activityBytes as ArrayBuffer) };
  }

  destroy(): void {
    for (const b of this.buffers) b.destroy();
  }
}
