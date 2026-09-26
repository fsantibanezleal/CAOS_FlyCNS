/**
 * Graded (non-spiking) point neurons as flyvis computes them (`flycns.dynamics.graded.GradedReference`), in float64,
 * with the reference's operation order: each neuron's synaptic input summed in connection order, then
 * `v + rate * (((-v + bias) + syn) + current) * dt` with `rate = 1 / max(tau, dt)`.
 */

export interface GradedNetwork {
  bias: Float64Array;
  timeConstS: Float64Array;
  source: Int32Array;
  target: Int32Array;
  weight: Float64Array;
  /** the neurons that receive light, and the column each of them sees */
  inputNeuron: Int32Array;
  inputColumn: Int32Array;
  nColumns: number;
}

export class GradedEngine {
  readonly n: number;
  readonly rate: Float64Array;
  private readonly synaptic: Float64Array;

  constructor(
    readonly net: GradedNetwork,
    readonly dt: number,
  ) {
    this.n = net.bias.length;
    this.rate = Float64Array.from(net.timeConstS, (tau) => 1.0 / Math.max(tau, dt));
    this.synaptic = new Float64Array(this.n);
  }

  /** Per-neuron input for one frame of per-column intensities (zero off the input neurons), summed in order. */
  columnCurrent(intensity: ArrayLike<number>, out?: Float64Array): Float64Array {
    const current = out ?? new Float64Array(this.n);
    current.fill(0);
    const { inputNeuron, inputColumn } = this.net;
    for (let j = 0; j < inputNeuron.length; j++) {
      const i = inputNeuron[j] as number;
      current[i] = (current[i] as number) + (intensity[inputColumn[j] as number] as number);
    }
    return current;
  }

  /** One Euler step with a per-neuron input `current`; returns a new state. */
  step(v: Float64Array, current: Float64Array): Float64Array {
    const { source, target, weight, bias } = this.net;
    const syn = this.synaptic;
    syn.fill(0);
    for (let e = 0; e < source.length; e++) {
      const release = Math.max(v[source[e] as number] as number, 0.0);
      const t = target[e] as number;
      syn[t] = (syn[t] as number) + (weight[e] as number) * release;
    }
    const out = new Float64Array(this.n);
    const dt = this.dt;
    for (let i = 0; i < this.n; i++) {
      const velocity =
        (this.rate[i] as number) * (-(v[i] as number) + (bias[i] as number) + (syn[i] as number) + (current[i] as number));
      out[i] = (v[i] as number) + velocity * dt;
    }
    return out;
  }

  /**
   * The state after `preSteps` steps of uniform intensity `grey`, from the resting potentials unless given. The
   * reference takes `int(t_pre_s / dt)` steps; a bundle's scenario carries that count as `pre_steps`.
   */
  steadyState(preSteps: number, grey = 0.5, initial?: Float64Array): Float64Array {
    let v = initial ? Float64Array.from(initial) : Float64Array.from(this.net.bias);
    const current = this.columnCurrent(new Float64Array(this.net.nColumns).fill(grey));
    for (let k = 0; k < preSteps; k++) v = this.step(v, current);
    return v;
  }

  /**
   * Step through `frames` frames of per-column intensity (row-major, frames x nColumns); returns the final state and
   * the activity after each step for the neurons in `record` (all when omitted), as float32, row-major.
   */
  run(
    intensity: Float64Array,
    frames: number,
    initial?: Float64Array,
    record?: Int32Array,
  ): { final: Float64Array; activity: Float32Array } {
    let v = initial ? Float64Array.from(initial) : Float64Array.from(this.net.bias);
    const rec = record ?? Int32Array.from({ length: this.n }, (_, i) => i);
    const activity = new Float32Array(frames * rec.length);
    const current = new Float64Array(this.n);
    const columns = this.net.nColumns;
    for (let k = 0; k < frames; k++) {
      v = this.step(v, this.columnCurrent(intensity.subarray(k * columns, (k + 1) * columns), current));
      for (let j = 0; j < rec.length; j++) activity[k * rec.length + j] = v[rec[j] as number] as number;
    }
    return { final: v, activity };
  }
}
