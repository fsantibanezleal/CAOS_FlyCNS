/**
 * The WGSL kernels of the GPU engines. Each kernel is its own module with bindings numbered from 0 in the order the
 * engine passes its buffers, so a pipeline's inferred layout holds exactly those bindings. No kernel binds more than
 * seven storage buffers, under the default limit of eight per stage. The step they implement, kernel by kernel, is
 * the CPU engines' step in the same order (`docs/design/features/browser/design.md`).
 */

/** MurmurHash3_x86_32 of (neuron, step) seeded with the run's seed, in u32 arithmetic: the bits of `hash3`. */
export const HASH_WGSL = /* wgsl */ `
fn mm_rotl(x: u32, r: u32) -> u32 {
  return (x << r) | (x >> (32u - r));
}

fn mm_block(h0: u32, k0: u32) -> u32 {
  var k = k0 * 0xcc9e2d51u;
  k = mm_rotl(k, 15u);
  k = k * 0x1b873593u;
  var h = h0 ^ k;
  h = mm_rotl(h, 13u);
  return h * 5u + 0xe6546b64u;
}

fn mm_fmix(h0: u32) -> u32 {
  var h = h0;
  h = h ^ (h >> 16u);
  h = h * 0x85ebca6bu;
  h = h ^ (h >> 13u);
  h = h * 0xc2b2ae35u;
  return h ^ (h >> 16u);
}

fn hash3(seed: u32, neuron: u32, step: u32) -> u32 {
  var h = mm_block(seed, neuron);
  h = mm_block(h, step);
  return mm_fmix(h ^ 8u);
}
`;

/** The hash over a list of keys (seed, neuron, step), three u32 per key: the check of the WGSL hash. */
export const HASH_KERNEL_WGSL = /* wgsl */ `
${HASH_WGSL}
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&out)) {
    return;
  }
  out[i] = hash3(keys[3u * i], keys[3u * i + 1u], keys[3u * i + 2u]);
}
`;

/**
 * The LIF step's shared declarations. Params mirrors the uniform `LIFGpu` writes (24 words); Clock is the step
 * counter the tick kernel advances, so a batch of steps is one command buffer; Neuron packs the per-neuron state.
 */
const LIF_PRELUDE = /* wgsl */ `
struct Params {
  n: u32,
  delay: u32,
  slots: u32,
  seed: u32,
  v0: f32,
  vrst: f32,
  vth: f32,
  a: f32,
  b: f32,
  c: f32,
  wEvent: f32,
  adaptMv: f32,
  adaptDecay: f32,
  scaleInv: f32,
  rateDecay: f32,
  rateAmount: f32,
  nMod: u32,
  modSteps: u32,
  modFrames: u32,
  nTrace: u32,
  flags: u32,
  spikeCap: u32,
  pad0: u32,
  pad1: u32,
}

struct Clock {
  k: u32,
  kb: u32,
}

struct Neuron {
  v: f32,
  g: f32,
  adapt: f32,
  last: i32,
  refr: i32,
  flags: u32,
}

const MUTE: u32 = 1u;
const FREE: u32 = 2u;
const SPIKED: u32 = 4u;
const ACT: u32 = 8u;

const ADAPTING: u32 = 1u;
const HAS_GINPUT: u32 = 2u;
const HAS_RATE: u32 = 4u;
const HAS_MOD: u32 = 8u;
`;

/** The LIF step as seven kernels, dispatched in this order once per step. */
export const LIF_WGSL = {
  /** Exact integration of v and g for free neurons, the threshold, and the spike flag in the delay ring. */
  integrate: /* wgsl */ `
${LIF_PRELUDE}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(3) var<storage, read_write> ring: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.n) {
    return;
  }
  let k = clock.k;
  var nr = neurons[i];
  var flags = nr.flags & ~(FREE | SPIKED);
  var reaching = 0u;
  if (i32(k) - nr.last >= nr.refr) {
    var rest = P.v0;
    if ((P.flags & ADAPTING) != 0u) {
      rest = P.v0 - nr.adapt;
    }
    nr.v = rest + (nr.v - rest) * P.a + nr.g * P.c;
    nr.g = nr.g * P.b;
    if (nr.v > P.vth) {
      nr.last = i32(k);
      flags = flags | SPIKED;
      if ((flags & MUTE) == 0u) {
        reaching = 1u;
      }
    } else {
      flags = flags | FREE;
    }
  }
  nr.flags = flags;
  neurons[i] = nr;
  ring[(k % P.slots) * P.n + i] = reaching;
}
`,

  /** The rows of the neurons that spiked delay_steps ago, pushed into int32 fixed-point accumulators. */
  deliver: /* wgsl */ `
${LIF_PRELUDE}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read> ring: array<u32>;
@group(0) @binding(3) var<storage, read> indptr: array<u32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> weights: array<i32>;
@group(0) @binding(6) var<storage, read_write> acc: array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x;
  if (j >= P.n) {
    return;
  }
  let k = clock.k;
  if (k < P.delay) {
    return;
  }
  if (ring[((k - P.delay) % P.slots) * P.n + j] == 0u) {
    return;
  }
  let end = indptr[j + 1u];
  for (var e = indptr[j]; e < end; e++) {
    atomicAdd(&acc[indices[e]], weights[e]);
  }
}
`,

  /** The delivery and the extra input into g, then Poisson and modulated events into v, for free neurons only. */
  apply: /* wgsl */ `
${LIF_PRELUDE}
${HASH_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> gInput: array<f32>;
@group(0) @binding(5) var<storage, read> actThr: array<u32>;
@group(0) @binding(6) var<storage, read> modSlot: array<i32>;
@group(0) @binding(7) var<storage, read> modThr: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.n) {
    return;
  }
  let k = clock.k;
  let delivered = atomicExchange(&acc[i], 0);
  var nr = neurons[i];
  if ((nr.flags & FREE) == 0u) {
    return;
  }
  nr.g = nr.g + f32(delivered) * P.scaleInv;
  if ((P.flags & HAS_GINPUT) != 0u) {
    nr.g = nr.g + gInput[i];
  }
  if ((nr.flags & ACT) != 0u) {
    if (hash3(P.seed, i, k) < actThr[i]) {
      nr.v = nr.v + P.wEvent;
    }
  }
  if ((P.flags & HAS_MOD) != 0u) {
    let m = modSlot[i];
    if (m >= 0) {
      let frame = min(k / P.modSteps, P.modFrames - 1u);
      if (hash3(P.seed, i, k) < modThr[frame * P.nMod + u32(m)]) {
        nr.v = nr.v + P.wEvent;
      }
    }
  }
  neurons[i] = nr;
}
`,

  /** This step's fixed events, in order, by one invocation (they are rare; two may hit one neuron). */
  events: /* wgsl */ `
${LIF_PRELUDE}
struct Event {
  neuron: u32,
  dv: f32,
}

@group(0) @binding(0) var<storage, read> clock: Clock;
@group(0) @binding(1) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(2) var<storage, read> eventIndptr: array<u32>;
@group(0) @binding(3) var<storage, read> events: array<Event>;

@compute @workgroup_size(1)
fn main() {
  let k = clock.k;
  if (k + 1u >= arrayLength(&eventIndptr)) {
    return;
  }
  let end = eventIndptr[k + 1u];
  for (var e = eventIndptr[k]; e < end; e++) {
    let ev = events[e];
    var nr = neurons[ev.neuron];
    if ((nr.flags & FREE) != 0u) {
      nr.v = nr.v + ev.dv;
      neurons[ev.neuron] = nr;
    }
  }
}
`,

  /** Resets, adaptation, the spike appended to the batch's list, and the hybrid's filtered rate. */
  reset: /* wgsl */ `
${LIF_PRELUDE}
struct Counter {
  n: atomic<u32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(3) var<storage, read_write> spikes: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> counter: Counter;
@group(0) @binding(5) var<storage, read_write> rate: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.n) {
    return;
  }
  var nr = neurons[i];
  let spiked = (nr.flags & SPIKED) != 0u;
  if (spiked) {
    nr.v = P.vrst;
    nr.g = 0.0;
  }
  if ((P.flags & ADAPTING) != 0u) {
    nr.adapt = nr.adapt * P.adaptDecay;
    if (spiked) {
      nr.adapt = nr.adapt + P.adaptMv;
    }
  }
  neurons[i] = nr;
  if (spiked) {
    let at = atomicAdd(&counter.n, 1u);
    if (at < P.spikeCap) {
      spikes[at] = vec2<u32>(clock.kb, i);
    }
  }
  if ((P.flags & HAS_RATE) != 0u) {
    var r = rate[i] * P.rateDecay;
    if (spiked) {
      r = r + P.rateAmount;
    }
    rate[i] = r;
  }
}
`,

  /** The traced neurons' voltages after the resets. */
  trace: /* wgsl */ `
${LIF_PRELUDE}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read> neurons: array<Neuron>;
@group(0) @binding(3) var<storage, read> traceNeurons: array<u32>;
@group(0) @binding(4) var<storage, read_write> traces: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x;
  if (j >= P.nTrace) {
    return;
  }
  traces[clock.kb * P.nTrace + j] = neurons[traceNeurons[j]].v;
}
`,

  /** The step counter: the absolute step and the step within the batch. */
  tick: /* wgsl */ `
${LIF_PRELUDE}
@group(0) @binding(0) var<storage, read_write> clock: Clock;

@compute @workgroup_size(1)
fn main() {
  clock.k = clock.k + 1u;
  clock.kb = clock.kb + 1u;
}
`,
} as const;

/** The graded step and the recording of chosen neurons. */
export const GRADED_WGSL = {
  /**
   * One Euler step per neuron: the synaptic input gathered over the neuron's incoming connections in their original
   * order, the light over its input columns, the feedback when present, then v + rate (((-v + bias) + syn) + current) dt.
   */
  step: /* wgsl */ `
struct Params {
  n: u32,
  hasFeedback: u32,
  dt: f32,
  pad: u32,
}

struct Node {
  bias: f32,
  rate: f32,
  inStart: u32,
  inEnd: u32,
  colStart: u32,
  colEnd: u32,
}

struct Edge {
  src: u32,
  w: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> edges: array<Edge>;
@group(0) @binding(3) var<storage, read> cols: array<u32>;
@group(0) @binding(4) var<storage, read> intensity: array<f32>;
@group(0) @binding(5) var<storage, read> feedback: array<f32>;
@group(0) @binding(6) var<storage, read> vOld: array<f32>;
@group(0) @binding(7) var<storage, read_write> vNew: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.n) {
    return;
  }
  let node = nodes[i];
  var syn = 0.0;
  for (var e = node.inStart; e < node.inEnd; e++) {
    let edge = edges[e];
    syn = syn + edge.w * max(vOld[edge.src], 0.0);
  }
  var current = 0.0;
  for (var j = node.colStart; j < node.colEnd; j++) {
    current = current + intensity[cols[j]];
  }
  if (P.hasFeedback != 0u) {
    current = current + feedback[i];
  }
  let v = vOld[i];
  let velocity = node.rate * (((-v + node.bias) + syn) + current);
  vNew[i] = v + velocity * P.dt;
}
`,

  /** The values of the neurons in `record`, gathered into `out`. */
  record: /* wgsl */ `
@group(0) @binding(0) var<storage, read> record: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x;
  if (j >= arrayLength(&record)) {
    return;
  }
  out[j] = values[record[j]];
}
`,
} as const;

/** The couplings of the hybrid: gathers over connection lists, and release above zero or above grey. */
export const COUPLING_WGSL = {
  /**
   * out[i] = post(start + sum of w x x[src] over i's connections, in their order), where start is 0 or out[i]
   * (accumulate) and post multiplies by the scale (mode 1), divides by it (mode 2) or leaves the sum (mode 0).
   */
  gather: /* wgsl */ `
struct Params {
  n: u32,
  mode: u32,
  scale: f32,
  accumulate: u32,
}

struct Edge {
  src: u32,
  w: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> edges: array<Edge>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.n) {
    return;
  }
  let r = ranges[i];
  var total = 0.0;
  if (P.accumulate != 0u) {
    total = out[i];
  }
  for (var e = r.x; e < r.y; e++) {
    let edge = edges[e];
    total = total + edge.w * x[edge.src];
  }
  if (P.mode == 1u) {
    total = total * P.scale;
  } else if (P.mode == 2u) {
    total = total / P.scale;
  }
  out[i] = total;
}
`,

  /** max(x, 0) (mode 0, the grey release) or max(x, 0) - base (mode 1, the deviation from grey). */
  release: /* wgsl */ `
struct Params {
  n: u32,
  mode: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.n) {
    return;
  }
  var r = max(x[i], 0.0);
  if (P.mode == 1u) {
    r = r - base[i];
  }
  out[i] = r;
}
`,
} as const;
