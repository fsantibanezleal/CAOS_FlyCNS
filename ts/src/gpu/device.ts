/**
 * WebGPU plumbing shared by the GPU engines: the device, buffers, readback, and the fixed-point scale of the LIF
 * delivery. The usage and map-mode flags are the WebGPU constants written as numbers, so the module runs wherever a
 * `GPUDevice` exists (a page, a worker, or Node through Dawn) without relying on the `GPUBufferUsage` global.
 */

export const USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;
const MAP_MODE_READ = 0x0001;

/** Workgroup size of every per-neuron and per-connection kernel. */
export const WORKGROUP = 256;

/** The largest sum of quantised weights allowed into one neuron: 2^30, half the int32 range, as headroom. */
const FIXED_POINT_LIMIT = 2 ** 30;

/**
 * A device from `gpu` (or `navigator.gpu`), with the adapter's largest buffer limits requested: the whole CNS needs
 * storage bindings of about 100 MB, near the default limit of 128 MiB.
 */
export async function requestDevice(gpu?: GPU): Promise<GPUDevice> {
  const g = gpu ?? (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu;
  if (!g) throw new Error("WebGPU is not available: there is no navigator.gpu and no GPU object was given");
  const adapter = await g.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU is available but no adapter was found");
  const limits = adapter.limits;
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxBufferSize: limits.maxBufferSize,
    },
  });
}

/**
 * The fixed-point scale of the LIF delivery: the largest power of two S such that the sum of |w| x S over the
 * incoming connections of any neuron stays within 2^30. A delivery is then an int32 sum, the same whatever order the
 * threads add in, and dequantising by 1/S is exact because S is a power of two.
 */
export function fixedPointScale(indices: ArrayLike<number>, weights: ArrayLike<number>, n: number): number {
  const incoming = new Float64Array(n);
  for (let e = 0; e < indices.length; e++) {
    const t = indices[e] as number;
    incoming[t] = (incoming[t] as number) + Math.abs(weights[e] as number);
  }
  let largest = 0;
  for (let i = 0; i < n; i++) largest = Math.max(largest, incoming[i] as number);
  if (largest === 0) return 1;
  return 2 ** Math.floor(Math.log2(FIXED_POINT_LIMIT / largest));
}

/** The weights on the fixed-point grid of `scale`. */
export function quantise(weights: ArrayLike<number>, scale: number): Int32Array {
  const out = new Int32Array(weights.length);
  for (let e = 0; e < weights.length; e++) out[e] = Math.round((weights[e] as number) * scale);
  return out;
}

/** A storage buffer holding `data` (or `byteLength` zero bytes); never smaller than 32 bytes, the widest stride. */
export function storageBuffer(device: GPUDevice, data: ArrayBufferView | number, label?: string): GPUBuffer {
  const byteLength = typeof data === "number" ? data : data.byteLength;
  const size = Math.max(32, Math.ceil(byteLength / 4) * 4);
  const buffer = device.createBuffer({
    label,
    size,
    usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST,
    mappedAtCreation: typeof data !== "number",
  });
  if (typeof data !== "number") {
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
  }
  return buffer;
}

/** A uniform buffer holding `data`. */
export function uniformBuffer(device: GPUDevice, data: ArrayBuffer, label?: string): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.ceil(data.byteLength / 16) * 16,
    usage: USAGE.UNIFORM | USAGE.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
  buffer.unmap();
  return buffer;
}

/** Copy `byteLength` bytes of each source into one staging buffer, submit, and return the bytes of each. */
export async function readBuffers(
  device: GPUDevice,
  parts: { buffer: GPUBuffer; byteLength: number; offset?: number }[],
): Promise<ArrayBuffer[]> {
  const sizes = parts.map((p) => Math.ceil(p.byteLength / 4) * 4);
  const total = sizes.reduce((a, b) => a + b, 0);
  if (total === 0) return parts.map(() => new ArrayBuffer(0));
  const staging = device.createBuffer({ size: total, usage: USAGE.MAP_READ | USAGE.COPY_DST });
  const encoder = device.createCommandEncoder();
  let at = 0;
  parts.forEach((p, j) => {
    const size = sizes[j] as number;
    if (size > 0) encoder.copyBufferToBuffer(p.buffer, p.offset ?? 0, staging, at, size);
    at += size;
  });
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_MODE_READ);
  const mapped = staging.getMappedRange();
  at = 0;
  const out = parts.map((p, j) => {
    const copy = mapped.slice(at, at + p.byteLength);
    at += sizes[j] as number;
    return copy;
  });
  staging.unmap();
  staging.destroy();
  return out;
}

/** A compute pipeline from one WGSL module whose entry point is `main`, with the layout inferred. */
export function pipeline(device: GPUDevice, code: string, label: string): GPUComputePipeline {
  const module = device.createShaderModule({ label, code });
  return device.createComputePipeline({ label, layout: "auto", compute: { module, entryPoint: "main" } });
}

/** A bind group giving `buffers` to bindings 0, 1, 2, ... of the pipeline's group 0. */
export function bindGroup(device: GPUDevice, p: GPUComputePipeline, buffers: GPUBuffer[], label?: string): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: p.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

/** Workgroups needed to cover `count` invocations. */
export function groups(count: number): number {
  return Math.max(1, Math.ceil(count / WORKGROUP));
}

/**
 * CSR by target of connections (`source[e]`, `target[e]`, `weight[e]`), keeping each target's connections in their
 * original order, which is the order the CPU engines sum them in. Returns per-target [start, end) pairs and the
 * connections as (source, weight as float32) pairs.
 */
export function byTarget(
  source: ArrayLike<number>,
  target: ArrayLike<number>,
  weight: ArrayLike<number>,
  nTargets: number,
): { ranges: Uint32Array; edges: ArrayBuffer } {
  const count = new Uint32Array(nTargets + 1);
  for (let e = 0; e < target.length; e++) count[(target[e] as number) + 1] = (count[(target[e] as number) + 1] as number) + 1;
  for (let t = 0; t < nTargets; t++) count[t + 1] = (count[t + 1] as number) + (count[t] as number);
  const next = count.slice(0, nTargets);
  const edges = new ArrayBuffer(Math.max(8, target.length * 8));
  const src = new Uint32Array(edges);
  const w = new Float32Array(edges);
  for (let e = 0; e < target.length; e++) {
    const t = target[e] as number;
    const at = next[t] as number;
    next[t] = at + 1;
    src[2 * at] = source[e] as number;
    w[2 * at + 1] = weight[e] as number;
  }
  const ranges = new Uint32Array(Math.max(2, nTargets * 2));
  for (let t = 0; t < nTargets; t++) {
    ranges[2 * t] = count[t] as number;
    ranges[2 * t + 1] = count[t + 1] as number;
  }
  return { ranges, edges };
}
