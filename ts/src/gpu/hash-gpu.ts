/** The WGSL hash over a list of keys, for checking it against `hash3` and the Python reference. */

import { bindGroup, groups, pipeline, readBuffers, storageBuffer } from "./device.js";
import { HASH_KERNEL_WGSL } from "./shaders.js";

export async function hashGpu(
  device: GPUDevice,
  seed: ArrayLike<number>,
  neuron: ArrayLike<number>,
  step: ArrayLike<number>,
): Promise<Uint32Array> {
  const count = seed.length;
  const keys = new Uint32Array(count * 3);
  for (let i = 0; i < count; i++) {
    keys[3 * i] = seed[i] as number;
    keys[3 * i + 1] = neuron[i] as number;
    keys[3 * i + 2] = step[i] as number;
  }
  const p = pipeline(device, HASH_KERNEL_WGSL, "hash");
  const keyBuffer = storageBuffer(device, keys, "hash keys");
  const out = device.createBuffer({ size: Math.max(4, count * 4), usage: 0x0080 | 0x0004 });
  const encoder = device.createCommandEncoder({ label: "hash" });
  const pass = encoder.beginComputePass();
  pass.setPipeline(p);
  pass.setBindGroup(0, bindGroup(device, p, [keyBuffer, out]));
  pass.dispatchWorkgroups(groups(count));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const [bytes] = await readBuffers(device, [{ buffer: out, byteLength: count * 4 }]);
  keyBuffer.destroy();
  out.destroy();
  return new Uint32Array(bytes as ArrayBuffer);
}
