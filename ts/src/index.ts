/**
 * @fasl-work/flycns: the browser side of flycns.
 *
 * The display version (X.XX.XXX) lives in the repository's VERSION file; this constant is its semantic form and
 * must equal the npm manifest's version (a test checks it).
 */
export const VERSION = "0.7.0";

export { hash3, fmix32, eventThreshold, poissonEvent } from "./rng.js";
export {
  Compiled,
  CompiledError,
  COMPILED_SCHEMA,
  readCompiled,
  readManifest,
  decodeArray,
  fetchSource,
  sha256Hex,
  toNumbers,
  toInt32,
} from "./compiled.js";
export type { FileSource, Manifest, ArrayEntry, TypedArray } from "./compiled.js";
export { readRecording, readExpected, spikeCounts, ratesHz, spikeTimes, RECORDING_SCHEMA, EXPECTED_SCHEMA } from "./recording.js";
export type { Run } from "./recording.js";
export { LIFEngine, LIFState } from "./lif.js";
export type { LIFConstants, LIFNetwork, Drive, ModulatedDrive } from "./lif.js";
export { GradedEngine } from "./graded.js";
export type { GradedNetwork } from "./graded.js";
export { HybridEngine, OwnSource, LatticeSource } from "./hybrid.js";
export type { HybridConstants, HybridNetwork, HybridRun, LatticeMap, Source } from "./hybrid.js";
export {
  Bundle,
  loadBundle,
  loadScenario,
  lifFromBundle,
  gradedFromBundle,
  hybridFromBundle,
  driveFromCompiled,
  ENGINE_SCHEMA,
  SCENARIO_SCHEMA,
} from "./bundle.js";
export type { BundleRelease, Scenario, ScenarioRun } from "./bundle.js";
export { createWorkerHandler } from "./worker.js";
export type { WorkerRequest, WorkerResponse, Files } from "./worker.js";
export { HASH_WGSL, LIF_WGSL, GRADED_WGSL, COUPLING_WGSL } from "./gpu/shaders.js";
export { LIFGpu } from "./gpu/lif-gpu.js";
export { HybridGpu } from "./gpu/hybrid-gpu.js";
export { fixedPointScale, requestDevice } from "./gpu/device.js";
