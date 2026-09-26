/**
 * The worker protocol: a page without WebGPU runs the CPU engine off its main thread. The handler is a plain
 * function over messages, so it is tested in Node without a Worker; `worker-entry.ts` binds it to `self`.
 *
 * Messages in:  { type: "load", scenario | bundle }  a scenario's files (name -> bytes) or a bundle's;
 *               { type: "run", steps, drive?, seed?, traceNeurons?, intensity?, frames?, preSteps?, ... }
 *               { type: "step", count, gInput? }     advance a started LIF run by `count` steps
 * Messages out: { type: "loaded", engine, n }, { type: "spikes", ... }, { type: "done", run }, { type: "error" }
 */

import { Bundle, gradedFromBundle, hybridFromBundle, lifFromBundle, loadBundle, loadScenario, type Scenario } from "./bundle.js";
import type { FileSource } from "./compiled.js";
import type { GradedEngine } from "./graded.js";
import type { HybridEngine, HybridRun } from "./hybrid.js";
import type { Drive, LIFEngine, LIFState } from "./lif.js";
import type { Run } from "./recording.js";

export type Files = Record<string, Uint8Array>;

export type WorkerRequest =
  | { type: "load"; scenario?: Files; bundle?: Files }
  | {
      type: "run";
      steps?: number;
      frames?: number;
      drive?: Drive;
      seed?: number;
      traceNeurons?: Int32Array;
      intensity?: Float64Array;
      preSteps?: number;
      grey?: number;
      recordGraded?: Int32Array;
    }
  | { type: "start"; drive?: Drive; seed?: number; traceNeurons?: Int32Array }
  | { type: "step"; count: number; gInput?: Float64Array }
  | { type: "finish" };

export type WorkerResponse =
  | { type: "loaded"; engine: string; n: number }
  | { type: "spikes"; from: number; to: number; spikes: Int32Array[] }
  | { type: "done"; run: Run; hybrid?: Omit<HybridRun, "spikes"> }
  | { type: "graded"; activity: Float32Array; final: Float64Array }
  | { type: "error"; message: string };

function filesSource(files: Files): FileSource {
  return async (name) => {
    const bytes = files[name];
    if (!bytes) throw new Error(`no file ${name}`);
    return bytes;
  };
}

/** Build a message handler; `post` receives every response. */
export function createWorkerHandler(post: (response: WorkerResponse) => void): (request: WorkerRequest) => Promise<void> {
  let bundle: Bundle | undefined;
  let scenario: Scenario | undefined;
  let lif: LIFEngine | undefined;
  let graded: GradedEngine | undefined;
  let hybrid: HybridEngine | undefined;
  let state: LIFState | undefined;

  const fail = (message: string) => post({ type: "error", message });

  return async (request) => {
    try {
      switch (request.type) {
        case "load": {
          if (request.scenario) {
            scenario = await loadScenario(filesSource(request.scenario));
            bundle = scenario.bundle;
          } else if (request.bundle) {
            scenario = undefined;
            bundle = await loadBundle(filesSource(request.bundle));
          } else {
            return fail("load needs a scenario or a bundle");
          }
          lif = graded = hybrid = undefined;
          state = undefined;
          let n = 0;
          if (bundle.engine === "lif") {
            lif = lifFromBundle(bundle);
            n = lif.n;
          } else if (bundle.engine === "graded") {
            graded = gradedFromBundle(bundle);
            n = graded.n;
          } else {
            hybrid = hybridFromBundle(bundle);
            n = hybrid.nSpiking + hybrid.graded.n;
          }
          return post({ type: "loaded", engine: bundle.engine, n });
        }
        case "run": {
          if (!bundle) return fail("nothing loaded");
          const s = scenario;
          const steps = request.steps ?? s?.run.steps ?? 0;
          const drive = request.drive ?? s?.drive ?? {};
          const seed = request.seed ?? s?.run.seed ?? 0;
          const trace = request.traceNeurons ?? s?.traceNeurons ?? new Int32Array(0);
          const intensity = request.intensity ?? s?.intensity;
          if (lif) return post({ type: "done", run: lif.run(steps, drive, seed, trace) });
          if (graded) {
            if (!intensity) return fail("a graded run needs frames of intensity");
            const frames = request.frames ?? steps;
            const out = graded.run(intensity, frames);
            return post({ type: "graded", activity: out.activity, final: out.final });
          }
          if (hybrid) {
            if (!intensity) return fail("a hybrid run needs frames of intensity");
            const frames = request.frames ?? steps;
            const preSteps = request.preSteps ?? s?.run.pre_steps;
            if (preSteps === undefined) return fail("a hybrid run needs its lead-in step count");
            const out = hybrid.run(intensity, frames, {
              drive,
              seed,
              preSteps,
              grey: request.grey ?? s?.run.grey,
              recordGraded: request.recordGraded ?? s?.recordGraded,
              traceSpiking: trace,
              onFrame: (f, spikes) => post({ type: "spikes", from: f, to: f + 1, spikes }),
            });
            const { spikes, ...rest } = out;
            return post({ type: "done", run: spikes, hybrid: rest });
          }
          return fail("no engine");
        }
        case "start": {
          if (!lif) return fail("start needs a LIF bundle");
          state = lif.start(request.drive ?? {}, request.seed ?? 0, request.traceNeurons);
          return;
        }
        case "step": {
          if (!lif || !state) return fail("step before start");
          const from = state.k;
          const spikes: Int32Array[] = [];
          for (let i = 0; i < request.count; i++) spikes.push(lif.advance(state, request.gInput));
          return post({ type: "spikes", from, to: state.k, spikes });
        }
        case "finish": {
          if (!lif || !state) return fail("finish before start");
          return post({ type: "done", run: lif.finish(state) });
        }
      }
    } catch (error) {
      fail((error as Error).message);
    }
  };
}
