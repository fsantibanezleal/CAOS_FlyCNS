/**
 * Engine bundles and scenarios (`flycns.bundle`): the arrays an engine steps over, with every constant of the step
 * in the manifest. `loadBundle` verifies and reads one; the `*FromBundle` builders make the CPU engines from it;
 * `loadScenario` reads a parity scenario (a bundle plus its run).
 */

import { Compiled, type FileSource, readCompiled } from "./compiled.js";
import { GradedEngine, type GradedNetwork } from "./graded.js";
import { HybridEngine, type HybridConstants, type HybridNetwork, type LatticeMap, LatticeSource } from "./hybrid.js";
import { type Drive, LIFEngine, type LIFConstants, type LIFNetwork } from "./lif.js";

export const ENGINE_SCHEMA = "flycns.engine/1";
export const SCENARIO_SCHEMA = "flycns.scenario/1";

export interface BundleRelease {
  engine: "lif" | "graded" | "hybrid";
  lif?: LIFConstants;
  graded?: { dt_s: number; n_columns: number };
  hybrid?: HybridConstants;
  source?: "own" | "lattice";
  lattice?: { n_columns: number; count: number };
  n_neurons?: number;
  n_spiking?: number;
  n_graded?: number;
  [key: string]: unknown;
}

export interface ScenarioRun {
  steps: number;
  seed: number;
  t_pre_s: number;
  grey: number;
  modulated_steps: number;
  pre_steps?: number;
}

export class Bundle {
  constructor(readonly compiled: Compiled) {}

  get release(): BundleRelease {
    return this.compiled.release as unknown as BundleRelease;
  }

  get engine(): BundleRelease["engine"] {
    return this.release.engine;
  }

  lifNetwork(): LIFNetwork {
    const c = this.compiled;
    return { indptr: c.numbers("lif_indptr"), indices: c.int32("lif_indices"), weightsMv: c.float64("lif_weight_mv") };
  }

  gradedNetwork(prefix = "graded", nColumns?: number): GradedNetwork {
    const c = this.compiled;
    const columns = nColumns ?? (prefix === "graded" ? this.release.graded?.n_columns : this.release.lattice?.n_columns);
    if (columns === undefined) throw new Error(`the bundle states no column count for ${prefix}`);
    return {
      bias: c.float64(`${prefix}_bias`),
      timeConstS: c.float64(`${prefix}_time_const_s`),
      source: c.int32(`${prefix}_source`),
      target: c.int32(`${prefix}_target`),
      weight: c.float64(`${prefix}_weight`),
      inputNeuron: c.int32(`${prefix}_input_neuron`),
      inputColumn: c.int32(`${prefix}_input_column`),
      nColumns: columns,
    };
  }

  hybridNetwork(): HybridNetwork {
    const c = this.compiled;
    return {
      lif: this.lifNetwork(),
      graded: this.gradedNetwork(),
      bridgeSource: c.int32("bridge_source"),
      bridgeTarget: c.int32("bridge_target"),
      bridgeWeight: c.float64("bridge_weight"),
      feedbackSource: c.int32("feedback_source"),
      feedbackTarget: c.int32("feedback_target"),
      feedbackWeight: c.float64("feedback_weight"),
      spikingNeuron: c.has("spiking_neuron") ? c.numbers("spiking_neuron") : undefined,
    };
  }

  latticeMap(): LatticeMap {
    const c = this.compiled;
    return {
      unit: c.int32("map_unit"),
      side: c.get("map_side") as Uint8Array,
      indptr: c.numbers("map_indptr"),
      node: c.int32("map_node"),
      weight: c.float32("map_weight"),
    };
  }

  latticeNetworks(): GradedNetwork[] {
    const lattice = this.release.lattice;
    if (!lattice) throw new Error("the bundle has no lattice source");
    return Array.from({ length: lattice.count }, (_, k) => this.gradedNetwork(`lattice${k}`, lattice.n_columns));
  }
}

export async function loadBundle(source: FileSource, options: { verify?: boolean; only?: string[] } = {}): Promise<Bundle> {
  return new Bundle(await readCompiled(source, { schema: ENGINE_SCHEMA, ...options }));
}

export function lifFromBundle(bundle: Bundle): LIFEngine {
  const constants = bundle.release.lif;
  if (!constants) throw new Error("the bundle has no LIF constants");
  return new LIFEngine(bundle.lifNetwork(), constants);
}

export function gradedFromBundle(bundle: Bundle): GradedEngine {
  const graded = bundle.release.graded;
  if (!graded) throw new Error("the bundle has no graded constants");
  return new GradedEngine(bundle.gradedNetwork(), graded.dt_s);
}

export function hybridFromBundle(bundle: Bundle): HybridEngine {
  const { lif, hybrid, graded } = bundle.release;
  if (!lif || !hybrid || !graded) throw new Error("the bundle has no hybrid constants");
  const engine = new HybridEngine(bundle.hybridNetwork(), lif, hybrid);
  if (bundle.release.source === "lattice") {
    const engines = bundle.latticeNetworks().map((net) => new GradedEngine(net, graded.dt_s));
    engine.source = new LatticeSource(engines, bundle.latticeMap(), engine.graded.n);
  }
  return engine;
}

/** A parity scenario: the bundle, the drive and the run's settings (`flycns.bundle.write_scenario`). */
export interface Scenario {
  bundle: Bundle;
  run: ScenarioRun;
  drive: Drive;
  traceNeurons: Int32Array;
  /** row-major frames of intensity, or undefined for a LIF scenario */
  intensity?: Float64Array;
  recordGraded?: Int32Array;
}

export function driveFromCompiled(c: Compiled, modulatedSteps: number): Drive {
  const activate = new Map<number, number>();
  const neurons = c.numbers("drive_activate_neuron");
  const rates = c.float64("drive_activate_rate_hz");
  for (let j = 0; j < neurons.length; j++) activate.set(neurons[j] as number, rates[j] as number);
  const events = new Map<number, { neurons: number[]; dvMv: number[] }>();
  const steps = c.numbers("drive_event_step");
  const eventNeurons = c.numbers("drive_event_neuron");
  const dv = c.float64("drive_event_dv_mv");
  for (let j = 0; j < steps.length; j++) {
    const step = steps[j] as number;
    let entry = events.get(step);
    if (!entry) {
      entry = { neurons: [], dvMv: [] };
      events.set(step, entry);
    }
    entry.neurons.push(eventNeurons[j] as number);
    entry.dvMv.push(dv[j] as number);
  }
  const drive: Drive = { activate, events, silenced: c.numbers("drive_silenced") };
  if (c.has("drive_modulated_neuron")) {
    const modNeurons = c.int32("drive_modulated_neuron");
    const shape = c.shape("drive_modulated_rate_hz");
    drive.modulated = {
      neurons: modNeurons,
      rates: c.float64("drive_modulated_rate_hz"),
      frames: shape[0] as number,
      stepsPerFrame: modulatedSteps,
    };
  }
  return drive;
}

export async function loadScenario(source: FileSource): Promise<Scenario> {
  const compiled = await readCompiled(source, { schema: SCENARIO_SCHEMA });
  const release = compiled.release as unknown as BundleRelease & { run: ScenarioRun };
  const run = release.run;
  return {
    bundle: new Bundle(compiled),
    run,
    drive: driveFromCompiled(compiled, run.modulated_steps),
    traceNeurons: compiled.int32("trace_neurons"),
    intensity: compiled.has("stimulus_intensity") ? compiled.float64("stimulus_intensity") : undefined,
    recordGraded: compiled.has("record_graded") ? compiled.int32("record_graded") : undefined,
  };
}
