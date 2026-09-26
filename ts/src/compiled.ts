/**
 * The compiled directory (`flycns.compiled`): a manifest and one little-endian binary file per array. The loader
 * refuses a directory whose schema is not the expected one, whose arrays do not match the manifest's SHA-256, or
 * whose byte length does not match the declared shape: an engine never runs on a graph that is not the one that was
 * compiled.
 *
 * Files come from a `FileSource`, so the same loader reads from `fetch` in a page and from the file system in Node.
 */

export const COMPILED_SCHEMA = "flycns.compiled/1";

export type FileSource = (name: string) => Promise<Uint8Array>;

export type TypedArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | Float32Array
  | Float64Array;

export interface ArrayEntry {
  name: string;
  dtype: string;
  shape: number[];
  file: string;
  bytes: number;
  sha256: string;
}

export interface Manifest {
  schema: string;
  release: Record<string, unknown>;
  sources: unknown[];
  counts: Record<string, unknown>;
  strings: Record<string, string[]>;
  arrays: ArrayEntry[];
}

export class CompiledError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CompiledError";
  }
}

const CONSTRUCTORS: Record<string, { new (buffer: ArrayBuffer): TypedArray; BYTES_PER_ELEMENT: number }> = {
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  float32: Float32Array,
  float64: Float64Array,
};

if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
  throw new Error("flycns reads little-endian arrays and this platform is big-endian");
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

function product(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** A verified compiled directory: its manifest and its arrays by name. */
export class Compiled {
  constructor(
    readonly manifest: Manifest,
    readonly arrays: Map<string, TypedArray>,
  ) {}

  get release(): Record<string, unknown> {
    return this.manifest.release;
  }

  get counts(): Record<string, unknown> {
    return this.manifest.counts;
  }

  get strings(): Record<string, string[]> {
    return this.manifest.strings;
  }

  has(name: string): boolean {
    return this.arrays.has(name);
  }

  shape(name: string): number[] {
    const entry = this.manifest.arrays.find((e) => e.name === name);
    if (!entry) throw new CompiledError(`no array named ${name}`);
    return entry.shape;
  }

  get(name: string): TypedArray {
    const array = this.arrays.get(name);
    if (array === undefined) throw new CompiledError(`no array named ${name}`);
    return array;
  }

  /** An integer array as plain numbers (int64 included; every value must be a safe integer). */
  numbers(name: string): Float64Array {
    return toNumbers(this.get(name));
  }

  /** An index array as Int32Array (int32 and int64 accepted; every value must fit). */
  int32(name: string): Int32Array {
    return toInt32(this.get(name));
  }

  float64(name: string): Float64Array {
    const a = this.get(name);
    if (a instanceof Float64Array) return a;
    if (a instanceof Float32Array) return Float64Array.from(a);
    throw new CompiledError(`${name} is ${a.constructor.name}, not a float array`);
  }

  float32(name: string): Float32Array {
    const a = this.get(name);
    if (a instanceof Float32Array) return a;
    if (a instanceof Float64Array) return Float32Array.from(a);
    throw new CompiledError(`${name} is ${a.constructor.name}, not a float array`);
  }
}

export function toNumbers(a: TypedArray): Float64Array {
  if (a instanceof BigInt64Array) {
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) {
      const v = a[i] as bigint;
      if (v > BigInt(Number.MAX_SAFE_INTEGER) || v < -BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new CompiledError(`an int64 value (${v}) is not a safe integer`);
      }
      out[i] = Number(v);
    }
    return out;
  }
  return Float64Array.from(a as ArrayLike<number>);
}

export function toInt32(a: TypedArray): Int32Array {
  if (a instanceof Int32Array) return a;
  const numbers = toNumbers(a);
  for (let i = 0; i < numbers.length; i++) {
    const v = numbers[i] as number;
    if (v > 2147483647 || v < -2147483648 || !Number.isInteger(v)) {
      throw new CompiledError(`a value (${v}) does not fit an Int32Array`);
    }
  }
  return Int32Array.from(numbers);
}

/** Decode one array file against its manifest entry: dtype, byte length and (unless `verify` is false) SHA-256. */
export async function decodeArray(entry: ArrayEntry, bytes: Uint8Array, verify = true): Promise<TypedArray> {
  const ctor = CONSTRUCTORS[entry.dtype];
  if (!ctor) throw new CompiledError(`array ${entry.name}: dtype ${entry.dtype} is not allowed`);
  const expected = product(entry.shape) * ctor.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expected) {
    throw new CompiledError(
      `array ${entry.name}: ${bytes.byteLength} bytes where shape [${entry.shape}] needs ${expected}`,
    );
  }
  if (verify) {
    const digest = await sha256Hex(bytes);
    if (digest !== entry.sha256) {
      throw new CompiledError(`array ${entry.name}: SHA-256 ${digest} differs from the manifest's ${entry.sha256}`);
    }
  }
  // a fresh, aligned buffer: file readers hand out views into pooled memory whose offset need not be a multiple of
  // the element size
  const aligned = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(aligned).set(bytes);
  return new ctor(aligned);
}

export async function readManifest(source: FileSource, schema: string): Promise<Manifest> {
  let text: string;
  try {
    text = new TextDecoder("utf-8").decode(await source("manifest.json"));
  } catch (error) {
    throw new CompiledError(`no readable manifest.json (${(error as Error).message})`);
  }
  const manifest = JSON.parse(text) as Manifest;
  if (manifest.schema !== schema) throw new CompiledError(`schema ${manifest.schema} is not ${schema}`);
  return manifest;
}

/**
 * Read a compiled directory, checking every array against the manifest. `only` restricts the arrays read (a page
 * loads partition-first); every array read is still verified.
 */
export async function readCompiled(
  source: FileSource,
  options: { schema?: string; verify?: boolean; only?: string[] } = {},
): Promise<Compiled> {
  const manifest = await readManifest(source, options.schema ?? COMPILED_SCHEMA);
  const arrays = new Map<string, TypedArray>();
  for (const entry of manifest.arrays) {
    if (options.only && !options.only.includes(entry.name)) continue;
    let bytes: Uint8Array;
    try {
      bytes = await source(entry.file);
    } catch (error) {
      throw new CompiledError(`array ${entry.name} is missing its file ${entry.file} (${(error as Error).message})`);
    }
    arrays.set(entry.name, await decodeArray(entry, bytes, options.verify ?? true));
  }
  return new Compiled(manifest, arrays);
}

/** A file source over `fetch`, for a directory published at `baseUrl` (with or without a trailing slash). */
export function fetchSource(baseUrl: string, init?: RequestInit): FileSource {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return async (name) => {
    const response = await fetch(base + name, init);
    if (!response.ok) throw new Error(`${response.status} for ${base + name}`);
    return new Uint8Array(await response.arrayBuffer());
  };
}
