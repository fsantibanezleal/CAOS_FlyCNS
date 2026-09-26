/**
 * The Web Worker entry: `new Worker(new URL("@fasl-work/flycns/worker", import.meta.url), { type: "module" })`.
 * Binds the protocol handler to the worker's message port.
 */

import { createWorkerHandler, type WorkerRequest } from "./worker.js";

const scope = self as unknown as { postMessage: (m: unknown) => void; onmessage: ((e: MessageEvent) => void) | null };
const handle = createWorkerHandler((response) => scope.postMessage(response));
scope.onmessage = (event: MessageEvent) => {
  void handle(event.data as WorkerRequest);
};
