#!/usr/bin/env python3
"""Extract the flyvis 1.2.0 pretrained ensemble into flycns's format, with a parity recording of flyvis itself.

flyvis (Lappalainen et al., Nature 2024, doi:10.1038/s41586-024-07939-3; code MIT, github.com/TuragaLab/flyvis) is a
type-level model of the fly's visual system on a 721-column hexagonal lattice; its 50 pretrained networks share one
connectome and differ in 734 trained numbers each (a resting potential and a time constant per cell type, a unitary
strength per pair of types). flycns transfers those numbers onto the neuron-level MaleCNS optic lobes, so it needs
them, in flyvis's own order, as data.

This script needs flyvis 1.2.0 and its dependencies, which flycns does not depend on: run it in a separate
environment (Python 3.9 to 3.12) with flyvis and flycns installed, and ``FLYVIS_ROOT_DIR`` pointing at the extracted
``results_pretrained_models.zip``. It writes three directories in the compiled style (a manifest with the SHA-256 of
every array):

- ``ensemble/``: the parameter orders (types, pairs of types, synapse-count groups) and, for each of the 50 networks,
  the resting potentials, time constants and unitary strengths; the signs and mean synapse counts, which training
  never changes, once (the script checks that they are equal in every network).
- ``lattice-000/``: network 000 in full (nodes with type and hexagonal coordinates, edges with their weight
  ``sign x synapse count x strength``, node parameters, the photoreceptor input index), so another engine can run
  exactly the network flyvis runs.
- ``responses-000/``: what flyvis computes for network 000 on a fixed stimulus (grey, a full-field flash, a moving
  edge), every node at every step, from flyvis's own steady state. This is the parity target of the graded engine.

Usage::

    FLYVIS_ROOT_DIR=/path/to/flyvis_models python scripts/extract_flyvis_ensemble.py OUT_DIR
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

N_MODELS = 50
DT_S = 1 / 200            # flyvis's evaluation step (the training step was 1/50 s)
T_PRE_S = 2.0             # grey before the stimulus, as flyvis's steady_state
GREY = 0.5


def stimulus(u: np.ndarray, v: np.ndarray, n_frames: int) -> np.ndarray:
    """Input per frame and column: grey 0.5; a full-field flash to 1.0 from 100 to 300 ms; grey; then an ON edge
    sweeping along the u axis at one column per 20 ms from 400 ms, leaving the swept columns at 1.0."""
    x = np.full((n_frames, len(u)), GREY, dtype=np.float32)
    t = np.arange(n_frames) * DT_S
    flash = (t >= 0.100) & (t < 0.300)
    x[flash] = 1.0
    start = 0.400
    for i, ti in enumerate(t):
        if ti >= start:
            front = u.min() + (ti - start) / 0.020
            x[i, u <= front] = 1.0
    return x


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows_safe_datamate() -> None:
    """datamate 1.0.0 (flyvis's storage layer) opens an h5 file for writing, fails to find its dataset, and deletes
    the file while its handle is still open, which Windows refuses (WinError 32) the first time flyvis builds its
    connectome. The replacement closes every handle before replacing the file; what it writes is the same (one
    dataset named "data"). Harmless elsewhere."""
    import datamate.directory
    import datamate.io
    import h5py

    def write_h5(path: Path, val) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        elif path.exists():
            path.unlink()
        with h5py.File(path, libver="latest", mode="w") as f:
            f["data"] = np.asarray(val)

    datamate.io._write_h5 = write_h5
    datamate.directory._write_h5 = write_h5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", type=int, default=200, help="stimulus frames at 1/200 s (default 1 s)")
    args = parser.parse_args()

    windows_safe_datamate()
    import flyvis
    import torch
    from flyvis import NetworkView

    from flycns.compiled import write_compiled

    root = Path(flyvis.results_dir)
    view = NetworkView("flow/0000/000")
    net = view.init_network(checkpoint="best")
    ctome = net.connectome

    # --- the parameter orders, from flyvis's own parameter objects
    bias_p, tau_p = net.node_params["bias"], net.node_params["time_const"]
    sign_p, strength_p, count_p = (net.edge_params[k] for k in ("sign", "syn_strength", "syn_count"))
    types = [str(k) for k in bias_p.keys]
    assert types == [str(k) for k in tau_p.keys]
    pairs = [(str(a), str(b)) for a, b in strength_p.keys]
    assert pairs == [(str(a), str(b)) for a, b in sign_p.keys]
    groups = [(str(a), str(b), int(du), int(dv)) for a, b, du, dv in count_p.keys]
    type_index = {t: i for i, t in enumerate(types)}
    pair_index = {p: i for i, p in enumerate(pairs)}

    # --- the 50 checkpoints, raw values turned into the values the dynamics use
    checkpoints = [root / "flow" / "0000" / f"{m:03d}" / "best_chkpt" for m in range(N_MODELS)]
    bias = np.zeros((N_MODELS, len(types)), dtype=np.float32)
    tau = np.zeros((N_MODELS, len(types)), dtype=np.float32)
    strength = np.zeros((N_MODELS, len(pairs)), dtype=np.float32)
    signs = counts = None
    for m, path in enumerate(checkpoints):
        state = torch.load(path, map_location="cpu", weights_only=False)["network"]
        bias[m] = state["nodes_bias"].numpy()
        tau[m] = state["nodes_time_const"].numpy()
        strength[m] = state["edges_syn_strength"].numpy()
        s = state["edges_sign"].numpy().astype(np.int8)
        c = np.exp(state["edges_syn_count"].numpy().astype(np.float64))     # Lognormal: n_syn = exp(raw)
        if signs is None:
            signs, counts = s, c
        assert np.array_equal(s, signs), f"sign differs in network {m}"
        assert np.allclose(c, counts, rtol=1e-6), f"synapse counts differ in network {m}"

    # the same numbers the network itself uses (a check of the orders above)
    params = net._param_api()
    node_types = ctome.nodes.type[:].astype(str)
    assert np.allclose(params.nodes.bias.detach().numpy(), bias[0][[type_index[t] for t in node_types]])
    edge_src_t = ctome.edges.source_type[:].astype(str)
    edge_tar_t = ctome.edges.target_type[:].astype(str)
    edge_pair = np.array([pair_index[(a, b)] for a, b in zip(edge_src_t, edge_tar_t, strict=True)], dtype=np.int32)
    weight = params.edges.weight.detach().numpy().astype(np.float32)
    assert np.allclose(weight, signs[edge_pair] * params.edges.syn_count.detach().numpy() * strength[0][edge_pair])

    source = {
        "key": "flyvis-1.2.0-pretrained-flow-0000",
        "url": "https://github.com/TuragaLab/flyvis (results_pretrained_models.zip, via flyvis download_pretrained)",
        "license": "MIT (Copyright (c) 2023 Janne K. Lappalainen, Fabian D. Tschopp, Mason McGill, Jakob H. Macke, "
                   "Srinivas C. Turaga)",
        "checkpoints_sha256": [sha256_file(p) for p in checkpoints],
    }
    release = {"model": "flyvis", "version": "1.2.0", "ensemble": "flow/0000", "paper": "doi:10.1038/s41586-024-07939-3"}
    write_compiled(args.out / "ensemble", {
        "pair_source": np.array([type_index[a] for a, _ in pairs], dtype=np.uint16),
        "pair_target": np.array([type_index[b] for _, b in pairs], dtype=np.uint16),
        "pair_sign": signs,
        "group_pair": np.array([pair_index[(a, b)] for a, b, _, _ in groups], dtype=np.uint16),
        "group_du": np.array([du for *_, du, _ in groups], dtype=np.int16),
        "group_dv": np.array([dv for *_, dv in groups], dtype=np.int16),
        "group_n_syn": counts.astype(np.float32),
        "bias": bias,
        "time_const_s": tau,
        "strength": strength,
    }, {"release": release, "sources": [source],
        "counts": {"models": N_MODELS, "types": len(types), "pairs": len(pairs), "groups": len(groups)},
        "strings": {"type": types}})

    # --- network 000 in full
    u = ctome.nodes.u[:].astype(np.int16)
    v = ctome.nodes.v[:].astype(np.int16)
    input_types = ctome.input_cell_types[:].astype(str).tolist()
    input_index = np.stack([ctome.nodes.layer_index[t][:] for t in input_types]).astype(np.int32)
    write_compiled(args.out / "lattice-000", {
        "node_type": np.array([type_index[t] for t in node_types], dtype=np.uint16),
        "node_u": u,
        "node_v": v,
        "node_bias": params.nodes.bias.detach().numpy().astype(np.float32),
        "node_time_const_s": params.nodes.time_const.detach().numpy().astype(np.float32),
        "edge_source": ctome.edges.source_index[:].astype(np.int32),
        "edge_target": ctome.edges.target_index[:].astype(np.int32),
        "edge_pair": edge_pair,
        "edge_weight": weight,
        "input_index": input_index,
    }, {"release": {**release, "network": "000"}, "sources": [source],
        "counts": {"nodes": int(net.n_nodes), "edges": int(net.n_edges), "columns": int(input_index.shape[1])},
        "strings": {"type": types, "input_type": input_types}})

    # --- what flyvis computes: steady state under grey, then the stimulus, every node at every step
    columns_u = u[input_index[0]].astype(np.float64)
    columns_v = v[input_index[0]].astype(np.float64)
    x = stimulus(columns_u, columns_v, args.frames)
    with torch.no_grad():
        initial = net.steady_state(t_pre=T_PRE_S, dt=DT_S, batch_size=1, value=GREY)
        net.stimulus.zero(1, args.frames)
        net.stimulus.add_input(torch.from_numpy(x)[None, :, None, :])
        activity = net(net.stimulus(), DT_S, state=initial).numpy()[0].astype(np.float32)
    write_compiled(args.out / "responses-000", {
        "initial_activity": initial.nodes.activity.detach().numpy()[0].astype(np.float32),
        "stimulus": x,
        "activity": activity,
    }, {"release": {**release, "network": "000", "dt_s": DT_S, "t_pre_s": T_PRE_S, "grey": GREY,
                    "stimulus": "grey; full-field flash to 1.0 from 100 to 300 ms; ON edge along u from 400 ms at "
                                "one column per 20 ms"},
        "sources": [source], "counts": {"frames": args.frames, "nodes": int(net.n_nodes)}})
    print(f"wrote {args.out}: {len(types)} types, {len(pairs)} pairs, {len(groups)} groups, {N_MODELS} networks; "
          f"lattice {net.n_nodes} nodes, {net.n_edges} edges; responses {args.frames} frames")


if __name__ == "__main__":
    main()
