# The spiking model (U3a): design

## The model, exactly as published

Shiu et al., *Nature* 634:210-219 (2024), doi:10.1038/s41586-024-07763-9; code `philshiu/Drosophila_brain_model`
(`model.py`, MIT). Every neuron is a leaky integrate-and-fire point neuron with a synaptic variable that decays:

$$\tau_m \frac{dv}{dt} = v_0 - v + g, \qquad \tau \frac{dg}{dt} = -g,$$

both frozen while the neuron is refractory. A spike is emitted when $v > v_\text{th}$; the neuron is then reset,
$v \leftarrow v_\text{rst}$, $g \leftarrow 0$, and stays refractory for $t_\text{rfc}$. A presynaptic spike adds
$w_{ij} = s_j\, n_{ij}\, w_\text{syn}$ to $g_i$ after the delay $t_\text{dly}$, where $s_j$ is the presynaptic sign and
$n_{ij}$ the synapse count. Constants: $v_0 = v_\text{rst} = -52$ mV, $v_\text{th} = -45$ mV, $\tau_m = 20$ ms,
$\tau = 5$ ms, $t_\text{rfc} = 2.2$ ms, $t_\text{dly} = 1.8$ ms, $w_\text{syn} = 0.275$ mV, $\Delta t = 0.1$ ms.

**Exact integration.** Between events the system is linear, so each step uses its closed form (what Brian2's
`linear` method computes):

$$g(t+\Delta t) = g\,e^{-\Delta t/\tau}, \qquad
v(t+\Delta t) = v_0 + (v - v_0)\,e^{-\Delta t/\tau_m} + g\,\frac{\tau}{\tau - \tau_m}\left(e^{-\Delta t/\tau} - e^{-\Delta t/\tau_m}\right).$$

**Activation** (optogenetic, in the paper): Poisson events at a rate $r$ add $w_\text{syn} f_\text{poi}$ to $v$ directly
($f_\text{poi} = 250$, so one event crosses threshold), and the activated neurons have no refractory period.
**Silencing** zeroes every synapse *from* a neuron, as `silence()` in `model.py` does
(`syn.w['{i} == i'] = 0`, the presynaptic index): the silenced neuron still receives input and may spike, and its
spikes reach no one. It can be given through the weights (`synaptic_weights(..., silenced=...)`) or through the
drive (`Drive.silenced`, which drops the neuron's spikes before delivery); the two give the same run.

## The order of one step (Brian2's default schedule)

1. **groups**: integrate $v$ and $g$ for neurons that are not refractory (refractoriness ends once
   $t - t_\text{last} \ge t_\text{rfc}$ in whole steps);
2. **thresholds**: neurons with $v > v_\text{th}$ that are not refractory spike; their last-spike time is now;
3. **synapses**: spikes emitted $t_\text{dly}$ ago add their weights to $g$; Poisson and fixed events add to $v$;
   **only neurons that are not refractory receive them**, and a neuron that spiked this step already is refractory;
4. **resets**: neurons that spiked this step get $v \leftarrow v_\text{rst}$, $g \leftarrow 0$.

**Input that arrives during refractoriness is discarded.** Both $v$ and $g$ are declared `(unless refractory)` in the
published equations, and for such variables Brian2 applies a synaptic update only to postsynaptic neurons that are not
refractory. Its generated code for `v += w` reads `add.at(v, post[not_refractory], w[not_refractory])`, and the
thresholder clears `not_refractory` for the neurons that spike. So a refractory neuron neither integrates nor
accumulates input: whatever reaches it in those 2.2 ms is lost. An input on the spike step is lost as well, to the
reset. The first transcription here kept refractory input in the frozen $g$ and diverged from Brian2 at the tenth
spike of the first parity circuit; the parity test found it (2026-09-23).

The parity test is a literal Brian2 transcription of `model.py` run on three small circuits (40 neurons, 220
connections, 2,000 steps) with deterministic drive; it must give the same spike times and neuron indices, and does.

## Implementations

| Engine | Where | Arithmetic | Determinism |
|---|---|---|---|
| `LIFReference` | NumPy, CPU | float64 state; delivery by accumulating each spiking neuron's CSR row in a fixed order | a pure function of (graph, drive, seed) |
| `LIFTorch` | PyTorch, CUDA or CPU | float32 state; delivery pushed from the spiking neurons' CSR rows with `index_add_` | compared by tolerance, never claimed bit-identical (R-305: the same neurons and counts over 200 ms of moderate drive; under strong drive, a sample of the same process) |

**Randomness is counter-based.** A Poisson event for neuron $i$ at step $k$ occurs when
$h(\text{seed}; i, k) < \lfloor r\,\Delta t\,2^{32} \rfloor$, where $h$ is MurmurHash3_x86_32 (Appleby, public
domain) of the 8-byte little-endian key $(i, k)$ seeded with the run's seed. It is made of 32-bit multiplies, adds,
xors, shifts and rotations, so TypeScript (`Math.imul`, `>>> 0`) and WGSL (`u32`) compute it identically, and any
MurmurHash3 implementation checks it (the tests use the `mmh3` package). A first ad hoc hash had a fixed point at zero
(seed 0, neuron 0, step 0 always gave an event) and was replaced before release.

**Why agreement is measured two ways.** On the whole MaleCNS the published model has two states. In the low one,
under the moderate drive, the network fires about 8,000 spikes per 50 ms; at a random moment it can switch into a
high one of about 44,000 per 50 ms, carried by the central brain with the mushroom-body output and dopaminergic
neurons (MBON11, PPL101) among the most active. Under the strong drive the 500 ms total therefore ranges from about
0.27 million spikes (no switch) to 0.52 million (an early one), seed by seed. Float32 and float64 runs share every
input event, but individual spike times part once the two arithmetics first decide a threshold differently (after
about 6,300 spikes of the moderate drive, around 110 ms), and from then on the runs are two samples of the same
process, which may switch at different moments. So over 200 ms of moderate drive the GPU engine must fire the same
neurons the same number of times (measured: Jaccard 1.0, count correlation 1.0000), and under the strong drive its
trials must look like trials of the reference: five GPU trials against five independent reference trials correlate
at 0.991, where the reference against itself, over the 126 splits of ten seeds, ranges from 0.778 to 0.996 with a
5th percentile of 0.891 and a median of 0.983.

## Recording

Spikes are recorded as a CSR over steps (`tick_indptr`, `neuron_index`), plus optional voltage traces of chosen
neurons; the recording is a directory in the compiled-format style (manifest with hashes and arrays).

## Null graphs

Built from the compiled graph, each returning a new CSR with the same number of connections:

- **N1, degree-preserving rewiring**: within each partition block (a pair of pre and post partitions among the left
  and right optic lobes, the central brain and the nerve cord), the targets of the block's connections are permuted.
  Every connection keeps its presynaptic neuron, so its sign, and its count; every neuron keeps its out-degree and its
  in-degree, and every block its number of connections. A permutation can create a self connection or a repeated
  pair; each such collision is repaired by exchanging its target with that of a random connection of the same block,
  for at most 50 rounds. Collisions still left are dropped and reported (`unresolved`), so the degrees are exact
  whenever `unresolved` is 0.
- **N2, size-matched random graph**: each block keeps its number of connections, placed between uniformly random
  distinct pairs of its pre and post partitions (no self connections, no repeats), carrying the block's real counts in
  random order.
- **N3, sign shuffle**: the wiring and counts unchanged, the neuron signs permuted among the neurons that have a sign.
