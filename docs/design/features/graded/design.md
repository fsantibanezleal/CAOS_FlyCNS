# The graded optic lobe (U3b): design

Written 2026-09-23 before the code of this unit. U3b has three parts, released in order: **the graded engine and the
flyvis ensemble** (this document's first half, 0.04.000); **the transfer onto the MaleCNS optic lobes** (0.05.000);
**the coupling to the spiking model and the stabilisers** (0.06.000). Each part adds its requirements to
`requirements.md` before its code.

## Why a graded optic lobe

Most neurons of the lamina and medulla do not fire spikes: they signal with graded changes of membrane potential and
release transmitter continuously. A whole-CNS model in which they spike does not carry vision past the lamina (the
flyverse project's audits, dossier 01 section 2.14 in the plan). The published model of the fly visual system that
predicts neural activity from the connectome, flyvis (Lappalainen et al., *Nature* 2024,
doi:10.1038/s41586-024-07939-3; code MIT), treats every visual neuron as a passive graded point neuron, and was
trained end to end on optic flow; ON and OFF pathways and T4/T5 direction selectivity emerge in it and match
measurements across 26 studies. flycns uses the same dynamics and flyvis's trained numbers.

## The dynamics, as flyvis 1.2.0 computes them

`PPNeuronIGRSynapses` (passive point neurons, instantaneous graded release):

$$\tau_i^\text{eff}\,\frac{dV_i}{dt} = -V_i + b_i + \sum_j w_{ij}\,\max(V_j, 0) + x_i(t), \qquad
\tau_i^\text{eff} = \max(\tau_i, \Delta t),$$

integrated by forward Euler, $V \leftarrow V + \Delta t\,\dot V$. $b_i$ is the resting potential of the neuron's
type, $\tau_i$ its time constant, $x_i$ the input (the stimulus intensity, added to the photoreceptors R1 to R8 of the
neuron's column, the same value for all eight), and

$$w_{ij} = \sigma_{t_j t_i}\; N_{t_j t_i}(\Delta u, \Delta v)\; \alpha_{t_j t_i},$$

with $\sigma$ the sign of the pair of types (fixed, from the literature), $N$ the mean synapse count of the pair at
the columnar offset between the two neurons (fixed, from the FIB-25 and FIB-19 reconstructions), and $\alpha$ the
unitary strength of the pair of types (trained, non-negative). The state starts at the resting potentials and is
brought to its steady state by two seconds of uniform grey (intensity 0.5). flyvis trains at $\Delta t$ = 1/50 s
and evaluates at 1/200 s.

**The ensemble, measured.** 50 networks, each with 65 resting potentials, 65 time constants and 604 strengths (734
trained numbers); signs (376 excitatory, 228 inhibitory pairs) and the 2,355 mean counts are identical in all 50.
73.7% of the trained time constants lie below the training step, where they acted as 20 ms; the slowest types are C2
(median 239 ms), Tm2 (136 ms) and C3 (67 ms). Resting potentials range from -1.41 to 2.63 (median 0.53). The trained
strength times the pair's mean count has a median of 0.036 (10th to 90th percentile 0 to 0.29); 3% of strengths sit
at zero. The same pair's strength varies widely across networks (median coefficient of variation 1.20), so results
are reported over the ensemble, never from one network alone.

## Two engines, and flyvis itself as the oracle

| Engine | Arithmetic | Input to each neuron |
|---|---|---|
| `GradedReference` (NumPy) | float64 | `bincount` over the connections, by target, in a fixed order |
| `GradedTorch` (PyTorch, CUDA or CPU) | float32 | `index_add_` over the connections |

The oracle is flyvis running its own network: `scripts/extract_flyvis_ensemble.py`, run in a separate environment
with flyvis 1.2.0, writes network 000 in full (45,669 neurons, 1,513,231 connections with their weights, the input
index of the 721 columns) and what flyvis computes on it for a fixed stimulus (two seconds of grey to the steady
state; then a full-field flash from 0.5 to 1.0 between 100 and 300 ms; then an ON edge sweeping the lattice at one
column per 20 ms from 400 ms), every neuron at every 5 ms step. The engines must reproduce that recording.

## The ensemble as data

The extraction also writes the 50 networks' parameters in flyvis's own order (types, pairs of types and count groups
exactly as flyvis's parameter objects list them, checked against the values the network uses) with the SHA-256 of
every source checkpoint and flyvis's MIT notice. The 225 KB directory ships inside the package
(`flycns/data/flyvis-1.2.0-ensemble/`), so the transfer onto MaleCNS needs neither flyvis nor the checkpoints.

## Part 2: the transfer onto the MaleCNS optic lobes (0.05.000)

**What becomes graded.** Every optic-lobe intrinsic neuron of MaleCNS, every photoreceptor, and every neuron of a
type flyvis models wherever the release files it (TmY14 is filed as a visual projection neuron): 95,925 neurons and
the 9.07 million connections among them. Everything else stays spiking and is coupled in part 3.

**Classes and mapping.** 49 flyvis types exist in MaleCNS under the same name. Photoreceptors pool into R1-R6 (the
release does not tell R1 from R6), R7 and R8 (their pale, yellow and dorsal-rim subtypes are one flyvis type each);
TmY9a and TmY9b stand for flyvis's TmY9. flyvis's Am, Mi3, Mi11, Mi12 and Tm28 have no MaleCNS counterpart by name.
MaleCNS's Am1 is not flyvis's lamina amacrine: it receives no R1-R6 synapse where flyvis's Am receives 227 per column,
so it is not mapped.

**Neurons** of a mapped class take the mean resting potential and time constant of their flyvis types; the other
classes take flyvis's initial resting potential (0.5) and the network's median trained time constant.

**Connections of a mapped pair keep flyvis's trained strength per synapse, capped per neuron.** The two datasets count
synapses on one scale: over the 551 pairs both have, the median ratio of MaleCNS's synapses per target to flyvis's,
weighted by flyvis's drive, is 1.08 (Mi1 onto T4a: 68.4 against 68.0). Transferring the strength per synapse keeps
the release's own pair-by-pair differences, and three measured failures shaped the rest of the rule:

1. A per-pair renormalisation (every pair's mean drive made flyvis's) concentrates flyvis's whole drive on the few
   synapses of pairs the release barely has: one weight reached 1,528.
2. Trained strengths on the release's synapses as they are drive the network to infinity: TmY4 onto TmY4 is seven
   times denser in MaleCNS than in flyvis's column (a per-target drive of 2.5 against 0.37, a loop gain above one),
   and the network diverges within 100 ms of grey. flyvis's seven-column reconstructions under-count lateral wiring.
3. Capping each pair's average drive at flyvis's delays the divergence to 700 ms (T5d), through neurons that receive
   several times their class's average.

So each neuron's drive from each presynaptic class is capped at flyvis's: where a neuron receives ``c`` synapses from
a class for which flyvis's column has ``N``, and ``c > N``, those synapses are scaled by ``N / c``. On network 000 the
cap binds on 68% of the transferred synapses. Where MaleCNS gives fewer synapses, the connection keeps flyvis's
strength and is weaker, as the release has it. (Making every neuron's drive exactly flyvis's, scaling up as well as
down, was measured too: 47 of 50 networks then diverge.)

**CT1 compartments.** flyvis models CT1 as one medulla (M10) and one lobula (Lo1) compartment per column, electrically
separate; MaleCNS has one CT1 per side. As one unit, CT1 sums every column and carries the eye-wide average; T5 cells,
whose local inhibition CT1 provides, then lose most of their direction selectivity. Each CT1 connection therefore
moves to the compartment of its partner's column (the partner's own, or the column that feeds it most synapses),
of the kind flyvis pairs the partner's type with: 3,536 compartments, all 41,557 CT1 connections moved.

**Every other connection** (unmapped classes, or mapped pairs flyvis has no connection for; 69% of the synapses)
takes flyvis's initialisation scale: each neuron receives from each such class a total of 0.01 x 2 (flyvis's strength
scale times its median number of columnar offsets per pair), spread over that class's synapses, with the sign of the
presynaptic transmitter.

**Stand-in photoreceptors.** 945 of the 1,771 columns have no reconstructed R1-R6 terminal, 585 no R7 and 450 no R8.
Where a column's photoreceptors deliver fewer synapses onto one of its columnar neurons than complete columns do
(six R1-R6, one R7, one R8; medians measured on those columns, e.g. R1-R6 onto L1 211.5 and onto L2 221), one flagged
stand-in per column and group supplies the difference and sees the column's light: 1,712 for R1-R6, 1,506 for R7,
1,655 for R8. Light reaches 10,768 photoreceptors, 5,895 of them real.

**Positions of neurons the release gives no column.** T4, T5, Tm3, T2 and TmY cells have no column in the
annotations. Their viewing direction is the synapse-weighted mean direction of their presynaptic partners' columns,
iterated so second-order cells inherit it; their home column (for the CT1 split) is the column that feeds them most.

**The eyes.** Measuring direction selectivity through the U2 eye model exposed an error in it: the lattice's vertical
was one 60-degree step off (``docs/models/01_eyes.md``). The eye model now sets the vertical by the dorsal rim; the
T4 preferred directions, measured afterwards, then match the known ones.

**Direction selectivity** is measured with flyvis's own measures (peak rectified response while the edge crosses the
neuron; DSI as the vector sum over 12 directions against the larger polarity's sum; preferred direction the vector
sum's angle), which reproduce flyvis's numbers on its lattice. On MaleCNS, full-field ON and OFF edges at flyvis's
three fastest speeds cross each modelled eye; every T4 and T5 cell within 15 degrees of the eye's centre is measured.

**Measured on all 50 networks** (`scripts/measure_optic_lobe.py`, `docs/models/04_optic_lobe.md`): where flyvis's
own T4 subtype is selective with the known direction (91 cases), the transferred one keeps it within 45 degrees on
both eyes in 78 (86%, median error 8 to 11 degrees); T5 keeps it in 15 of 39 (38%), with small selectivity. 41 of 50
transferred networks stay bounded over 20 s of grey (flyvis's own lattice: 49).

## Part 3: the whole CNS coupled, engines E1 to E4 (0.06.000)

**The split.** The graded units of part 2 (95,925 optic-lobe neurons plus stand-ins and CT1 compartments) run every
5 ms, flyvis's step, validated against flyvis; the other 70,775 neurons of the release run as the published LIF
every 0.1 ms. 13.76 million connections join spiking neurons, 1.94 million run from graded to spiking neurons (7.44
million synapses) and 0.83 million from spiking to graded neurons (2.75 million synapses).

**The bridge.** A graded neuron releases continuously; its release above the grey steady state,
``D = max(V, 0) - max(V_grey, 0)``, acts on each spiking target as a spike train of rate ``beta x D``: each of the
connection's ``n`` synapses adds ``sign x w_syn`` per spike, so ``g`` gains ``beta D sign n w_syn`` per second. It is
added in the synapses slot of every LIF step, for neurons that are not refractory, recomputed every graded step. At
grey the spiking CNS receives nothing, as the published model fires nothing without input. ``beta``, spikes per
second per unit of release, is the free parameter every hybrid in the survey has; 100 by default, and the product
reports it. Measured over 25 to 400: the visual projection neurons' spikes during a flash scale nearly in proportion
(8,382 to 133,641), the nerve-cord motor neurons' far less (2,624 to 5,036); light reaches the motor neurons at every
value.

**Feedback.** A spiking neuron acts on a graded target as release of ``r / beta``, ``r`` its rate filtered with the
published synaptic time constant (5 ms); each graded target receives from each class of spiking presynaptic neurons
flyvis's initialisation drive, ``0.01 x 2``, spread over the class's synapses, with the transmitter's sign.

**E1, the published model everywhere.** Light enters as Poisson input to every photoreceptor with a column, at 300 Hz
times the column's intensity (grey gives the paper's 150 Hz), with no refractory period, as the paper's activated
neurons. It fails exactly as the flyverse project reported: under a flash the photoreceptors fire 515,213 spikes and
no other neuron fires one. The mechanism is the sign: photoreceptors are histaminergic, inhibitory, and inhibiting a
spiking neuron that is silent does nothing, while in the fly the lamina's graded neurons signal by being
hyperpolarised.

**E2** is parts 1 and 2 with the bridge and feedback: under a flash the whole CNS answers (41,019 spikes in visual
projection neurons, 69,378 in the central brain, 5,455 in descending and 3,500 in nerve-cord motor neurons), silent
at grey.

**E3** runs flyvis's own network per eye on its lattice and gives each MaleCNS neuron of a flyvis type the activity
of the lattice cell of its type that looks where it looks (72,414 units mapped, 5,417 outside the lattice); the
lattice's layout in flyvis's stimulus frame is measured from flyvis's own moving edges, and flyvis's frame is the
mirror of the eyes' (its T4a prefers front-to-back motion in the eyes' frame, as it must). Under the flash it carries
activity to the central brain as E2 does (45,558 spikes in visual projection neurons, 3,584 in motor neurons).

**E4** is E2 with flyverse's stabilisers, in a stated order: each connection capped at 60 synapse-equivalents,
same-type connections scaled by 0.1, every neuron whose input then exceeds 5,000 synapse-equivalents scaled down to
it, and spike-frequency adaptation of 1.5 mV per spike decaying with 200 ms. On the published model under the strong
gustatory drive, ten seeds' 500 ms totals spread from 271,834 to 519,691 spikes (two states); with the stabilisers
from 165,430 to 168,527 (one state).
