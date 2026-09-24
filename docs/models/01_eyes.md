# The two compound eyes

A fly sees through two compound eyes of several hundred ommatidia each; every ommatidium looks in one direction and
feeds one column of the optic lobe. `flycns.eyes` builds both eyes over the columns the release actually mapped, so
that light entering the model eye reaches the real neurons of the right column.

## What is real, what is derived, what is modelled

| Part | Where it comes from | Status |
|---|---|---|
| The columns of each eye: 879 left, 892 right, with their two hexagonal coordinates | the MaleCNS annotations | real |
| The photoreceptors and lamina neurons of each column; pale, yellow and dorsal-rim columns | compiled from the release's connectivity | real, derived |
| How the lattice lies in the animal: which hex axis runs dorsal, which anterior, which offsets are neighbours | measured on the release (below) | real, derived |
| Each column's viewing direction | a model, scaled to the eye's measured extent | modelled, stated |
| Each ommatidium's acceptance | a Gaussian with the measured dark-adapted width | modelled from a measurement |

## Measuring the lattice on the release

The release labels every medulla synapse with its column (`ME_R_col_12_29`) and every synapse with its neuropil. One
pass over the 13 GB synapse table gives (i) the 3D centre of every medulla column and (ii) the centroids of landmark
neuropils. The landmarks give the body axes: **left** from the right medulla to the left one; **anterior** from the
mushroom-body calyces to the antennal lobes; **dorsal** from the gnathal ganglia to the protocerebral bridge, each
made orthogonal to the previous ones. Measured on MaleCNS v1.0 (release frame): left = (0.999, -0.030, -0.019),
anterior = (0.004, 0.619, -0.785), dorsal = (-0.035, -0.785, -0.619), a right-handed frame.

A least-squares fit of the column centres,

$$\mathbf{c}(h_1, h_2) \approx \mathbf{c}_0 + h_1\,\mathbf{e}_1 + h_2\,\mathbf{e}_2,$$

gives the 3D step $\mathbf{e}_k$ of each hex axis. Measured (right eye, micrometres): $\mathbf{e}_1$ has anterior
component $-5.05$ and dorsal $+0.64$ (it runs posterior), $\mathbf{e}_2$ has anterior $-0.54$ and dorsal $+6.20$ (it
runs dorsal); the left eye matches. The six nearest columns of each column, tallied, give the neighbour offsets
$(\pm1, 0)$, $(0, \pm1)$ and $\pm(1, 1)$ in both eyes.

## From the medulla to the eye

**The chiasm.** Between the lamina and the medulla, the first optic chiasm crosses the fibres horizontally: the
anterior-posterior order of columns is mirrored, the dorsal-ventral order kept. So the hex axis that runs posterior in
the medulla belongs to ommatidia that look further **forward**.

**The ideal lattice.** In the eye the lattice is taken regular. The dorsal axis $h_2$ is the vertical neighbour
direction; because $(1, 1)$ is a neighbour offset, the other axis sits 120 degrees from it, so the sum of the two is
also one step:

$$\mathbf{p}(h_1, h_2) = h_1 \begin{pmatrix} \cos 30^\circ \\ -\sin 30^\circ \end{pmatrix} + h_2 \begin{pmatrix} 0 \\ 1 \end{pmatrix}
\quad\text{(forward, up), in steps.}$$

**Placement on the sphere.** The equator is the median row (as many columns above as below; an assumption). The
columns within half a step of the equator span, in the measurements of Zhao et al. (*Nature* 646:135-142, 2025,
doi:[10.1038/s41586-025-09276-5](https://doi.org/10.1038/s41586-025-09276-5)), from 10 degrees into the opposite
hemisphere in front to 155 degrees behind. That fixes the step angle:

$$\Delta\varphi = \frac{155^\circ - (-10^\circ)}{p_{\text{front}} - p_{\text{back}}}.$$

Each column is then placed at angular distance $\rho = \Delta\varphi\,\lVert \mathbf{p} - \mathbf{p}_c \rVert$ from the
eye's centre direction (azimuth 72.5 degrees, elevation 0) along the lattice bearing (an azimuthal-equidistant
placement). Radial distances from the centre are exact; tangential spacing at distance $\rho$ is compressed by
$\sin\rho / \rho$ (0.69 at the ends of the equator). Real eyes also vary their spacing across the eye, flatter in the
centre than at the edges (Zhao et al. 2025); the model's variation is a consequence of the placement, not a
measurement.

## The eyes this gives for MaleCNS v1.0

| Quantity | Left | Right |
|---|---|---|
| Columns | 879 | 892 |
| Implied inter-ommatidial angle $\Delta\varphi$ | 5.60 deg | 5.60 deg |
| Neighbour spacing, median (5th to 95th percentile) | 5.29 deg (4.08 to 5.60) | 5.28 deg (4.07 to 5.60) |
| Farthest look past the midline (off the equator) | 14.8 deg | 17.1 deg |
| Dorsal-rim columns, mean elevation | 40.4 deg | 34.4 deg |
| Pale and yellow columns, mean elevation | 10 to 12 deg | 12 to 16 deg |

The implied $\Delta\varphi$ of 5.6 degrees is close to the smallest measured inter-ommatidial angle of the
Drosophila eye, 4.5 degrees in its lateral part (Gonzalez-Bellido, Wardill and Juusola, *PNAS* 108:4224, 2011,
doi:[10.1073/pnas.1014438108](https://doi.org/10.1073/pnas.1014438108)). The dorsal-rim columns, identified only from
their photoreceptor subtypes, sit well above the colour columns in both eyes: an independent check that the vertical
orientation is right. Some dorsal-rim labels reach low elevations (a few columns in each eye), consistent with a small
number of misassigned R7d/R8d terminals in a sparsely reconstructed retina.

## Sampling a scene

A scene arrives as an equirectangular panorama in the body frame. Each ommatidium integrates it with a Gaussian
acceptance of full width at half maximum $\Delta\rho$, weighted by each pixel's solid angle and normalised:

$$L_i = \frac{\sum_k w_{ik} I_k}{\sum_k w_{ik}}, \qquad
w_{ik} = \exp\!\left(-\frac{\theta_{ik}^2}{2\sigma^2}\right)\cos\phi_k\,\Delta\lambda\,\Delta\phi,
\qquad \sigma = \frac{\Delta\rho}{2\sqrt{2\ln 2}},$$

with $\theta_{ik}$ the angle between ommatidium $i$'s direction and pixel $k$, $\phi_k$ the pixel's latitude, and
$\Delta\rho$ = 8.23 degrees, the measured dark-adapted half-width of R1-R6 (Gonzalez-Bellido et al. 2011). Weights
beyond three sigma are dropped. Ground truth (depth, object labels) is read under each ommatidium's central ray and,
where an acceptance-weighted value is wanted, through the same weights, so input and truth are sampled identically.

## Assumptions and limits

- Directions are modelled; the real lens map of this animal was not imaged. The extent comes from other flies (three
  females in Zhao et al.); the equator is the median row.
- One inter-ommatidial angle for the whole lattice; spacing then varies only through the placement.
- One acceptance width for all photoreceptors and all light levels (the dark-adapted value; light adaptation narrows
  it).
- Luminance only. Pale and yellow columns are recorded so a spectral model can be added; none is claimed here.
