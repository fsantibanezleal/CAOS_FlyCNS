"""Direction selectivity: moving edges, the peak responses they evoke, and flyvis's two measures of them.

The measures are flyvis's (``flyvis.analysis.moving_bar_responses``, Lappalainen et al., Nature 2024), written out so
flycns can apply them to any network, and held to flyvis's own numbers by ``tests/test_motion_parity.py``:

- the **peak response** of a neuron to an edge is the largest rectified voltage, ``max(V, 0)``, while the edge
  crosses the neuron's receptive field (NaN frames, flyvis's padding after short stimuli, are skipped);
- the **direction selectivity index** of a neuron, for one edge polarity, is the length of the vector sum over
  directions of its peak responses, ``|sum_theta r(theta) e^(i theta)|``, divided by the larger of the two
  polarities' sums ``sum_theta |r(theta)|``, averaged over speeds;
- its **preferred direction** is the angle of the vector sum over directions and speeds.

``edge_on_eye`` renders a full-field moving edge on the modelled eye: straight in a local tangent frame around a
reference direction, blurred by the ommatidium's Gaussian acceptance, moving at a stated angular speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optic_lobe import azimuth_elevation

COLUMN_DEG = 5.8          # flyvis's column spacing, the unit of its speeds
LED_DEG = 2.25            # flyvis's stimulus pixel
GREY = 0.5

#: Known preferred directions (Maisak et al., Nature 500:212-216, 2013) in the anatomical frame used here:
#: 0 = front to back, 90 = upward, 180 = back to front, 270 = downward. T4 respond to ON edges, T5 to OFF edges.
KNOWN_PREFERRED_DEG = {"T4a": 0.0, "T4b": 180.0, "T4c": 90.0, "T4d": 270.0,
                       "T5a": 0.0, "T5b": 180.0, "T5c": 90.0, "T5d": 270.0}
#: The same directions in flyvis's stimulus frame (``angular_distance_to_known``), whose 0 is back to front.
FLYVIS_KNOWN_PREFERRED_DEG = {"T4a": 180.0, "T4b": 0.0, "T4c": 90.0, "T4d": 270.0,
                              "T5a": 180.0, "T5b": 0.0, "T5c": 90.0, "T5d": 270.0}


def flyvis_time_window(speed: float, from_column: float, to_column: float, start: float = -10,
                       end: float = 11) -> tuple[float, float]:
    """flyvis's ``time_window``: when the edge, starting ``start`` LEDs from the central column and moving at
    ``speed`` columns per second, crosses from ``from_column`` to ``to_column`` (plus one LED)."""
    start_in_columns = start * LED_DEG / COLUMN_DEG
    to_column = to_column + LED_DEG / COLUMN_DEG
    t_start = (abs(start_in_columns) - abs(from_column)) / speed
    return t_start, t_start + (to_column - from_column) / speed


def flyvis_peaks(responses: np.ndarray, time_s: np.ndarray, speed: np.ndarray, offsets=(-10, 11)) -> np.ndarray:
    """Peak rectified responses (samples x neurons) of flyvis's central neurons, with flyvis's default window."""
    from_degree, to_degree = offsets[0] * LED_DEG, (offsets[1] - 1) * LED_DEG
    peaks = np.zeros((responses.shape[0], responses.shape[2]))
    for i in range(responses.shape[0]):
        t0, t1 = flyvis_time_window(float(speed[i]), from_degree / COLUMN_DEG, to_degree / COLUMN_DEG, *offsets)
        window = (time_s >= t0) & (time_s <= t1)
        peaks[i] = np.nan_to_num(np.nanmax(np.maximum(responses[i][window], 0.0), axis=0), nan=0.0)
    return peaks


@dataclass
class Selectivity:
    dsi: np.ndarray                 # (neurons, 2): OFF, ON
    preferred_rad: np.ndarray       # (neurons, 2)


def direction_selectivity(peaks: np.ndarray, angle_deg: np.ndarray, intensity: np.ndarray,
                          speed: np.ndarray) -> Selectivity:
    """flyvis's DSI and preferred direction from peak responses (samples x neurons)."""
    angles = np.unique(angle_deg)
    speeds = np.unique(speed)
    n = peaks.shape[1]
    vec = np.zeros((2, len(speeds), n), dtype=complex)
    norm = np.zeros((2, len(speeds), n))
    for k, pol in enumerate((0, 1)):
        for j, s in enumerate(speeds):
            for a in angles:
                row = np.flatnonzero((angle_deg == a) & (intensity == pol) & (speed == s))
                if len(row) != 1:
                    raise ValueError(f"expected one sample for angle {a}, intensity {pol}, speed {s}")
                r = peaks[row[0]]
                vec[k, j] += r * np.exp(1j * np.deg2rad(a))
                norm[k, j] += np.abs(r)
    normalisation = norm.max(axis=0)                                  # the larger polarity, per speed
    dsi = (np.abs(vec) / (normalisation[None] + 1e-15)).mean(axis=1)  # averaged over speeds
    preferred = np.angle(vec.sum(axis=1))
    return Selectivity(dsi=dsi.T, preferred_rad=preferred.T)


def erf(z: np.ndarray) -> np.ndarray:
    """The error function to 1.5e-7 (Abramowitz and Stegun 7.1.26), vectorised."""
    z = np.asarray(z, dtype=np.float64)
    s = np.sign(z)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t         * np.exp(-a * a)
    return s * y


def angular_distance_deg(a_deg, b_deg) -> np.ndarray:
    d = np.abs((np.asarray(a_deg) - np.asarray(b_deg) + 180.0) % 360.0 - 180.0)
    return d


# ------------------------------------------------------------------------------------------------ edges on an eye

def local_frame(azimuth_deg: np.ndarray, elevation_deg: np.ndarray, centre_az: float, centre_el: float):
    """Tangent-plane coordinates (degrees) around (centre_az, centre_el): x toward the back, y upward."""
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    a0, e0 = np.deg2rad(centre_az), np.deg2rad(centre_el)
    # gnomonic-free, distance-preserving at the centre: an azimuthal-equidistant projection
    cos_c = np.sin(e0) * np.sin(el) + np.cos(e0) * np.cos(el) * np.cos(az - a0)
    c = np.arccos(np.clip(cos_c, -1.0, 1.0))
    k = np.where(c > 1e-12, c / np.sin(np.maximum(c, 1e-12)), 1.0)
    x = k * np.cos(el) * np.sin(az - a0)
    y = k * (np.cos(e0) * np.sin(el) - np.sin(e0) * np.cos(el) * np.cos(az - a0))
    return np.rad2deg(x), np.rad2deg(y)


@dataclass
class EdgeSweep:
    """A full-field edge crossing an eye: per-frame column intensities and when it crosses each column."""

    intensity: np.ndarray           # (frames, columns)
    position_deg: np.ndarray        # (columns,) each column's coordinate along the direction of motion
    front_start_deg: float
    speed_deg_s: float
    t_pre_s: float


def edge_on_eye(x_deg: np.ndarray, y_deg: np.ndarray, angle_deg: float, polarity: int, speed_deg_s: float,
                dt_s: float, extent_deg: float, acceptance_fwhm_deg: float = 8.23, t_pre_s: float = 1.0,
                t_post_s: float = 1.0) -> EdgeSweep:
    """An edge moving along ``angle_deg`` (0: +x, 90: +y) at ``speed_deg_s``; the swept side becomes 1 (ON) or 0
    (OFF) on grey. It starts ``extent_deg`` behind the centre and stops ``extent_deg`` past it. Each column sees the
    edge through a Gaussian acceptance of the given full width at half maximum."""
    theta = np.deg2rad(angle_deg)
    p = x_deg * np.cos(theta) + y_deg * np.sin(theta)
    sigma = acceptance_fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    n_pre, n_move, n_post = int(round(t_pre_s / dt_s)), int(round(2 * extent_deg / speed_deg_s / dt_s)), \
        int(round(t_post_s / dt_s))
    front = np.concatenate([np.full(n_pre, -extent_deg - 5 * sigma),
                            -extent_deg + speed_deg_s * dt_s * np.arange(n_move),
                            np.full(n_post, extent_deg)])
    target = 1.0 if polarity == 1 else 0.0
    covered = 0.5 * (1.0 + erf((front[:, None] - p[None, :]) / (sigma * np.sqrt(2.0))))   # swept acceptance
    covered[:n_pre] = 0.0
    intensity = GREY + (target - GREY) * covered
    return EdgeSweep(intensity=intensity.astype(np.float32), position_deg=p, front_start_deg=-extent_deg,
                     speed_deg_s=speed_deg_s, t_pre_s=t_pre_s)


def peaks_on_eye(responses: np.ndarray, sweep: EdgeSweep, positions_deg: np.ndarray, dt_s: float,
                 half_window_deg: float = 10 * LED_DEG) -> np.ndarray:
    """Peak rectified response of each recorded neuron (frames x neurons) while the edge front is within
    ``half_window_deg`` of the neuron's own position (flyvis's window, centred on each neuron)."""
    t = np.arange(responses.shape[0]) * dt_s - sweep.t_pre_s
    front = sweep.front_start_deg + sweep.speed_deg_s * t
    peaks = np.zeros(responses.shape[1])
    for j, p in enumerate(positions_deg):
        window = (t >= 0) & (front >= p - half_window_deg) & (front <= p + half_window_deg + LED_DEG)
        peaks[j] = float(np.max(np.maximum(responses[window, j], 0.0))) if window.any() else 0.0
    return peaks


# ------------------------------------------------------------------------------ the experiment on the MaleCNS eyes

T4_T5 = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
SPEEDS_COLUMNS = (13.0, 19.0, 25.0)            # flyvis's three fastest speeds, in its 5.8-degree columns per second
ANGLES_DEG = tuple(range(0, 360, 30))


@dataclass
class EyeSelectivity:
    """Per recorded neuron: where it is, and its selectivity for ON and OFF edges."""

    side: str
    centre_deg: tuple[float, float]
    unit: np.ndarray                # optic-lobe unit index
    unit_type: np.ndarray           # class name
    x_deg: np.ndarray
    y_deg: np.ndarray
    dsi: np.ndarray                 # (neurons, 2): OFF, ON
    preferred_deg: np.ndarray       # (neurons, 2), anatomical frame: 0 front to back, 90 up

    def summary(self) -> dict:
        """Per T4/T5 subtype, with its own polarity (ON for T4, OFF for T5): the neurons, their median DSI, the
        fraction whose preferred direction is within 45 degrees of the known one, and the circular mean direction
        weighted by DSI."""
        out = {}
        for t in T4_T5:
            m = self.unit_type == t
            if not m.any():
                continue
            k = 1 if t.startswith("T4") else 0
            pd, dsi = self.preferred_deg[m, k], self.dsi[m, k]
            mean = float(np.degrees(np.angle(np.sum(dsi * np.exp(1j * np.deg2rad(pd))))) % 360)
            out[t] = {"neurons": int(m.sum()), "median_dsi": float(np.median(dsi)),
                      "within_45_of_known": float(np.mean(angular_distance_deg(pd, KNOWN_PREFERRED_DEG[t]) <= 45)),
                      "mean_preferred_deg": mean,
                      "mean_error_deg": float(angular_distance_deg(mean, KNOWN_PREFERRED_DEG[t]))}
        return out


def eye_centre(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> tuple[float, float]:
    """The middle of an eye's equatorial band: the median azimuth of the columns within 10 degrees of the equator."""
    band = np.abs(elevation_deg) < 10
    return float(np.median(azimuth_deg[band])), 0.0


def selectivity_on_eye(engine, lobe, eye, steady, unit_dirs, unit_side, radius_deg: float = 15.0,
                       extent_deg: float = 45.0,
                       dt_s: float = 1 / 200, t_pre_s: float = 1.0, t_post_s: float = 0.5,
                       speeds_columns=SPEEDS_COLUMNS, angles_deg=ANGLES_DEG) -> EyeSelectivity:
    """Full-field ON and OFF edges in 12 directions and three speeds across one eye (the other eye sees grey), from
    the grey steady state; the T4 and T5 cells of that side whose viewing direction (``unit_dirs``, from
    ``optic_lobe.unit_directions``) lies within ``radius_deg`` of the eye's centre are recorded, and flyvis's
    measures applied to each of them. ``unit_side`` names each unit's side."""
    centre = eye_centre(eye.azimuth_deg, eye.elevation_deg)
    x_col, y_col = local_frame(eye.azimuth_deg, eye.elevation_deg, *centre)
    names = np.array(lobe.classes, dtype=object)[lobe.unit_class]
    units = np.flatnonzero(np.isin(names, T4_T5))
    # T4 and T5 have no column in the release: their position is the weighted mean of their inputs' columns
    az, el = azimuth_elevation(unit_dirs[units], eye.side)
    ux, uy = local_frame(az, el, *centre)
    on_eye = np.isin(lobe.column[units], eye.column_index) | (unit_side[units] == eye.side)
    near = on_eye & (np.hypot(ux, uy) <= radius_deg)
    units, ux, uy = units[near], ux[near], uy[near]

    peaks, s_angle, s_int, s_speed = [], [], [], []
    for speed in speeds_columns:
        sweeps, labels = [], []
        for angle in angles_deg:
            for polarity in (0, 1):
                sweep = edge_on_eye(x_col, y_col, angle, polarity, speed * COLUMN_DEG, dt_s, extent_deg,
                                    t_pre_s=t_pre_s, t_post_s=t_post_s)
                full = np.full((sweep.intensity.shape[0], lobe.network.n_columns), GREY, dtype=np.float32)
                full[:, eye.column_index] = sweep.intensity
                sweeps.append((sweep, full))
                labels.append((angle, polarity))
        responses = engine.run_batch(np.stack([f for _, f in sweeps]), steady, units)
        for (sweep, _), (angle, polarity), resp in zip(sweeps, labels, responses, strict=True):
            theta = np.deg2rad(angle)
            positions = ux * np.cos(theta) + uy * np.sin(theta)
            peaks.append(peaks_on_eye(resp, sweep, positions, dt_s))
            s_angle.append(angle)
            s_int.append(polarity)
            s_speed.append(speed)
    sel = direction_selectivity(np.array(peaks), np.array(s_angle, dtype=float), np.array(s_int, dtype=float),
                                np.array(s_speed, dtype=float))
    return EyeSelectivity(side=eye.side, centre_deg=centre, unit=units, unit_type=names[units], x_deg=ux, y_deg=uy,
                          dsi=sel.dsi, preferred_deg=np.degrees(sel.preferred_rad) % 360)
