"""flycns's motion measures on inputs whose answer is known: a cosine tuning curve, and an edge of known speed."""

from __future__ import annotations

import numpy as np
import pytest

from flycns.motion import (
    COLUMN_DEG,
    LED_DEG,
    direction_selectivity,
    edge_on_eye,
    erf,
    flyvis_time_window,
    local_frame,
    peaks_on_eye,
)


def test_measures_on_known_tuning_and_a_moving_edge():
    # a cosine-tuned neuron preferring 60 degrees for ON edges, half as responsive and untuned for OFF edges:
    # sum_theta (1 + cos(theta - 60)) e^(i theta) = 6 e^(i 60) over 12 directions, sum_theta |r| = 12
    angles = np.arange(0, 360, 30, dtype=float)
    rows, a, pol, speed = [], [], [], []
    for s in (13.0, 19.0):
        for theta in angles:
            for p in (0, 1):
                rows.append([1 + np.cos(np.deg2rad(theta - 60)) if p == 1 else 0.5])
                a.append(theta)
                pol.append(p)
                speed.append(s)
    sel = direction_selectivity(np.array(rows), np.array(a), np.array(pol, dtype=float), np.array(speed))
    assert sel.dsi[0, 1] == pytest.approx(0.5)                   # ON: 6 / 12
    assert sel.dsi[0, 0] == pytest.approx(0.0, abs=1e-12)        # OFF: untuned
    assert np.degrees(sel.preferred_rad[0, 1]) == pytest.approx(60.0)

    # flyvis's window: from the start of the sweep to when the edge has passed 22.5 degrees beyond the centre
    t0, t1 = flyvis_time_window(19.0, -22.5 / COLUMN_DEG, 22.5 / COLUMN_DEG)
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx((45.0 + LED_DEG) / COLUMN_DEG / 19.0)

    assert erf(np.array([0.0, 0.5, 1.0, -2.0])) == pytest.approx([0.0, 0.5204999, 0.8427008, -0.9953223], abs=2e-7)

    # an ON edge moving upward (90 degrees) at 60 degrees per second across a column grid
    x, y = np.meshgrid(np.arange(-20, 21, 5.0), np.arange(-20, 21, 5.0))
    x, y = x.ravel(), y.ravel()
    dt = 1 / 200
    sweep = edge_on_eye(x, y, 90.0, 1, 60.0, dt, extent_deg=30.0, t_pre_s=0.2, t_post_s=0.2)
    assert np.all(sweep.intensity[:int(0.2 / dt)] == 0.5)        # grey before the edge
    half = np.argmax(sweep.intensity >= 0.75, axis=0)            # when each column is half covered
    moving = half[(y == -10) | (y == 10)]
    # columns 20 degrees apart along the motion are crossed 20 / 60 s apart; columns across the motion together
    assert (half[y == 10].mean() - half[y == -10].mean()) * dt == pytest.approx(20 / 60, abs=2 * dt)
    assert np.ptp(half[y == 0]) == 0 and len(moving) > 0
    assert sweep.intensity.max() <= 1.0 and sweep.intensity.min() >= 0.5

    # a neuron whose response peaks when the edge crosses it is read inside its own window only
    responses = np.zeros((len(sweep.intensity), 2))
    k = int(0.2 / dt) + int((10 + 30) / 60 / dt)                 # the edge front reaches y = 10
    responses[k, 0] = 3.0
    responses[5, 1] = 9.0                                        # before the edge: outside every window
    peaks = peaks_on_eye(responses, sweep, np.array([10.0, 10.0]), dt)
    assert peaks.tolist() == [3.0, 0.0]


def test_local_frame_is_distance_preserving_at_the_centre():
    x, y = local_frame(np.array([80.0, 70.0, 70.0]), np.array([0.0, 10.0, 0.0]), 70.0, 0.0)
    assert x == pytest.approx([10.0, 0.0, 0.0], abs=1e-9)
    assert y == pytest.approx([0.0, 10.0, 0.0], abs=1e-9)
