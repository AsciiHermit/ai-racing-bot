"""Precompute polylines (centerline, inner/outer boundary) from a Track,
for drawing. Pure geometry -- no pygame dependency -- so it's testable
headless and reusable if a different renderer is added later.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ars.core.interfaces import Track


@dataclass
class TrackPolylines:
    centerline: list[tuple[float, float]]
    inner_boundary: list[tuple[float, float]]
    outer_boundary: list[tuple[float, float]]
    bounds: tuple[float, float, float, float]  # min_x, max_x, min_y, max_y


def build_track_polylines(track: Track, step: float = 1.0) -> TrackPolylines:
    centerline: list[tuple[float, float]] = []
    inner: list[tuple[float, float]] = []
    outer: list[tuple[float, float]] = []

    s = 0.0
    while s < track.length:
        sample = track.sample_at_s(s)
        nx = -math.sin(sample.heading)
        ny = math.cos(sample.heading)
        half_w = sample.width / 2.0
        centerline.append((sample.centerline_x, sample.centerline_y))
        inner.append((sample.centerline_x - nx * half_w, sample.centerline_y - ny * half_w))
        outer.append((sample.centerline_x + nx * half_w, sample.centerline_y + ny * half_w))
        s += step

    # close the loops
    centerline.append(centerline[0])
    inner.append(inner[0])
    outer.append(outer[0])

    xs = [p[0] for p in outer] + [p[0] for p in inner]
    ys = [p[1] for p in outer] + [p[1] for p in inner]
    bounds = (min(xs), max(xs), min(ys), max(ys))

    return TrackPolylines(centerline=centerline, inner_boundary=inner, outer_boundary=outer, bounds=bounds)
