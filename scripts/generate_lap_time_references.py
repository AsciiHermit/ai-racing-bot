"""Phase 4: run the minimum-curvature solver against every track in the
checked-in manifest (train + held-out) and store the reference optimal
lap time alongside each track's spec in data/tracks/manifest.json, so
Phase 6 can look it up directly without re-solving.

Run: python scripts/generate_lap_time_references.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ars.solver.min_curvature import estimate_lap_time
from ars.track.manifest import build_track, load_manifest


def main() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "data" / "tracks" / "manifest.json"
    manifest = load_manifest(manifest_path)

    all_specs = (
        manifest["train"] + manifest["held_out"]["interpolation"] + manifest["held_out"]["extrapolation"]
    )
    for spec in all_specs:
        track = build_track(spec)
        lap_time = estimate_lap_time(track)
        if not math.isfinite(lap_time) or lap_time <= 0.0:
            raise SystemExit(f"solver produced an invalid lap time for {spec['id']}: {lap_time}")
        spec["reference_lap_time_s"] = lap_time

    manifest_path.write_text(json.dumps(manifest, indent=2))

    lap_times = [spec["reference_lap_time_s"] for spec in all_specs]
    print(f"wrote reference lap times for {len(all_specs)} tracks to {manifest_path}")
    print(f"min={min(lap_times):.2f}s max={max(lap_times):.2f}s mean={sum(lap_times)/len(lap_times):.2f}s")


if __name__ == "__main__":
    main()
