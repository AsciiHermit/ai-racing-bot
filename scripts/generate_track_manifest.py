"""Generate and check in Phase 2's track manifest
(data/tracks/manifest.json): a reproducible train set plus a 30-track
held-out set (15 interpolation + 15 extrapolation), per wayfinder tickets
#11/#12/#13. Tracks are stored as specs (generator + seed + params), not
raw control points -- see ars.track.manifest.build_track() to regenerate
one, or ars.track.manifest.build_manifest() to see how this was built.

Run: python scripts/generate_track_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path

from ars.track.manifest import build_manifest


def main() -> None:
    manifest = build_manifest()  # defaults: 80 train, 15 interpolation, 15 extrapolation

    out_path = Path(__file__).resolve().parent.parent / "data" / "tracks" / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    diversity = manifest["diversity_check"]
    print(f"wrote {out_path}")
    print(
        f"train={len(manifest['train'])} "
        f"interpolation={len(manifest['held_out']['interpolation'])} "
        f"extrapolation={len(manifest['held_out']['extrapolation'])}"
    )
    print(f"diversity check passed: {diversity['passed']} (min_pairwise_distance={diversity['min_pairwise_distance']:.4f})")
    if not diversity["passed"]:
        raise SystemExit("diversity check failed -- do not check in this manifest")


if __name__ == "__main__":
    main()
