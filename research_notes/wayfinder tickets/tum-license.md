# Ticket: Verify TUM `global_racetrajectory_optimization`'s license/maintenance status

- Issue: https://github.com/AsciiHermit/ai-racing-bot/issues/15
- Referenced from: `PROBLEM_DEFINITION.md`, `IMPLEMENTATION_PLAN.md:97`, `IMPLEMENTATION_PLAN.md:182`
- Investigation date: 2026-09-27
- Method: primary-source lookup against the live GitHub repo (web pages, GitHub REST API `api.github.com/repos/...`, and raw file contents via `raw.githubusercontent.com`). No `git`/`gh` commands were run against this project's own repo; no branches/commits/comments were created.

## Cited findings

### 1. Exact repository name and URL

The repo exists exactly as named, under the TUM Institute of Automotive Technology's GitHub org:

- **URL**: https://github.com/TUMFTM/global_racetrajectory_optimization
- **Full name (from GitHub API `full_name` field)**: `TUMFTM/global_racetrajectory_optimization`
- **API description field**: "This repository contains multiple approaches for generating global racetrajectories."
- **Created**: 2019-05-21 (`created_at` from `api.github.com/repos/TUMFTM/global_racetrajectory_optimization`)
- **Default branch**: `master`

There is no evidence the repo has been renamed — the name matches the docs' citation exactly, no redirect was encountered. I did *not* find a second, competing "global_racetrajectory_optimization" repo elsewhere; TUMFTM does host several related-but-distinct repos that are easy to confuse with this one and are worth distinguishing in case a future ticket cites the wrong one:
- `TUMFTM/trajectory_planning_helpers` — shared low-level math/helper functions, a dependency of the main repo, not itself the optimizer.
- `TUMFTM/racetrack-database` — track centerline/width CSVs for ~20 real tracks, a data source, not an optimizer.
- `TUMFTM/laptime-simulation` — quasi-steady-state lap-time simulator, a downstream consumer of a race line, not the line generator.

None of these three is "global_racetrajectory_optimization"; the repo cited in the project docs is the correct, unambiguous match.

### 2. License

Checked the LICENSE file directly (`raw.githubusercontent.com/TUMFTM/global_racetrajectory_optimization/master/LICENSE`) and cross-checked against the GitHub API's detected license metadata — both agree:

- **License**: GNU **Lesser** General Public License, version 3 (LGPL-3.0)
- API fields: `license.name = "GNU Lesser General Public License v3.0"`, `license.spdx_id = "LGPL-3.0"`
- The raw LICENSE file's body is the standard LGPLv3 text (FSF, "Version 3, 29 June 2007"), confirming it's the unmodified LGPL-3.0, not a custom/dual license.
- The README (`Readme.md` at repo root) has **no separate license/citation section** — no "Citation"/"cite"/"paper" attribution clause beyond a plain "References" section listing the team's academic papers (contact persons, no formal citation requirement text).

**Is it permissive enough to adapt/vendor?** LGPL-3.0 is copyleft, but of the weaker "library" kind: you can *use* the code (call it, import it, run it as a subprocess/library) from a differently-licensed project without that project inheriting LGPL obligations, and you can *modify and redistribute* the library itself as long as those modifications stay under LGPL-3.0 and the source is made available. The practical constraint for ai-racing-bot is:
- Treating it as an **external dependency** (installed, imported, called) — no license conflict with whatever license ai-racing-bot ends up with.
- **Vendoring/copying and modifying its source directly into this project's codebase** — the copied/modified files (and, per LGPL §4, any larger work if it's converted from a "library-usage" relationship into a "combined work") would need to remain available under LGPL-3.0 terms, with the LGPL notice retained and source made available to recipients.
- **This project currently has no LICENSE file at all** (confirmed via `Glob **/LICENSE*` at `D:\ai-racing-bot` — no matches). Until ai-racing-bot picks its own license, "compatible with LGPL" is moot for the project as a whole, but it matters immediately for *this specific dependency*: if the team vendors modified TUM code, that portion is bound by LGPL-3.0 regardless of what license the rest of the project later adopts, and the project will need a LICENSE file (or at least a NOTICE) that discloses that.

### 3. Maintenance status

Checked via the GitHub API and the commits/branches/releases endpoints directly (not a secondary description):

- **Last commit to `master`**: 2021-04-01T12:45:50Z, SHA `a9995e2f5407f22eb7fb9dceac2b71a35276bb41`, message "Update powertrain plot layout" (author: Thomas Herrmann). Confirmed via `api.github.com/repos/.../commits?per_page=1`.
- **`pushed_at` on the repo object is 2023-07-06**, which is *not* a master-branch commit — cross-checking the `branches` endpoint shows this timestamp belongs to an automated `dependabot/pip/scipy-1.10.0` branch (a dependency-bump PR branch that was never merged), not real project activity. Master itself has been untouched since April 2021.
- **Total commit count on master**: 56 (from repo page metadata).
- **Releases/tags**: none — `api.github.com/repos/.../releases` returned an empty array. There has never been a tagged release.
- **Open issues**: 6 actual issues (list checked directly: #18 "Quadprog version" Sep 2024, #17 "Use min_curv to warm start min_time" Jul 2024, #15 "scikit version conflict on running main_globaltraj.py" Aug 2023, #12 "Display output" May 2023, #11 spline_approximation error Oct 2022, #10 a CasADi/IPOPT error question Sep 2022).
- **Open PRs**: 3 (the API's combined `open_issues_count` = 9 = 6 issues + 3 PRs, consistent with the issues-list count of 6).
- **Stars/forks**: 615 stars / 240 forks — meaningful community interest/reuse, but not itself evidence of active maintenance.
- **Archived flag**: `archived: false`, `disabled: false` — GitHub does not consider it formally archived, but the maintainers have not marked it read-only either; it's just dormant.

**Verdict**: The repo is **dormant, not archived** — no commits merged to master in ~5.5 years (since April 2021), no tagged releases ever, and several open issues (some over a year old, including at least two reporting the code failing to run out-of-the-box against newer `scipy`/`scikit-learn` versions — #15 and implicitly #18 re: `quadprog`) that have received no maintainer response visible on the issue list. This strongly suggests it is **not usable as a drop-in, currently-installable dependency** without dependency-pinning work first — the reported version-conflict issues (open, unresolved, 2022–2024) indicate it needs the original pinned dependency versions (or manual patching) to run at all on a modern Python environment. Treat any adoption as "vendor + patch," not "pip install and go."

### 4. What it actually does / input format

Confirmed directly from the raw `Readme.md` and a sample input file, not a secondary summary:

- It implements exactly the family of algorithms the project's docs assume: **shortest-path**, **minimum-curvature** (with an optional iterative refinement step), **minimum-time**, and **minimum-time with powertrain modeling** (thermal behavior, power losses, battery state-of-charge) racing-line optimizers, run via `main_globaltraj.py`. The minimum-curvature method is described in the README as approximating minimum-time performance in corners but diverging where acceleration limits aren't fully exploited; minimum-time is the more accurate but much more parameter- and compute-heavy option.
- **Input format**: confirmed by reading an actual sample track file directly — `inputs/tracks/berlin_2018.csv`, first lines:
  ```
  # x_m,y_m,w_tr_right_m,w_tr_left_m
  216.01,5.1944,5.6174,4.2348
  216.95,6.2147,5.42,4.3626
  ...
  ```
  i.e. a plain CSV of **centerline x/y coordinates in meters plus per-point track width to the right and left of centerline, in meters** — not a parametric spline object, not a set of control points/knots. Internally the code fits its own spline (`helper_funcs_glob/src/prep_track.py` handles track preparation/spline fitting) from these discrete centerline+width samples.
  - **Implication for "consuming this project's spline-based track representation"**: this repo does not consume an arbitrary spline object directly — it wants discretized centerline+width samples in this specific 4-column CSV convention (`x_m,y_m,w_tr_right_m,w_tr_left_m`). If ai-racing-bot's track representation is a parametric spline (e.g., a set of control points/knots or a continuous curve object), integration would require writing an export/adapter step that samples that spline at some resolution and writes it out in this `x_m,y_m,w_tr_right_m,w_tr_left_m` CSV convention (or an in-memory equivalent of the same array shape) before calling into `main_globaltraj.py` / the underlying `prep_track.py` pipeline. This is a small but real adapter, not zero-effort "just call it with our spline."
  - It additionally optionally consumes vehicle-dynamics/ggv-diagram files and friction-map files (`inputs/frictionmaps/*.json`) for the min-time/friction-aware modes; these are optional extras beyond the minimum requirement.

## Gaps

- I could not verify a precise "last release" beyond "no releases exist" — there is no versioned release to point to, only raw commit history.
- I did not attempt to actually clone/run the code (out of scope for a read-only research task), so I cannot independently confirm the exact severity of the dependency-version issues (#15, #18) beyond what's stated in the issue titles/dates on the issue list — I did not read the full issue bodies/comment threads, only the list view.
- I did not check for forks of this repo that might have a maintained/patched fork (some dormant TUM repos have community forks with dependency fixes); if the team wants to adopt this, checking the forks list for a maintained fork with fixed `requirements.txt` pins would be a reasonable quick follow-up before deciding "vendor + patch ourselves" vs. "use an existing patched fork."
- ai-racing-bot's own eventual project license is undecided (no LICENSE file exists yet), so I can't give a final "yes this is compatible with our license" — only "here is exactly what LGPL-3.0 requires and here's the fact that we currently have no license to compare it against."

## Recommendation for the Phase 4 build-vs-adapt ticket

TUM's `global_racetrajectory_optimization` (https://github.com/TUMFTM/global_racetrajectory_optimization) is real, correctly named, and does implement the minimum-curvature/minimum-time racing-line optimization the docs assumed — but it is LGPL-3.0 (fine to use as an unmodified external dependency; requires care and LGPL notice/source-availability if the team vendors and modifies its source directly, and the ai-racing-bot project itself has no LICENSE file yet to reconcile against), and it has had no commits to `master` since April 2021, no tagged releases ever, and multiple open, unresolved issues from 2022–2024 about it failing to run against newer `scipy`/`scikit-learn`/`quadprog` versions. It is dormant, not archived — usable, but only as "vendor a pinned copy and expect to patch dependency versions yourself," not as a maintained pip-installable dependency. Its input contract is a specific 4-column CSV (`x_m,y_m,w_tr_right_m,w_tr_left_m` — discretized centerline + track width), so wiring it to this project's spline-based track representation requires writing a small sampling/export adapter, not a zero-effort integration. Given the dependency-pinning work implied by the open issues plus the adapter needed for the input format, this should be scoped in the Phase 4 ticket as a bounded but real integration effort (pin+patch a vendored copy, write the CSV export adapter) rather than either "trivial drop-in" or "not usable at all" — and the license question is a non-blocker for using it as an external reference/tool, but becomes a real one the moment any of its source is copied and modified into this repo.
