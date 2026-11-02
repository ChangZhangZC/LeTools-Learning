# Data Wheel migration inventory — kdc_ARBiM -> LeTools-Learning

Status: **source-code audit + directory contract only**. No deployment or
rollout code has yet been ported. Reference source is
`ChangZhangZC/kdc_ARBiM@feature/flywheel`. Target is
`ChangZhangZC/LeTools-Learning@feature/rl100-data`.

## Verified behavior in the old implementation

| Source file | Behavior to preserve |
| --- | --- |
| `kuavo_deploy/eval_kuavo.py` | Interactive `9. data_wheel` choice, prompts for a positive **number of successful** rollouts; ordinary `8. auto_test` removes wheel environment flags. |
| `kuavo_deploy/eval_kuavo.py` | Progress bar `Data Wheel [██░] successes/target | Attempts: n`; polls `evaluation_autotest.log`, parses `Success Count: X / Already eval episodes: Y`, and rejects stale logs using mtime. |
| `kuavo_deploy/src/eval/sim_auto_test.py` | `ARBIM_DATA_WHEEL=1` enables an unbounded attempt loop via `itertools.count()` until `ARBIM_FLYWHEEL_SUCCESS_TARGET` successful episodes. Normal eval uses exactly `eval_episodes` attempts. |
| `kuavo_deploy/src/eval/sim_auto_test.py` | Normal output `outputs/eval/<task>/<method>/<timestamp>/epoch<epoch>`, wheel output `epoch<epoch>_flywheel`; separate `evaluation_autotest.log`, videos, and staging. |
| `kuavo_deploy/src/eval/sim_auto_test.py` | In wheel mode a `RolloutRecorder` is allocated; `ARBIM_DW_POST_SUCCESS_STEPS=5` default from menu, allowed 0–20, records short post-success tail when environment permits. |
| `kuavo_deploy/src/eval/rollout_recorder.py` | Records `T` actions and `T+1` observations; saves only successful episodes as `arbim_rollout_staging_v2` NPY, annotates sparse Reward V2; leaves invalid-reward successes usable for IL, not RL. |
| `post_training/src/post_rl/data/staging_to_lerobot.py` | Converts staging to LeRobot, retains reward provenance `meta/arbim_staging_provenance.json` and does not insert a synthetic final action. |
| `post_training/src/post_rl/data/{data_prepare,zarr_merge,flywheel_report}.py` | Validated sparse reward extraction, Zarr conversion/merge, per-round counts and manifest/report checks. |

**Naming correction:** The audited `feature/flywheel` uses
`epoch<epoch>_flywheel` as its evaluation directory suffix; it does **not**
use the literal `run_time_flywheel` in the files inspected. Maintain a
separate wheel-labelled evaluation run without silently renaming existing
run identifiers. Other historical branches may have different names.

## Differences in current LeTools (migration risks)

- Interactive entrypoint has changed from old
  `kuavo_deploy/eval_kuavo.py` to **`kuavo_deploy/eval.py`**.
  The current LeTools menu contains five options, plus a separate Roban
  whole-body menu; do **not** copy/paste the old 1–10 menu and overwrite it.
  Append a separate Flywheel option with appropriate menu numbering.
- LeTools `kuavo_deploy/src/eval/sim_auto_test.py` uses
  `resolve_eval_output_dir(pretrained_path, Path("outputs/eval"))`, unlike
  the old `outputs/eval/task/method/timestamp/epochN` composition. Flywheel
  output must reuse/extend this resolver rather than reconstructing a path
  that ignores arbitrary `pretrained_path`.
- The old `RolloutRecorder(require_depth=True)` expects three Depth
  streams while the first RL1002D integration trains with three **RGB**.
  Decide and test a capture policy separately: never silently fabricate
  missing Depth, and preserve Depth if actually captured.
- The old Staging/LeRobot converter depended on LeRobot 0.4.x and
  associated patches. The target checkout pins LeRobot 0.5.x; revalidate
  DatasetReader, create/finalize and metadata APIs rather than copying the
  old monkey patches.
- A success count is not a reward-valid count. Persist successes with
  invalid sparse Reward for possible IL, but exclude them from RL export.
- The current unbounded retry loop risks indefinite operation with a weak
  policy. Consider an **optional**, explicitly configured max-attempts or
  timeout safeguard that reports partial progress without changing the
  default success-target meaning.

## Target separation of persistent data vs run outputs

```text
data/flywheel/round_<id>/
    staging/                 # durable source rollouts, all successful captures
    lerobot/                 # authoritative converted archive
    offline_rl.zarr/         # derived cache, validated rewards only
    manifest.json            # source, policy, task, counts, reward provenance

outputs/eval/<eval-run-id>_flywheel/
    evaluation_autotest.log  # progress/status and attempt summaries
    rollout_*.mp4            # visual evaluation artifact
    metrics.json             # counts/success-rate/completion state
```

Both paths must be linked via a shared `round_id` in each run's manifest and
log. `data/` is git-ignored, while scripts and configs remain in their
existing project locations. **Do not move or delete** old rollout sources
automatically when integrating this layout.

## Proposed integration batches and required checks

1. **Deploy UI:** add a Flywheel choice to current `kuavo_deploy/eval.py`;
   preserve all existing choices, Roban routing, pause/stop/log controls,
   positive-success-target prompt and live progress indicator.
2. **Eval runner:** adapt the current `sim_auto_test.py` to success-target
   collection with separate wheel-labelled output; normal `auto_test`
   remains unchanged. Add written progress and attempt count with no
   cross-run log contamination.
3. **Recorder:** port `rollout_recorder.py` behind a clean optional hook,
   direct staging writes to `data/flywheel/round_<id>/staging`; retain
   `T+1` observation contract, verified Reward status, atomic save,
   optional post-success tail, and record capture modality provenance.
4. **Conversion:** adapt staging → LeRobot 0.5.x and verified
   LeRobot → Zarr; keep IL-valid / RL-valid distinction.
5. **Data Wheel E2E validation:** check failed attempts are never archived as
   successful staging; progress counts match collected files; all
   `T+1` observations are preserved; produced dataset is replayable and
   manifests connect `data/` with `outputs/eval/`.

No entries in this plan claim that Flywheel has already been implemented
in LeTools; those are future migration steps.
