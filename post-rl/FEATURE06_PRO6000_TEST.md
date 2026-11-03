# Feature 06 / pro6000: isolated 10-episode BC smoke

The data at $HOME/kdc_ARBiM/data/icra_task1_lerobot/lerobot contains
the source LeRobot dataset (reportedly about 600 recording bags; verify
actual total_episodes in meta/info.json). The LeTools repository is
$HOME/LeTools-Learning; use conda environment letools_arbim on pro6000.

Important: one bag usually corresponds to one LeRobot episode in this
pipeline, but this is a metadata fact to verify, not an assumption. The
script extracts the FIRST ten dataset EPISODES, not 10 random files.

LeRobot v3 video and parquet files can pack multiple episodes together.
Do NOT copy the first 10 files or manually edit meta/info.json. The
helper below uses the pinned native LeRobot reader and writer to create
an independent valid dataset with exactly 10 episodes. This requires
decoding/encoding only those selected episodes; no other bags are
re-encoded. Original dataset remains untouched.

## A. Checkout and verify source metadata

    cd "$HOME/LeTools-Learning"
    conda activate letools_arbim
    git fetch origin
    git switch feature/rl100-bc-production-v2
    git pull --ff-only origin feature/rl100-bc-production-v2
    git submodule update --init third_party/lerobot third_party/rl100
    mkdir -p post_traing/outputs

    export DATA_SOURCE="$HOME/kdc_ARBiM/data/icra_task1_lerobot/lerobot"
    export DATA10="$HOME/LeTools-Learning/post_traing/datasets/icra_task1_test10"

    python - <<'PY'
    import json, os
    from pathlib import Path
    path = Path(os.environ["DATA_SOURCE"]) / "meta" / "info.json"
    info = json.loads(path.read_text())
    print("source:", path.parent.parent)
    print("LeRobot version:", info.get("codebase_version"))
    print("episodes:", info.get("total_episodes"))
    print("frames:", info.get("total_frames"))
    print("cameras:", [k for k in info["features"] if k.startswith("observation.images.")])
    assert int(info["total_episodes"]) >= 10
    PY

Stop here if metadata is missing or is not compatible with the pinned
LeRobot v3 Reader.

## B. Extract 10 episodes without modifying the original 600

    python post-rl/test/extract_lerobot_test_subset.py \
      --source "$DATA_SOURCE" \
      --output "$DATA10" \
      --count 10 --start 0 \
      --source-repo-id local/icra_task1_full \
      --repo-id local/icra_task1_test10 \
      --video-backend pyav \
      2>&1 | tee post_traing/outputs/feature06_extract10.log

Expected "[PASS] independent LeRobot dataset: 10 episodes, ... frames".
The new directory contains independently generated data/meta/videos,
and extraction_manifest.json. The helper refuses to overwrite any
pre-existing output. If the extraction fails midway, inspect or move
that partial directory and use a different new --output path.

The new metadata will report exactly 10 episodes; the number of frames
varies depending on original recording lengths. It is NOT necessarily
1482 frames (that count belongs to a different 10-bag dataset used in
the previous server's Feature03 smoke).

## C. Local BC production verification (no simulator or ROS needed)

    set -o pipefail
    PYTHONPATH=post-rl/src python -m unittest discover \
      -s post-rl/test -p test_extract_lerobot_subset.py -v \
      2>&1 | tee post_traing/outputs/feature06_extract_unit.log

    PYTHONPATH=post-rl/src python -m unittest discover \
      -s post-rl/test -p test_bc_production.py -v \
      2>&1 | tee post_traing/outputs/feature06_unit.log

    python post-rl/test/smoke_bc_production.py \
      --dataset-root "$DATA10" \
      --repo-id local/icra_task1_test10 \
      --device cuda:0 \
      --output-dir post_traing/outputs/feature06_bc_smoke_pro6000 \
      2>&1 | tee post_traing/outputs/feature06_bc_smoke_pro6000.log

The GPU smoke uses the existing small bc.yaml *architecture* but
overrides the dataset root/repo id in memory. It runs real LeRobot RGB,
train-only normalization, two optimizer updates, heldout validation,
checkpoint save, resume to step 3, best EMA checkpoint strict load by
the original Kuavo ModelServerAdapter and inference. It never starts
the actual Kuavo OpenSource simulator.

## D. Optional Feature03 unchanged regression with the same test10 dataset

The legacy Feature03 smoke reads its dataset path from a YAML. Generate
a runtime-only override outside tracked source files:

    python - <<'PY'
    import os
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("post-rl/config/bc.yaml")
    cfg.data.root = os.environ["DATA10"]
    cfg.data.repo_id = "local/icra_task1_test10"
    OmegaConf.save(cfg, "post_traing/outputs/feature03_pro6000_test10.yaml")
    PY

    python post-rl/test/smoke_bc_real.py \
      --config post_traing/outputs/feature03_pro6000_test10.yaml \
      --output-dir post_traing/outputs/feature03_bc_pro6000_regression \
      2>&1 | tee post_traing/outputs/feature06_feature03_regression.log

Use a NEW output-dir if rerunning; prior checkpoints are not overwritten.

Do not run full 100000-step production training on the 10-episode smoke
subset. A separate later experiment should select the full dataset and
final hyperparameters; no claim about Kuavo task performance follows
from these three-step interface tests.

Feature06 must be server-tested before merging it to arbim/dev.
