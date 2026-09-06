#!/usr/bin/env bash
# One-command real-data experiment: curate -> train -> evaluate -> QC.
#
# Run this AFTER `medimg_pipeline data-import --dataset ircad ...` has
# already produced data/ircad/ (see docs/data_access.md). This script
# does not download or import anything itself -- it only runs the
# already-imported, curated dataset through training and evaluation.
#
# Usage: bash scripts/run_real_experiment.sh [data-root] [config]
#   data-root defaults to data/ircad
#   config    defaults to configs/train_small.yaml

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${1:-data/ircad}"
CONFIG="${2:-configs/train_small.yaml}"
OUT="outputs/real_experiment"

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: $DATA_ROOT does not exist. Run 'medimg_pipeline data-import --dataset ircad ...' first."
  exit 1
fi

echo "1/3: curating $DATA_ROOT ..."
python -m medimg_pipeline curate \
  --data-root "$DATA_ROOT" \
  --output-manifest "$OUT/manifest.csv" \
  --output-qc-report "$OUT/qc_report.json"

echo
echo "2/3: training ($CONFIG) ..."
python - "$CONFIG" "$OUT/manifest.csv" "$OUT/training" << 'EOF'
import sys
import yaml

config_path, manifest_path, output_dir = sys.argv[1:4]
with open(config_path) as f:
    config = yaml.safe_load(f)
config["manifest_path"] = manifest_path
config["output_dir"] = output_dir
with open("/tmp/_real_experiment_train_config.yaml", "w") as f:
    yaml.safe_dump(config, f)
EOF
python -m medimg_pipeline train --config /tmp/_real_experiment_train_config.yaml

CHECKPOINT=$(ls "$OUT"/training/*_best.pt | head -1)

echo
echo "3/3: evaluating on held-out test subjects ..."
python -m medimg_pipeline evaluate \
  --manifest "$OUT/manifest.csv" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT/evaluation" \
  --split test

echo
echo "Done. Results:"
echo "  $OUT/evaluation/evaluation_results.csv   (per-subject Dice/IoU)"
echo "  $OUT/evaluation/evaluation_report.json"
echo "  $OUT/evaluation/qc_overlay_*.png          (ground truth vs. prediction)"
