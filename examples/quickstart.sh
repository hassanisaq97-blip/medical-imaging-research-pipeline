#!/usr/bin/env bash
# End-to-end walkthrough on synthetic data: curate -> train -> infer -> QC.
#
# This reproduces exactly what tests/test_training_and_inference_smoke.py
# checks automatically, but leaves its outputs on disk under
# examples/_quickstart_output/ for you to inspect (open the QC PNG, look
# at the manifest CSV, read the metadata JSON). Every result here is a
# synthetic smoke-test result -- see docs/segmentation_target_selection.md
# and README.md for what a real run against Multimodal-HC would need
# (a completed Data User Agreement; see docs/data_access.md).
#
# Usage: bash examples/quickstart.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=examples/_quickstart_output
rm -rf "$OUT"
mkdir -p "$OUT"

echo "1/5: generating a tiny synthetic dataset (10 subjects, no real patient data)..."
python -c "
from pathlib import Path
from tests.helpers import make_synthetic_volume_pair
root = Path('$OUT/synthetic_dataset')
for i in range(10):
    make_synthetic_volume_pair(root, f'sub-{i:03d}', shape=(32, 32, 32), spacing=(2.0, 2.0, 2.0))
"

echo "2/5: curating a manifest + QC report..."
python -m medimg_pipeline curate \
  --data-root "$OUT/synthetic_dataset" \
  --min-foreground-voxels 10 \
  --output-manifest "$OUT/manifest.csv" \
  --output-qc-report "$OUT/qc_report.json"

echo "3/5: writing a tiny CPU training config..."
cat > "$OUT/train_config.yaml" << EOF
manifest_path: $OUT/manifest.csv
output_dir: $OUT/training
experiment_name: quickstart
patch_size: [16, 16, 16]
batch_size: 1
channels: [4, 8, 16]
strides: [2, 2]
num_res_units: 1
num_epochs: 2
val_interval: 1
early_stopping_patience: 5
device: cpu
amp: false
seed: 42
EOF

echo "4/5: training for 2 epochs on CPU (synthetic smoke test, not a real result)..."
python -m medimg_pipeline train --config "$OUT/train_config.yaml"

echo "5/5: running inference + visual QC on one subject..."
python -m medimg_pipeline infer \
  --input "$OUT/synthetic_dataset/sub-000/ct/sub-000_ct.nii.gz" \
  --checkpoint "$OUT/training/quickstart_best.pt" \
  --output "$OUT/inference" \
  --device cpu \
  --qc

echo
echo "Done. Inspect:"
echo "  $OUT/manifest.csv          (curated dataset manifest)"
echo "  $OUT/qc_report.json        (curation QC report)"
echo "  $OUT/training/quickstart.csv  (per-epoch training metrics)"
echo "  $OUT/inference/*_seg.nii.gz   (predicted mask)"
echo "  $OUT/inference/qc_overlay.png (visual QC)"
