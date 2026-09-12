#!/bin/bash
# Driver for pptsm_visualize_pipeline.py — training-consistent detection + single-object
# tracking + PP-TSM classification visualization.
#
# Usage:
#   bash run_pptsm_visualize.sh full  <video> <out.mp4> [extra args...]
#       -> annotate the ENTIRE video (all persons, all frames) -> out.mp4 + out.jsonl
#   bash run_pptsm_visualize.sh clips <video> <out_dir> [extra args...]
#       -> per-person cropped+annotated videos -> out_dir/person-XXX.mp4 (+ jsonl)
#
# Examples (run from the project root on the server):
#   bash scripts/run_pptsm_visualize.sh full  video.mpg out/full.mp4
#   bash scripts/run_pptsm_visualize.sh clips video.mpg out/persons --min-track-frames 20
#
# All pipeline options can be appended (--det-stride 3 --det-thresh 0.6 --min-person-h 150
# --iou-thresh 0.3 --size-ratio 0.6 --center-ratio 1.0 --track-buffer 5 --start-ms ... ).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- conda env -----------------------------------------------------------------
source /home/e-ai2/miniconda3/etc/profile.d/conda.sh
conda activate /home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle
export PYTHONPATH="$PROJECT_ROOT/third_party/PaddleVideo"

MODE="${1:?usage: run_pptsm_visualize.sh <full|clips> <video> <out> [args...]}"
shift

CONFIG=shoplift/configs/paddlevideo/pptsm_steal_frames_dense_fight16.yaml
WEIGHTS=outputs/wqh/paddlevideo/pptsm_steal_fight16/ppTSM_epoch_00120.pdparams

if [ "$MODE" = "full" ]; then
    VIDEO="${1:?need video}"; shift
    OUT="${1:?need out.mp4}"; shift
    mkdir -p "$(dirname "$OUT")"
    python scripts/pptsm_visualize_pipeline.py run-full \
        --video "$VIDEO" --config "$CONFIG" --weights "$WEIGHTS" \
        --out "$OUT" "$@"
elif [ "$MODE" = "clips" ]; then
    VIDEO="${1:?need video}"; shift
    OUTDIR="${1:?need out_dir}"; shift
    mkdir -p "$OUTDIR"
    python scripts/pptsm_visualize_pipeline.py run-clips \
        --video "$VIDEO" --config "$CONFIG" --weights "$WEIGHTS" \
        --out-dir "$OUTDIR" "$@"
else
    echo "unknown mode: $MODE (use 'full' or 'clips')" >&2
    exit 2
fi
