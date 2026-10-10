#!/usr/bin/env bash
# Full pipeline: pretrain -> coding SFT -> loop scaling -> self-improvement -> experiments -> charts.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d runs/pretrain-001/checkpoint-0150 ]; then
  ./run_gpu.sh scripts/train_pretrain.py --output runs/pretrain-001 \
    --steps 150 --checkpoints 50,100,150 --batch-size 16 --randomize-loops --seed 42
fi
if [ ! -d runs/coding-sft-001/checkpoint-0060 ]; then
  ./run_gpu.sh scripts/train_sft.py --initial-checkpoint runs/pretrain-001/checkpoint-0150 \
    --output runs/coding-sft-001 --steps 60 --checkpoints 60 --batch-size 16 --seed 7
fi
./run_gpu.sh scripts/evaluate_loop_scaling.py --checkpoint runs/coding-sft-001/checkpoint-0060 \
  --split confirm --per-family 1 --output artifacts/receipts/loop-scaling.json
./run_gpu.sh scripts/train_self_improve.py --initial-checkpoint runs/coding-sft-001/checkpoint-0060 \
  --output runs/self-improve-001 --cycles 2 --proposals-per-task 16 \
  --families-ladder reverse --per-family 4 --eval-per-family 2 --seed 73
./run_gpu.sh experiments/looped_vs_deep.py --output artifacts/experiments/looped-vs-deep \
  --steps 120 --batch-size 16 --seeds 11,22
./run_gpu.sh experiments/analyze_loop_experiments.py \
  --results artifacts/experiments/looped-vs-deep/results.json \
  --output artifacts/experiments/looped-vs-deep/analysis.json --higher-is-better
./run_gpu.sh experiments/loop_depth_policy.py --output artifacts/experiments/depth-policy \
  --steps 120 --batch-size 16 --seeds 11,22
./run_gpu.sh experiments/analyze_loop_experiments.py \
  --results artifacts/experiments/depth-policy/results.json \
  --output artifacts/experiments/depth-policy/analysis.json --higher-is-better
./run_gpu.sh scripts/make_charts.py --output artifacts/charts \
  --pretrain-receipt runs/pretrain-001/training-receipt.json \
  --loop-scaling-receipt artifacts/receipts/loop-scaling.json \
  --looped-deep-results artifacts/experiments/looped-vs-deep/results.json
echo "pipeline complete"
