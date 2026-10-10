@echo off
rem Full pipeline: pretrain -> coding SFT -> loop scaling -> self-improvement -> experiments -> charts.
setlocal
cd /d "%~dp0.."
if not exist "runs\pretrain-001\checkpoint-0150" (
  run_cpu.bat scripts\train_pretrain.py --output runs\pretrain-001 --steps 150 ^
    --checkpoints 50,100,150 --batch-size 16 --randomize-loops --seed 42
)
if not exist "runs\coding-sft-001\checkpoint-0060" (
  run_cpu.bat scripts\train_sft.py --initial-checkpoint runs\pretrain-001\checkpoint-0150 ^
    --output runs\coding-sft-001 --steps 60 --checkpoints 60 --batch-size 16 --seed 7
)
if not exist "artifacts\receipts\loop-scaling.json" (
  run_cpu.bat scripts\evaluate_loop_scaling.py --checkpoint runs\coding-sft-001\checkpoint-0060 ^
    --split confirm --per-family 1 --output artifacts\receipts\loop-scaling.json
)
if not exist "runs\self-improve-001\training-receipt.json" (
  run_cpu.bat scripts\train_self_improve.py --initial-checkpoint runs\coding-sft-001\checkpoint-0060 ^
    --output runs\self-improve-001 --cycles 2 --proposals-per-task 16 ^
    --families-ladder reverse --per-family 4 --eval-per-family 2 --seed 73
)
if not exist "artifacts\experiments\looped-vs-deep\results.json" (
  run_cpu.bat experiments\looped_vs_deep.py --output artifacts\experiments\looped-vs-deep ^
    --steps 120 --batch-size 16 --seeds 11,22
  run_cpu.bat experiments\analyze_loop_experiments.py ^
    --results artifacts\experiments\looped-vs-deep\results.json ^
    --output artifacts\experiments\looped-vs-deep\analysis.json --higher-is-better
)
if not exist "artifacts\experiments\depth-policy\results.json" (
  run_cpu.bat experiments\loop_depth_policy.py --output artifacts\experiments\depth-policy ^
    --steps 120 --batch-size 16 --seeds 11,22
  run_cpu.bat experiments\analyze_loop_experiments.py ^
    --results artifacts\experiments\depth-policy\results.json ^
    --output artifacts\experiments\depth-policy\analysis.json --higher-is-better
)
run_cpu.bat scripts\make_charts.py --output artifacts\charts ^
  --pretrain-receipt runs\pretrain-001\training-receipt.json ^
  --loop-scaling-receipt artifacts\receipts\loop-scaling.json ^
  --looped-deep-results artifacts\experiments\looped-vs-deep\results.json
echo pipeline complete
endlocal
