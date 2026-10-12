$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_cross_day_day1_day2.csv"
$CHECKPOINT = "runs\hybrid_regularization_spf256\label_smoothing_005_weight_decay_5e4\best.pt"

Set-Location $ROOT

# Fallback to the Center Loss best checkpoint if the LS+WD checkpoint is unavailable.
if (-not (Test-Path $CHECKPOINT)) {
    $CHECKPOINT = "runs\hybrid_center_loss_spf256\center_w001\best.pt"
}

function Run-Step {
    param([string]$StepName, [scriptblock]$Command)
    Write-Host ""
    Write-Host "==> $StepName"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $StepName"
        exit $LASTEXITCODE
    }
}

function Run-Seed {
    param([int]$EvalSeed)

    Run-Step "classifier eval_seed=$EvalSeed" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint $CHECKPOINT `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --eval-seed $EvalSeed `
            --mode classifier `
            --file-vote-mode mean_logits `
            --device $DEVICE `
            --out-dir "outputs\eval_seed_sensitivity_hybrid\seed_$EvalSeed\classifier"
    }

    Run-Step "prototype eval_seed=$EvalSeed" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint $CHECKPOINT `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --eval-seed $EvalSeed `
            --mode prototype `
            --file-vote-mode confidence_weighted `
            --device $DEVICE `
            --out-dir "outputs\eval_seed_sensitivity_hybrid\seed_$EvalSeed\prototype_confidence_weighted"
    }
}

foreach ($EvalSeed in @(0, 1, 2, 3, 4)) {
    Run-Seed $EvalSeed
}

Run-Step "summarize eval_seed_sensitivity_hybrid" {
    & $PY scripts\summarize_results.py --root outputs\eval_seed_sensitivity_hybrid
}

Run-Step "summarize eval seed mean/std" {
    & $PY scripts\summarize_eval_seed_sensitivity.py `
        --summary outputs\eval_seed_sensitivity_hybrid\summary.csv `
        --out outputs\eval_seed_sensitivity_hybrid\seed_sensitivity_summary.csv
}

Write-Host ""
Write-Host "Hybrid eval-seed sensitivity passed"
