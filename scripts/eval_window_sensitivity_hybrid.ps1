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

function Run-EvalSet {
    param([int]$Spf)

    Run-Step "classifier eval_samples_per_file=$Spf" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint $CHECKPOINT `
            --samples-per-file 256 `
            --eval-samples-per-file $Spf `
            --mode classifier `
            --file-vote-mode mean_logits `
            --device $DEVICE `
            --out-dir "outputs\eval_window_sensitivity_hybrid\spf_$Spf\classifier"
    }

    Run-Step "prototype eval_samples_per_file=$Spf" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint $CHECKPOINT `
            --samples-per-file 256 `
            --eval-samples-per-file $Spf `
            --mode prototype `
            --file-vote-mode confidence_weighted `
            --device $DEVICE `
            --out-dir "outputs\eval_window_sensitivity_hybrid\spf_$Spf\prototype_confidence_weighted"
    }
}

foreach ($Spf in @(32, 64, 128, 256, 512, 1024)) {
    Run-EvalSet $Spf
}

Run-Step "summarize eval_window_sensitivity_hybrid" {
    & $PY scripts\summarize_results.py --root outputs\eval_window_sensitivity_hybrid
}

Write-Host ""
Write-Host "Hybrid eval-window sensitivity passed"
