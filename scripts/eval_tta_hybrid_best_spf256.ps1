$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_cross_day_day1_day2.csv"
$CHECKPOINT = "runs\hybrid_regularization_spf256\label_smoothing_005_weight_decay_5e4\best.pt"

Set-Location $ROOT

# Fallback to the pre-regularization best hybrid checkpoint if the LS+WD checkpoint is unavailable.
if (-not (Test-Path $CHECKPOINT)) {
    $CHECKPOINT = "runs\cross_day_hybrid_cnnstem_spf256\rfhstu_cnnstem_cross_attn_chirp\best.pt"
}

function Run-Step {
    param([string]$Name, [scriptblock]$Command)
    Write-Host ""
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name"
        exit $LASTEXITCODE
    }
}

function Run-Eval {
    param(
        [string]$Name,
        [string]$Mode,
        [string]$VoteMode,
        [string[]]$TtaFlags
    )

    Run-Step "evaluate $Name" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint $CHECKPOINT `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --mode $Mode `
            --file-vote-mode $VoteMode `
            --device $DEVICE `
            --out-dir "outputs\tta_hybrid_best_spf256\$Name" `
            @TtaFlags
    }
}

Run-Eval "no_tta_classifier" "classifier" "mean_logits" @("--tta-mode", "none")
Run-Eval "bn_adapt_classifier" "classifier" "mean_logits" @("--tta-mode", "bn_adapt")
Run-Eval "tent_lr1e4_steps1_classifier" "classifier" "mean_logits" @("--tta-mode", "tent", "--tent-lr", "1e-4", "--tent-steps", "1")
Run-Eval "tent_lr1e5_steps1_classifier" "classifier" "mean_logits" @("--tta-mode", "tent", "--tent-lr", "1e-5", "--tent-steps", "1")
Run-Eval "tent_lr1e4_steps3_classifier" "classifier" "mean_logits" @("--tta-mode", "tent", "--tent-lr", "1e-4", "--tent-steps", "3")

Run-Eval "no_tta_prototype" "prototype" "confidence_weighted" @("--tta-mode", "none")
Run-Eval "bn_adapt_prototype" "prototype" "confidence_weighted" @("--tta-mode", "bn_adapt")
Run-Eval "tent_lr1e4_steps1_prototype" "prototype" "confidence_weighted" @("--tta-mode", "tent", "--tent-lr", "1e-4", "--tent-steps", "1")

Run-Step "summarize tta_hybrid_best_spf256" {
    & $PY scripts\summarize_results.py --root outputs\tta_hybrid_best_spf256
}

Write-Host ""
Write-Host "Hybrid TTA SPF256 evaluation passed"
