$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_cross_day_day1_day2.csv"
$CHECKPOINT = "runs\hybrid_regularization_spf256\label_smoothing_005_weight_decay_5e4\best.pt"

# Fallback to the pre-regularization best hybrid checkpoint if the LS+WD checkpoint is unavailable.
if (-not (Test-Path $CHECKPOINT)) {
    $CHECKPOINT = "runs\cross_day_hybrid_cnnstem_spf256\rfhstu_cnnstem_cross_attn_chirp\best.pt"
}

Set-Location $ROOT

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

function Run-FusionEval {
    param([string]$Alpha)

    Run-Step "evaluate score fusion alpha=$Alpha" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint $CHECKPOINT `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --mode classifier `
            --score-fusion `
            --fusion-alpha $Alpha `
            --file-vote-mode mean_prob `
            --device $DEVICE `
            --out-dir "outputs\fusion_hybrid_best_spf256\alpha_$Alpha"
    }
}

Run-FusionEval "0.25"
Run-FusionEval "0.5"
Run-FusionEval "0.75"

Run-Step "summarize fusion_hybrid_best_spf256" {
    & $PY scripts\summarize_results.py --root outputs\fusion_hybrid_best_spf256
}

Write-Host ""
Write-Host "Hybrid score fusion SPF256 evaluation passed"
