$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_cross_day_day1_day2.csv"
$CHECKPOINT = "runs\cross_day_hybrid_cnnstem_spf256\rfhstu_cnnstem_cross_attn_chirp\best.pt"

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

function Run-Eval {
    param(
        [string]$Mode,
        [string]$VoteMode
    )

    Run-Step "$Mode $VoteMode" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint $CHECKPOINT `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --mode $Mode `
            --file-vote-mode $VoteMode `
            --device $DEVICE `
            --out-dir "outputs\vote_modes_hybrid_spf256\$Mode\$VoteMode"
    }
}

Run-Eval "classifier" "mean_logits"
Run-Eval "classifier" "mean_prob"
Run-Eval "classifier" "confidence_weighted"
Run-Eval "prototype" "mean_logits"
Run-Eval "prototype" "mean_prob"
Run-Eval "prototype" "confidence_weighted"

Run-Step "summarize vote_modes_hybrid_spf256" {
    & $PY scripts\summarize_results.py --root outputs\vote_modes_hybrid_spf256
}

Write-Host ""
Write-Host "Hybrid SPF256 vote-mode evaluation passed"
