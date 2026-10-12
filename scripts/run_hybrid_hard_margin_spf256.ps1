$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_cross_day_day1_day2.csv"

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

function Run-Experiment {
    param(
        [string]$Name,
        [string[]]$ExtraFlags
    )

    $RunDir = "runs\hybrid_hard_margin_spf256\$Name"
    $OutRoot = "outputs\hybrid_hard_margin_spf256\$Name"

    Run-Step "finetune $Name" {
        & $PY scripts\finetune.py `
            --manifest $MANIFEST `
            --epochs 80 `
            --batch-size 16 `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --dim 64 `
            --depth 2 `
            --device $DEVICE `
            --label-smoothing 0.05 `
            --weight-decay 5e-4 `
            --patch-embed-type cnn_stem `
            --cnn-stem-dim 32 `
            --oob-fusion-type cross_attn_oob `
            --use-oob-cross-attention `
            --use-chirp-embedding `
            --out-dir $RunDir `
            @ExtraFlags
    }

    Run-Step "evaluate classifier mean_logits $Name" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint "$RunDir\best.pt" `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --mode classifier `
            --file-vote-mode mean_logits `
            --device $DEVICE `
            --out-dir "$OutRoot\classifier"
    }

    Run-Step "evaluate prototype confidence_weighted $Name" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint "$RunDir\best.pt" `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --mode prototype `
            --file-vote-mode confidence_weighted `
            --device $DEVICE `
            --out-dir "$OutRoot\prototype_confidence_weighted"
    }
}

Run-Experiment "ce_ls_wd_base" @()
Run-Experiment "hard_margin_w005_m02" @("--use-hard-margin", "--hard-margin-weight", "0.05", "--hard-margin", "0.2")
Run-Experiment "hard_margin_w01_m02" @("--use-hard-margin", "--hard-margin-weight", "0.1", "--hard-margin", "0.2")
Run-Experiment "hard_margin_w005_m01" @("--use-hard-margin", "--hard-margin-weight", "0.05", "--hard-margin", "0.1")

Run-Step "summarize hybrid_hard_margin_spf256" {
    & $PY scripts\summarize_results.py --root outputs\hybrid_hard_margin_spf256
}

Write-Host ""
Write-Host "Hybrid hard-margin SPF256 runner passed"
