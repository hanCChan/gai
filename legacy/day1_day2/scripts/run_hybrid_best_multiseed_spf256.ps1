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

function Run-Seed {
    param([int]$Seed)

    $RunDir = "runs\hybrid_best_multiseed_spf256\seed_$Seed"
    $OutRoot = "outputs\hybrid_best_multiseed_spf256\seed_$Seed"

    Run-Step "finetune seed_$Seed" {
        & $PY scripts\finetune.py `
            --manifest $MANIFEST `
            --epochs 80 `
            --batch-size 16 `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --dim 64 `
            --depth 2 `
            --device $DEVICE `
            --seed $Seed `
            --patch-embed-type cnn_stem `
            --cnn-stem-dim 32 `
            --oob-fusion-type cross_attn_oob `
            --use-oob-cross-attention `
            --use-chirp-embedding `
            --label-smoothing 0.05 `
            --weight-decay 5e-4 `
            --out-dir $RunDir
    }

    Run-Step "evaluate classifier seed_$Seed" {
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

    Run-Step "evaluate prototype confidence_weighted seed_$Seed" {
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

Run-Seed 0
Run-Seed 1
Run-Seed 2

Run-Step "summarize hybrid_best_multiseed_spf256" {
    & $PY scripts\summarize_results.py --root outputs\hybrid_best_multiseed_spf256
}

Run-Step "summarize multiseed stats" {
    & $PY scripts\summarize_multiseed.py `
        --summary outputs\hybrid_best_multiseed_spf256\summary.csv `
        --out outputs\hybrid_best_multiseed_spf256\multiseed_summary.csv
}

Write-Host ""
Write-Host "Hybrid best multiseed SPF256 runner passed"
