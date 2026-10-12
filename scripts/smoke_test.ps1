$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_all.csv"

Set-Location $ROOT

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name"
        exit $LASTEXITCODE
    }
}

Run-Step "pretrain_mae smoke" {
    & $PY scripts\pretrain_mae.py `
        --manifest $MANIFEST `
        --max-files 2 `
        --samples-per-file 2 `
        --epochs 1 `
        --batch-size 2 `
        --dim 32 `
        --depth 1 `
        --device $DEVICE `
        --out-dir runs\smoke_pretrain
}

Run-Step "finetune smoke" {
    & $PY scripts\finetune.py `
        --manifest $MANIFEST `
        --max-files 4 `
        --samples-per-file 2 `
        --epochs 1 `
        --batch-size 2 `
        --dim 32 `
        --depth 1 `
        --device $DEVICE `
        --out-dir runs\smoke_finetune `
        --use-contrastive `
        --use-adversarial
}

Run-Step "evaluate classifier smoke" {
    & $PY scripts\evaluate.py `
        --manifest $MANIFEST `
        --checkpoint runs\smoke_finetune\best.pt `
        --max-files 4 `
        --samples-per-file 2 `
        --batch-size 2 `
        --dim 32 `
        --depth 1 `
        --device $DEVICE
}

Run-Step "evaluate prototype smoke" {
    & $PY scripts\evaluate.py `
        --manifest $MANIFEST `
        --checkpoint runs\smoke_finetune\best.pt `
        --max-files 4 `
        --samples-per-file 2 `
        --batch-size 2 `
        --dim 32 `
        --depth 1 `
        --device $DEVICE `
        --prototype
}

Write-Host ""
Write-Host "Smoke test passed on device=$DEVICE"

