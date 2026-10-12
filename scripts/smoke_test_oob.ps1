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

function Run-OobMode {
    param(
        [string]$Mode,
        [bool]$UseCrossAttention
    )

    $OutDir = "runs\smoke_oob_$Mode"
    $CrossFlag = @()
    if ($UseCrossAttention) {
        $CrossFlag = @("--use-oob-cross-attention")
    }

    Run-Step "finetune $Mode" {
        & $PY scripts\finetune.py `
            --manifest $MANIFEST `
            --max-files 4 `
            --samples-per-file 2 `
            --epochs 1 `
            --batch-size 2 `
            --dim 32 `
            --depth 1 `
            --device $DEVICE `
            --out-dir $OutDir `
            --oob-fusion-type $Mode `
            @CrossFlag
    }

    Run-Step "evaluate classifier $Mode" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint "$OutDir\best.pt" `
            --max-files 4 `
            --samples-per-file 2 `
            --batch-size 2 `
            --dim 32 `
            --depth 1 `
            --device $DEVICE
    }

    Run-Step "evaluate prototype $Mode" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint "$OutDir\best.pt" `
            --max-files 4 `
            --samples-per-file 2 `
            --batch-size 2 `
            --dim 32 `
            --depth 1 `
            --device $DEVICE `
            --prototype
    }
}

Run-OobMode "no_oob" $false
Run-OobMode "concat_oob" $false
Run-OobMode "cross_attn_oob" $true

Write-Host ""
Write-Host "OOB smoke test passed on device=$DEVICE"

