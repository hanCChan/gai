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

function Run-MultiscaleCase {
    param(
        [string]$Name,
        [string]$OutDir,
        [string]$OobMode,
        [bool]$UseCrossAttention,
        [bool]$UseMultiscale
    )

    $CrossFlag = @()
    if ($UseCrossAttention) {
        $CrossFlag = @("--use-oob-cross-attention")
    }
    $MultiscaleFlag = @()
    if ($UseMultiscale) {
        $MultiscaleFlag = @("--use-multiscale", "--multiscale-ratios", "1,2,4", "--multiscale-fusion-type", "concat_pool")
    }

    Run-Step "finetune $Name" {
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
            --oob-fusion-type $OobMode `
            @CrossFlag `
            @MultiscaleFlag
    }

    Run-Step "evaluate classifier $Name" {
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

    Run-Step "evaluate prototype $Name" {
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

Run-MultiscaleCase "single_scale_concat_oob" "runs\smoke_ms_single" "concat_oob" $false $false
Run-MultiscaleCase "multiscale_concat_oob" "runs\smoke_ms_concat" "concat_oob" $false $true
Run-MultiscaleCase "multiscale_cross_attn_oob" "runs\smoke_ms_cross" "cross_attn_oob" $true $true

Write-Host ""
Write-Host "Multi-scale smoke test passed on device=$DEVICE"

