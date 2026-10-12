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
        [string[]]$Flags
    )

    $RunDir = "runs\cross_day_hybrid_cnnstem_spf256\$Name"
    $OutRoot = "outputs\cross_day_hybrid_cnnstem_spf256\$Name"

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
            --out-dir $RunDir `
            @Flags
    }

    Run-Step "evaluate classifier $Name" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint "$RunDir\best.pt" `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --mode classifier `
            --device $DEVICE `
            --out-dir "$OutRoot\classifier"
    }

    Run-Step "evaluate prototype $Name" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint "$RunDir\best.pt" `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --mode prototype `
            --device $DEVICE `
            --out-dir "$OutRoot\prototype"
    }
}

Run-Experiment "rfhstu_linear_cross_attn" @("--oob-fusion-type", "cross_attn_oob", "--use-oob-cross-attention", "--patch-embed-type", "linear")
Run-Experiment "rfhstu_cnnstem_concat" @("--patch-embed-type", "cnn_stem", "--cnn-stem-dim", "32", "--oob-fusion-type", "concat_oob")
Run-Experiment "rfhstu_cnnstem_cross_attn" @("--patch-embed-type", "cnn_stem", "--cnn-stem-dim", "32", "--oob-fusion-type", "cross_attn_oob", "--use-oob-cross-attention")
Run-Experiment "rfhstu_cnnstem_cross_attn_chirp" @("--patch-embed-type", "cnn_stem", "--cnn-stem-dim", "32", "--oob-fusion-type", "cross_attn_oob", "--use-oob-cross-attention", "--use-chirp-embedding")

Run-Step "summarize cross_day_hybrid_cnnstem_spf256" {
    & $PY scripts\summarize_results.py --root outputs\cross_day_hybrid_cnnstem_spf256
}

Write-Host ""
Write-Host "Cross-day hybrid CNN-stem SPF256 runner passed"
