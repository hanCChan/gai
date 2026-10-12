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

    $RunDir = "runs\cross_day_oob_main_spf128\$Name"
    $OutRoot = "outputs\cross_day_oob_main_spf128\$Name"

    Run-Step "finetune $Name" {
        & $PY scripts\finetune.py `
            --manifest $MANIFEST `
            --epochs 80 `
            --batch-size 16 `
            --samples-per-file 128 `
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
            --mode classifier `
            --device $DEVICE `
            --out-dir "$OutRoot\classifier"
    }

    Run-Step "evaluate prototype $Name" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint "$RunDir\best.pt" `
            --mode prototype `
            --device $DEVICE `
            --out-dir "$OutRoot\prototype"
    }
}

Run-Experiment "baseline_single_concat" @("--oob-fusion-type", "concat_oob")
Run-Experiment "cross_attn_oob_no_chirp" @("--oob-fusion-type", "cross_attn_oob", "--use-oob-cross-attention")
Run-Experiment "chirp_cross_attn_oob" @("--oob-fusion-type", "cross_attn_oob", "--use-oob-cross-attention", "--use-chirp-embedding")

Run-Step "summarize cross_day_oob_main_spf128" {
    & $PY scripts\summarize_results.py --root outputs\cross_day_oob_main_spf128
}

Write-Host ""
Write-Host "Cross-day OOB main SPF128 runner passed"

