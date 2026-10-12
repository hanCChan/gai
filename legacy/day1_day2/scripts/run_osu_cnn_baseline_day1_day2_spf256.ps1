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
        [string]$InputType
    )

    $RunDir = "runs\osu_cnn_day1_day2_spf256\$Name"
    $OutRoot = "outputs\osu_cnn_day1_day2_spf256\$Name"

    Run-Step "finetune $Name" {
        & $PY scripts\finetune.py `
            --manifest $MANIFEST `
            --model-type osu_cnn `
            --cnn-input-type $InputType `
            --epochs 80 `
            --batch-size 16 `
            --samples-per-file 256 `
            --eval-samples-per-file 256 `
            --device $DEVICE `
            --out-dir $RunDir
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

Run-Experiment "osu_cnn_iq" "iq"
Run-Experiment "osu_cnn_fft_inband" "fft_inband"
Run-Experiment "osu_cnn_fft_oob" "fft_oob"
Run-Experiment "osu_cnn_amp_phase" "amp_phase"

Run-Step "summarize osu_cnn_day1_day2_spf256" {
    & $PY scripts\summarize_results.py --root outputs\osu_cnn_day1_day2_spf256
}

Write-Host ""
Write-Host "OSU CNN Day1->Day2 SPF256 baseline runner passed"
