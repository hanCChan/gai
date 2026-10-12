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

    $RunDir = "runs\hybrid_supcon_v2_spf256\$Name"
    $OutRoot = "outputs\hybrid_supcon_v2_spf256\$Name"

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

    Run-Step "evaluate classifier $Name" {
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
Run-Experiment "supcon_8x2_w005_t01_proj" @("--use-supcon", "--supcon-weight", "0.05", "--supcon-temperature", "0.1", "--balanced-batch", "--devices-per-batch", "8", "--samples-per-device", "2", "--use-supcon-proj", "--supcon-proj-dim", "64")
Run-Experiment "supcon_4x4_w005_t01_proj" @("--use-supcon", "--supcon-weight", "0.05", "--supcon-temperature", "0.1", "--balanced-batch", "--devices-per-batch", "4", "--samples-per-device", "4", "--use-supcon-proj", "--supcon-proj-dim", "64")
Run-Experiment "supcon_4x4_w003_t01_proj" @("--use-supcon", "--supcon-weight", "0.03", "--supcon-temperature", "0.1", "--balanced-batch", "--devices-per-batch", "4", "--samples-per-device", "4", "--use-supcon-proj", "--supcon-proj-dim", "64")
Run-Experiment "supcon_4x4_w005_t01_proj_aug" @("--use-supcon", "--supcon-weight", "0.05", "--supcon-temperature", "0.1", "--balanced-batch", "--devices-per-batch", "4", "--samples-per-device", "4", "--use-supcon-proj", "--supcon-proj-dim", "64", "--augment-rf", "--aug-phase-std", "0.05", "--aug-amp-std", "0.05", "--aug-noise-std", "0.005", "--aug-time-shift", "32")

Run-Step "summarize hybrid_supcon_v2_spf256" {
    & $PY scripts\summarize_results.py --root outputs\hybrid_supcon_v2_spf256
}

Write-Host ""
Write-Host "Hybrid SupCon v2 SPF256 runner passed"
