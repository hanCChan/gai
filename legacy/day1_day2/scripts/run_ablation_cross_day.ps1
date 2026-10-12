$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_all.csv"

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

    $RunDir = "runs\ablation_cross_day\$Name"
    $OutRoot = "outputs\ablation_cross_day\$Name"

    Run-Step "finetune $Name" {
        & $PY scripts\finetune.py `
            --manifest $MANIFEST `
            --epochs 3 `
            --batch-size 8 `
            --samples-per-file 4 `
            --max-files 12 `
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
            --batch-size 8 `
            --samples-per-file 4 `
            --max-files 12 `
            --dim 64 `
            --depth 2 `
            --device $DEVICE `
            --mode classifier `
            --out-dir "$OutRoot\classifier"
    }

    Run-Step "evaluate prototype $Name" {
        & $PY scripts\evaluate.py `
            --manifest $MANIFEST `
            --checkpoint "$RunDir\best.pt" `
            --batch-size 8 `
            --samples-per-file 4 `
            --max-files 12 `
            --dim 64 `
            --depth 2 `
            --device $DEVICE `
            --mode prototype `
            --out-dir "$OutRoot\prototype"
    }
}

Run-Experiment "baseline_single_concat" @("--oob-fusion-type", "concat_oob")
Run-Experiment "chirp_concat" @("--use-chirp-embedding", "--oob-fusion-type", "concat_oob")
Run-Experiment "cross_attn_oob_no_chirp" @("--oob-fusion-type", "cross_attn_oob", "--use-oob-cross-attention")
Run-Experiment "chirp_cross_attn_oob" @("--use-chirp-embedding", "--oob-fusion-type", "cross_attn_oob", "--use-oob-cross-attention")
# Experimental only: multi-scale is not the current default formal model.
Run-Experiment "chirp_multiscale_concat" @("--use-chirp-embedding", "--use-multiscale", "--multiscale-ratios", "1,2,4", "--oob-fusion-type", "concat_oob")
# Experimental only: full_light is kept for ablation and is not the current default formal model.
Run-Experiment "full_light" @("--use-chirp-embedding", "--oob-fusion-type", "cross_attn_oob", "--use-oob-cross-attention", "--use-multiscale", "--multiscale-ratios", "1,2,4", "--multiscale-fusion-type", "concat_pool")

Write-Host ""
Write-Host "Ablation runner passed"
