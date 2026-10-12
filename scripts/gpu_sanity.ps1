$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_all.csv"
$OUT = "runs\gpu_sanity"
$EVAL_OUT = "outputs\gpu_sanity"

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

Run-Step "torch cuda info" {
    & $PY -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
}

Run-Step "3 epoch gpu finetune" {
    & $PY scripts\finetune.py `
        --manifest $MANIFEST `
        --max-files 8 `
        --samples-per-file 4 `
        --epochs 3 `
        --batch-size 8 `
        --dim 32 `
        --depth 1 `
        --device $DEVICE `
        --out-dir $OUT `
        --oob-fusion-type concat_oob
}

Run-Step "evaluate classifier" {
    & $PY scripts\evaluate.py `
        --manifest $MANIFEST `
        --checkpoint "$OUT\best.pt" `
        --max-files 8 `
        --samples-per-file 4 `
        --batch-size 8 `
        --dim 32 `
        --depth 1 `
        --device $DEVICE `
        --mode classifier `
        --out-dir "$EVAL_OUT\classifier"
}

Run-Step "evaluate prototype" {
    & $PY scripts\evaluate.py `
        --manifest $MANIFEST `
        --checkpoint "$OUT\best.pt" `
        --max-files 8 `
        --samples-per-file 4 `
        --batch-size 8 `
        --dim 32 `
        --depth 1 `
        --device $DEVICE `
        --mode prototype `
        --out-dir "$EVAL_OUT\prototype"
}

Write-Host ""
Write-Host "GPU sanity passed"

