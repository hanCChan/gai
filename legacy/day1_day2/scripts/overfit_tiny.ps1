$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$DEVICE = "cuda"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MANIFEST = "data\manifest_all.csv"
$OUT = "runs\overfit_tiny"

Set-Location $ROOT

Write-Host "==> overfit tiny sanity"
$Output = & $PY scripts\finetune.py `
    --manifest $MANIFEST `
    --max-files 4 `
    --samples-per-file 4 `
    --epochs 50 `
    --batch-size 4 `
    --dim 32 `
    --depth 1 `
    --device $DEVICE `
    --out-dir $OUT `
    --oob-fusion-type concat_oob `
    --use-chirp-embedding 2>&1
$Output | Write-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$FinalAcc = $null
foreach ($Line in $Output) {
    if ($Line -match "train_loss=.* acc=([0-9.]+)") {
        $FinalAcc = [double]$Matches[1]
    }
}
Write-Host "final_train_acc=$FinalAcc"
if ($null -ne $FinalAcc -and $FinalAcc -lt 0.9) {
    Write-Host "WARNING: final train acc is below 0.9; inspect data/model pipeline before formal experiments."
}

