$ErrorActionPreference = "Stop"

$PY = "D:\miniconda\envs\llm4ee\python.exe"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Set-Location $ROOT

& $PY scripts\compare_predictions.py `
    --a-pred "outputs\osu_cnn_day1_day2_spf256\osu_cnn_iq\classifier\predictions.csv" `
    --b-pred "outputs\cross_day_hybrid_cnnstem_spf256\rfhstu_cnnstem_cross_attn_chirp\classifier\predictions.csv" `
    --a-name "osu_cnn_iq_classifier" `
    --b-name "hybrid_cnnstem_cross_attn_chirp_classifier" `
    --out-dir "outputs\analysis_cnn_vs_hybrid"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
