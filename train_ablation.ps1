# train_ablation.ps1
# Train WT or no-WT model using existing main.py options.
#
# Usage:
#   With Wine-Tower (default):
#     .\train_ablation.ps1
#     .\train_ablation.ps1 -WineTower
#
#   Without Wine-Tower (Before state for comparison):
#     .\train_ablation.ps1 -NoWineTower
#
#   Custom episode counts:
#     .\train_ablation.ps1 -NoWineTower -Phase12Ep 2000 -Phase3Ep 1200
#
# Output:
#   -WineTower   -> models/wt_deep3_ablation.npz
#   -NoWineTower -> models/no_wt_deep3_model.npz
#
# Note: main.py and trainer.py are NOT modified.
#       Uses --no-wine-tower / --max-phase / --start-phase options.

param(
    [switch]$WineTower,
    [switch]$NoWineTower,
    [int]$Phase12Ep = 1500,
    [int]$Phase3Ep  = 900,
    [int]$Seed      = 42,
    [int]$Hidden    = 64,
    [int]$WtaK      = 4,
    [float]$Lr      = 0.005
)

if ($WineTower -and $NoWineTower) {
    Write-Error "Cannot use both -WineTower and -NoWineTower at the same time."
    exit 1
}
if (-not $WineTower -and -not $NoWineTower) {
    $WineTower = $true
}

$phase2Model = "models/phase2_ablation_seed${Seed}.npz"
if ($WineTower) {
    $finalModel = "models/wt_deep3_ablation.npz"
    $wtFlag     = ""
    $label      = "WITH Wine-Tower"
} else {
    $finalModel = "models/no_wt_deep3_model.npz"
    $wtFlag     = "--no-wine-tower"
    $label      = "NO Wine-Tower"
}

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Ablation Train: $label" -ForegroundColor Cyan
Write-Host "  hidden=$Hidden  wta-k=$WtaK  lr=$Lr  seed=$Seed" -ForegroundColor Cyan
Write-Host "  Phase1/2: $Phase12Ep ep  Phase3: $Phase3Ep ep" -ForegroundColor Cyan
Write-Host "  Output  : $finalModel" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "[Step 1/2] Phase 1/2 training $Phase12Ep episodes..." -ForegroundColor Yellow
uv run python main.py --episodes $Phase12Ep --hidden $Hidden --wta-k $WtaK --lr $Lr --seed $Seed --max-phase 2 --save-model $phase2Model

if ($LASTEXITCODE -ne 0) {
    Write-Error "Phase 1/2 training failed."
    exit 1
}
Write-Host "Phase 1/2 done. Saved: $phase2Model" -ForegroundColor Green

Write-Host ""
Write-Host "[Step 2/2] Phase 3 training $Phase3Ep episodes - $label" -ForegroundColor Yellow

if ($WineTower) {
    uv run python main.py --episodes $Phase3Ep --hidden $Hidden --wta-k $WtaK --lr $Lr --seed $Seed --start-phase 3 --load-model $phase2Model --save-model $finalModel
} else {
    uv run python main.py --episodes $Phase3Ep --hidden $Hidden --wta-k $WtaK --lr $Lr --seed $Seed --start-phase 3 --load-model $phase2Model --save-model $finalModel --no-wine-tower
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Phase 3 training failed."
    exit 1
}
Write-Host ""
Write-Host "Done! Model saved: $finalModel" -ForegroundColor Green