# run_seeds.ps1
# Phase2モデルを3seedで学習後、WTあり・なしの対照実験を各seed実行する

$seeds = @(42, 123, 456)

foreach ($seed in $seeds) {
    Write-Host "=== Seed $seed : Phase1/2 training ===" -ForegroundColor Cyan
    uv run python main.py --anchor-goal 5 --episodes 3000 --hidden 128 --wta-k 16 --lr 0.005 --max-phase 2 --seed $seed --save-model "models/phase2_seed$seed.npz"

    Write-Host "=== Seed $seed : Phase3 WITH Wine-Tower ===" -ForegroundColor Green
    uv run python main.py --anchor-goal 5 --episodes 900 --hidden 128 --wta-k 16 --lr 0.005 --load-model "models/phase2_seed$seed.npz" --start-phase 3 --seed $seed --save-history "results/wt_seed$seed.npz"

    Write-Host "=== Seed $seed : Phase3 NO Wine-Tower ===" -ForegroundColor Red
    uv run python main.py --anchor-goal 5 --episodes 900 --hidden 128 --wta-k 16 --lr 0.005 --load-model "models/phase2_seed$seed.npz" --start-phase 3 --seed $seed --no-wine-tower --save-history "results/no_wt_seed$seed.npz"
}

Write-Host "=== All seeds done! ===" -ForegroundColor Yellow
uv run python compare_winetower.py --multi-seed --out comparison2.png
