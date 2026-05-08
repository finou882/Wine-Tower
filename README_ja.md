[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19969376.svg)](https://doi.org/10.5281/zenodo.19969376)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finou882/wine-tower/blob/main/demo.ipynb)

> **English README is [here](README.md).**

---

# Wine-Tower (QRDM) — SNNのDead-Neuron問題の解決

本リポジトリは**Wine-Towerモデル (Quiescent-Recovery Diffusion Model, QRDM)** を実装します。
WTA-LIF スパイキングニューラルネットワークにおいて死滅（沈黙）したニューロンを、
再帰結合を通じた電位拡散で蘇生させる、生物学的に着想を得た手法です。

エージェントは3層深層WTA-LIFアーキテクチャとR-STDPを用いて、
**多段T字迷路（5ゴール・5分岐）** 生涯学習ベンチマークで訓練されます。

論文: *生物学的アプローチに基づいたLIFネットワークの電位拡散モデル (Wine-Tower) によるSpiking Neural NetworkのDead-Neuron問題の解決* — 井上文朗, 2026.

---

## 環境要件

```
Python >= 3.11
numpy
matplotlib
uv  (推奨)
```

依存パッケージのインストール:
```bash
# uv を使う場合（推奨）
uv sync

# pip を使う場合
pip install -r requirements.txt
```

---

## クイックスタート

```bash
# デフォルト学習 (Phase 1→3, seed 42, Wine-Tower ON)
uv run python main.py

# モデルと学習履歴を保存
uv run python main.py --save-model models/my_model.npz --save-history results/my_hist.npz

# コントロール条件: Wine-TowerとSTDPブーストを両方無効化
uv run python main.py --no-wine-tower --no-boost
```

---

## `main.py` 引数一覧

### 学習エピソード数
| 引数 | デフォルト | 説明 |
|---|---|---|
| `--episodes` | 1000 | 総学習エピソード数 |

### ゴール・カリキュラム
| 引数 | デフォルト | 説明 |
|---|---|---|
| `--fixed-goals ID [ID ...]` | None | 毎エピソード固定ゴールセットを使用（カリキュラム無効化） |
| `--n-hint-goals` | 3 | Phase 1/2 で各エピソードに提示するゴールキュー数 |
| `--anchor-goal ID` | None | 毎エピソード必ずキューに含めるゴールID |
| `--max-gap` | 50 | Phase 2 記憶カリキュラムの最大look-aheadギャップ（エピソード数） |

### ネットワークハイパーパラメータ
| 引数 | デフォルト | 説明 |
|---|---|---|
| `--hidden` | 64 | 隠れ層サイズ（3層各共通） |
| `--wta-k` | 4 | 1 simulation time step あたりのWTA勝者数 |
| `--encode-steps` | 10 | 1観測あたりのPoisson符号化ステップ数 |
| `--lr` | 0.005 | STDPベース学習率 η |
| `--no-recurrent` | — | 隠れ層→隠れ層の再帰結合を無効化 |
| `--seed` | 42 | 乱数シード |

### 学習オプション
| 引数 | デフォルト | 説明 |
|---|---|---|
| `--max-steps` | 50 | 1エピソードあたりの最大環境ステップ数 |
| `--replay-interval` | 20 | Nエピソードごとにリプレイ統合を実行 |
| `--verbose-every` | 50 | Nエピソードごとに統計を表示 |
| `--start-phase` | None | 学習開始時にカリキュラムをPhase 1/2/3にスキップ |
| `--max-phase` | 3 | カリキュラムがこのPhaseを超えたら学習を停止 |
| `--no-wine-tower` | — | Wine-Tower回復を無効化（アブレーション用） |
| `--no-boost` | — | STDPリワード増幅（×20/×2）を無効化（アブレーション用） |

### 入出力
| 引数 | デフォルト | 説明 |
|---|---|---|
| `--save-model FILE.npz` | None | 学習後に重みを保存 |
| `--load-model FILE.npz` | None | 学習前に重みを読み込み |
| `--save-history FILE.npz` | None | エピソードごとの統計（死滅ニューロン数・正答率など）を保存 |
| `--plot FILE.png` | None | 学習曲線プロットを保存 |
| `--weight-plot FILE.png` | None | 重みヒートマップ（W_in, W_out, W_rec）を保存 |

---

## 論文図の再現方法

### Figure 1 — 重みヒートマップ（Phase 1/2 モノカルチャー）
```bash
uv run python main.py --max-phase 2 --save-model models/phase2.npz
uv run python main.py --load-model models/phase2.npz --weight-plot articles/images/fig1.png
```

### Figure 2 — スパイクラスタープロット（Wine-Tower適用前後）
```bash
# Step 1: Phase 1/2 チェックポイントを学習（共通）
uv run python main.py --episodes 1500 --max-phase 2 --seed 42 --save-model models/phase2_raster.npz

# Step 2a: Phase 3（Wine-Tower無し）→ "Before" モデル
uv run python main.py --episodes 900 --seed 42 `
    --load-model models/phase2_raster.npz --start-phase 3 `
    --no-wine-tower --save-model models/no_wt_deep3_model.npz

# Step 2b: Phase 3（Wine-Tower有り）→ "After" モデル
uv run python main.py --episodes 900 --seed 42 `
    --load-model models/phase2_raster.npz --start-phase 3 `
    --save-model models/wt_deep3_model.npz

# Step 3: ラスタープロット生成
uv run python generate_raster_comparison.py
# 出力: articles/images/fig_raster.png
```

### Figure 3 — Wine-Tower vs コントロール（3シード、mean±std）
```bash
# Step 1: アブレーション学習を実行
#   (Phase1/2チェックポイント + Phase3を2条件×3シードに分岐)
.\run_boost_ablation.ps1

# Step 2: グラフ生成
uv run python compare_boost_ablation.py
# 出力: articles/images/fig6.png
```

---

## リポジトリ構成

```
main.py                        # エントリーポイント
src/snn_agent/
    agent.py                   # 3層WTA-LIFエージェント、Wine-Towerリプレイ
    trainer.py                 # 学習ループ、カリキュラム、フェーズ制御
    lif.py                     # LIFニューロン + 死滅ニューロン検出
    wta.py                     # k-WTA側抑制層
    stdp.py                    # R-STDP シナプス重み更新
    environment.py             # 多段T字迷路環境
    curriculum.py              # 3フェーズ・ゴールカリキュラム
generate_raster_comparison.py  # fig_raster: 適用前後のスパイクラスター
compare_boost_ablation.py      # fig6: WT+boost vs コントロール（3シード）
compare_winetower.py           # マルチシードWT vs No-WT比較
run_boost_ablation.ps1         # PowerShell: アブレーション実験実行
models/                        # 学習済み .npz モデル重み
results/                       # 保存済み学習履歴
articles/                      # LaTeX 論文ソース
```

---

## ライセンス

MIT License。[LICENSE](LICENSE) を参照。
