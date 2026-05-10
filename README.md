[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19969376.svg)](https://doi.org/10.5281/zenodo.19969376)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finou882/wine-tower/blob/main/demo.ipynb)

> **日本語版 README は [README_ja.md](README_ja.md) を参照してください。**

---

# Wine-Tower — Solving Dead-Neuron Problem in Spiking Neural Networks

This repository implements the **Wine-Tower model**,
a biologically-inspired mechanism that revives dead (silent) neurons in WTA-LIF Spiking Neural Networks
via voltage diffusion through recurrent connections.

The agent is trained on a **Multiple T-Maze** lifelong-learning benchmark (5 goals, 5 junctions)
using R-STDP with a 3-layer deep WTA-LIF architecture.

Paper: *Wine-Tower: Solving the Dead-Neuron Problem in SNNs via Biological Voltage Diffusion* — Fumiaki INOUE, 2026.

### Key Findings
- **Structural Recovery**: Wine-Tower revives nearly all dead neurons by diffusing membrane potential through recurrent connections.
- **WTA Attractor Wall**: Despite structural recovery, the accuracy improvement is blocked by attractor basins formed during early specialization — a phenomenon interpreted as a *computational critical period*.
- **Structure–Function Gap**: Physical revival of neurons does not guarantee functional reintegration; the revived neurons are captured by the dominant WTA attractor and lose task-discriminative firing diversity.

---

## Requirements

```
Python >= 3.11
numpy
matplotlib
uv  (recommended)
```

Install dependencies:
```bash
# with uv (recommended)
uv sync

# or with pip
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Default training (Phase 1→3, seed 42, Wine-Tower ON)
uv run python main.py

# Save model and training history
uv run python main.py --save-model models/my_model.npz --save-history results/my_hist.npz

# Control: disable Wine-Tower and STDP boost
uv run python main.py --no-wine-tower --no-boost
```

---

## `main.py` Arguments

### Training Budget
| Argument | Default | Description |
|---|---|---|
| `--episodes` | 1000 | Total training episodes |

### Goal / Curriculum
| Argument | Default | Description |
|---|---|---|
| `--fixed-goals ID [ID ...]` | None | Fix goal set every episode (bypasses curriculum) |
| `--n-hint-goals` | 3 | Number of goal cues shown per episode in Phase 1/2 |
| `--anchor-goal ID` | None | Always include this goal in the hint cue set |
| `--max-gap` | 50 | Max look-ahead gap (episodes) for Phase-2 memory curriculum |

### Network Hyperparameters
| Argument | Default | Description |
|---|---|---|
| `--hidden` | 64 | Hidden layer size (each of 3 layers) |
| `--wta-k` | 4 | WTA winners per simulation time step |
| `--encode-steps` | 10 | Poisson encoding steps per observation |
| `--lr` | 0.005 | STDP base learning rate η |
| `--no-recurrent` | — | Disable recurrent hidden→hidden synapses |
| `--seed` | 42 | Random seed |

### Training Options
| Argument | Default | Description |
|---|---|---|
| `--max-steps` | 50 | Max environment steps per episode |
| `--replay-interval` | 20 | Run replay consolidation every N episodes |
| `--verbose-every` | 50 | Print stats every N episodes |
| `--start-phase` | None | Skip curriculum to phase 1/2/3 at start |
| `--max-phase` | 3 | Stop training when curriculum advances past this phase |
| `--no-wine-tower` | — | Disable Wine-Tower recovery (ablation) |
| `--no-boost` | — | Disable STDP reward amplification ×20/×2 (ablation) |

### I/O
| Argument | Default | Description |
|---|---|---|
| `--save-model FILE.npz` | None | Save trained weights after training |
| `--load-model FILE.npz` | None | Load weights before training |
| `--save-history FILE.npz` | None | Save per-episode stats (dead neurons, accuracy, …) |
| `--plot FILE.png` | None | Save learning curve plot |
| `--weight-plot FILE.png` | None | Save weight heatmap (W_in, W_out, W_rec) |

---

## Reproducing Paper Figures

### Figure 1 — Weight heatmap (Phase 1/2 monoculture)
```bash
uv run python main.py --max-phase 2 --save-model models/phase2.npz
uv run python main.py --load-model models/phase2.npz --weight-plot articles/images/fig1.png
```

### Figure 2 — Wine-Tower vs Control (3 seeds, mean ± std)
```bash
# Step 1: run ablation training (Phase1/2 checkpoint + Phase3 fork x2 x3seeds)
.\run_boost_ablation.ps1

# Step 2: generate figure
uv run python compare_boost_ablation.py
# output: articles/images/fig6.png
```

### Figure 3 — Spike raster plot (Before / After Wine-Tower)
```bash
# Step 1: train Phase 1/2 checkpoint (shared)
uv run python main.py --episodes 1500 --max-phase 2 --seed 42 --save-model models/phase2_raster.npz

# Step 2a: Phase 3 WITHOUT Wine-Tower  →  "Before" model
uv run python main.py --episodes 900 --seed 42 \
    --load-model models/phase2_raster.npz --start-phase 3 \
    --no-wine-tower --save-model models/no_wt_deep3_model.npz

# Step 2b: Phase 3 WITH Wine-Tower  →  "After" model
uv run python main.py --episodes 900 --seed 42 \
    --load-model models/phase2_raster.npz --start-phase 3 \
    --save-model models/wt_deep3_model.npz

# Step 3: generate raster plot
uv run python generate_raster_comparison.py
# output: articles/images/fig_raster.png
```

### Figure 4 — WTA Attractor heatmap (goal-conditioned firing rate)
```bash
# Requires Phase 1/2 trained model
uv run python generate_attractor_heatmap.py \
    --model models/phase2_fig1_3000ep.npz \
    --out articles/images/fig_attractor.png
```

### (Optional) Multi-seed comparison with existing results
```bash
uv run python compare_winetower.py --multi-seed --out comparison_multiseed.png
```

---

## Repository Structure

```
main.py                        # entry point
src/snn_agent/
    agent.py                   # 3-layer WTA-LIF agent, Wine-Tower replay
    trainer.py                 # training loop, curriculum, phase control
    lif.py                     # LIF neuron + dead-neuron detection
    wta.py                     # k-WTA inhibition layer
    stdp.py                    # R-STDP synaptic weight update
    environment.py             # Multiple T-Maze environment
    curriculum.py              # 3-phase goal curriculum
generate_raster_comparison.py  # fig3: before/after spike raster
generate_attractor_heatmap.py  # fig4: goal-conditioned firing rate heatmap
compare_boost_ablation.py      # fig2: WT+boost vs control (3 seeds)
compare_winetower.py           # multi-seed WT vs no-WT comparison
run_boost_ablation.ps1         # PowerShell: run ablation experiment
models/                        # pre-trained .npz model weights
results/                       # saved training histories
articles/                      # LaTeX paper source (Japanese)
articles/eng/                  # LaTeX paper source (English)
```

---

## License

MIT License. See [LICENSE](LICENSE).
