"""
compare_boost_ablation.py
--------------------------
Two outputs from the ablation experiment:

  (1) fig_ablation.png  -- 6 individual lines (control x3, WT+boost x3)
                           success rate in Phase 3
      Saved to: articles/images/fig_ablation.png

  (2) fig6_new.png      -- improved fig6: dead neurons (top) + success rate
                           (bottom) as mean±std bands (3 seeds each)
      Saved to: articles/images/fig6.png  (overwrites)

Data source: results/boost_ablation/
  hist_ctrl_s{42,123,456}.npz   -- control (no-WT, no-boost)
  hist_wt_s{42,123,456}.npz     -- full method (WT ON, boost ON)

Usage:
  uv run python compare_boost_ablation.py
  uv run python compare_boost_ablation.py --window 30 --no-overwrite
"""

import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [42, 123, 456]
DATA_DIR = pathlib.Path("results/boost_ablation")
OUT_ABLATION = pathlib.Path("articles/images/fig_ablation.png")
OUT_FIG6     = pathlib.Path("articles/images/fig6.png")

COLOR_WT_SEEDS   = ["#1976D2", "#42A5F5", "#90CAF9"]   # blue family
COLOR_CTRL_SEEDS = ["#D32F2F", "#EF5350", "#FFCDD2"]   # red family
COLOR_WT_MEAN   = "#1565C0"
COLOR_CTRL_MEAN = "#B71C1C"


def load(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def rolling(arr, window):
    return np.convolve(arr.astype(float), np.ones(window) / window, mode="valid")


def total_dead(d):
    return (d["n_dead_hidden1"] + d["n_dead_hidden2"] + d["n_dead_hidden3"]).astype(float)


def load_seeds(prefix, seeds):
    """Return list of dicts, one per seed."""
    results = []
    for s in seeds:
        path = DATA_DIR / f"hist_{prefix}_s{s}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Run '.\\run_boost_ablation.ps1' first to generate training data."
            )
        results.append(load(str(path)))
    return results


# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation(wt_list, ctrl_list, window, out_path):
    """
    Figure: 6 individual success-rate curves (Phase-3 only).
    3 blue lines = WT+boost, 3 red lines = control (no-WT, no-boost).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (d, color) in enumerate(zip(ctrl_list, COLOR_CTRL_SEEDS)):
        roll = rolling(d["success"], window) * 100
        x = np.arange(window - 1, window - 1 + len(roll))
        lw = 1.8 if i == 0 else 1.2
        label = f"Control seed{SEEDS[i]}" if i == 0 else f"seed{SEEDS[i]}"
        ax.plot(x, roll, color=color, linewidth=lw,
                linestyle="--", alpha=0.9, label=label)

    for i, (d, color) in enumerate(zip(wt_list, COLOR_WT_SEEDS)):
        roll = rolling(d["success"], window) * 100
        x = np.arange(window - 1, window - 1 + len(roll))
        lw = 1.8 if i == 0 else 1.2
        label = f"WT+Boost seed{SEEDS[i]}" if i == 0 else f"seed{SEEDS[i]}"
        ax.plot(x, roll, color=color, linewidth=lw, alpha=0.9, label=label)

    # Legend with group labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=9, ncol=2)

    ax.set_xlabel("Episode (Phase 3)", fontsize=11)
    ax.set_ylabel(f"Rolling success rate ({window}-ep window, %)", fontsize=11)
    ax.set_ylim(0, 55)
    ax.set_title(
        "Ablation: Control (no-WT, no-boost) vs. Wine-Tower + STDP Boost  [3 seeds each]",
        fontsize=11)
    ax.grid(True, alpha=0.3)

    # Dashed separator legend entries
    from matplotlib.lines import Line2D
    custom = [
        Line2D([0], [0], color=COLOR_CTRL_SEEDS[0], linewidth=2, linestyle="--",
               label="Control (no-WT, no-boost)"),
        Line2D([0], [0], color=COLOR_WT_SEEDS[0],   linewidth=2,
               label="Wine-Tower + STDP Boost"),
    ]
    ax.legend(handles=custom + handles, labels=["Control (no-WT, no-boost)",
                                                 "Wine-Tower + STDP Boost"] + labels,
              fontsize=8, ncol=2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_fig6(wt_list, ctrl_list, window, out_path):
    """
    Improved fig6: two-panel figure with mean±std bands.
    Top  : dead hidden neurons (H1+H2+H3) over Phase-3 episodes
    Bottom: rolling success rate over Phase-3 episodes
    """
    def align(arrays):
        min_len = min(len(a) for a in arrays)
        return np.stack([a[:min_len] for a in arrays])

    dead_wt   = align([total_dead(d) for d in wt_list])
    dead_ctrl = align([total_dead(d) for d in ctrl_list])

    succ_wt   = align([d["success"].astype(float) for d in wt_list])
    succ_ctrl = align([d["success"].astype(float) for d in ctrl_list])

    T = dead_wt.shape[1]
    episodes = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={"hspace": 0.35})

    # ── Top panel: dead neurons ───────────────────────────────────
    for arr, color, label in [
        (dead_ctrl, COLOR_CTRL_MEAN, "Control (no-WT, no-boost)"),
        (dead_wt,   COLOR_WT_MEAN,   "Wine-Tower + STDP Boost"),
    ]:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax1.plot(episodes, mean, color=color, linewidth=2, label=label)
        ax1.fill_between(episodes, mean - std, mean + std,
                         color=color, alpha=0.15)

    ax1.set_ylabel("Total dead hidden neurons (H1+H2+H3)", fontsize=11)
    ax1.set_title(
        f"Dead Neuron Count: mean ± std  (seeds {SEEDS})", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Annotate final means
    for arr, color in [(dead_wt, COLOR_WT_MEAN), (dead_ctrl, COLOR_CTRL_MEAN)]:
        final = arr.mean(axis=0)[-1]
        ax1.annotate(f"end={final:.1f}",
                     xy=(episodes[-1], final),
                     xytext=(-65, 8), textcoords="offset points",
                     color=color, fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=color))

    # ── Bottom panel: success rate ────────────────────────────────
    for arr, color, label in [
        (succ_ctrl, COLOR_CTRL_MEAN, "Control (no-WT, no-boost)"),
        (succ_wt,   COLOR_WT_MEAN,   "Wine-Tower + STDP Boost"),
    ]:
        rolls = np.stack([rolling(arr[i], window) * 100 for i in range(len(SEEDS))])
        mean  = rolls.mean(axis=0)
        std   = rolls.std(axis=0)
        x     = np.arange(window - 1, window - 1 + len(mean))
        ax2.plot(x, mean, color=color, linewidth=2, label=label)
        ax2.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax2.set_xlabel("Episode (Phase 3)", fontsize=11)
    ax2.set_ylabel(f"Rolling success rate ({window}-ep window, %)", fontsize=11)
    ax2.set_title("Success Rate: mean ± std", fontsize=11)
    ax2.set_ylim(0, 55)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Wine-Tower + STDP Boost vs. Control  ·  Phase 3 (all-10-goals recall)\n"
        f"n = {len(SEEDS)} seeds each",
        fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=50,
                        help="Rolling window size for success rate")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Skip saving fig6.png if it already exists")
    args = parser.parse_args()

    wt_list   = load_seeds("wt",   SEEDS)
    ctrl_list = load_seeds("ctrl", SEEDS)

    plot_ablation(wt_list, ctrl_list, args.window, OUT_ABLATION)

    if args.no_overwrite and OUT_FIG6.exists():
        print(f"Skipped (--no-overwrite): {OUT_FIG6}")
    else:
        plot_fig6(wt_list, ctrl_list, args.window, OUT_FIG6)


if __name__ == "__main__":
    main()
