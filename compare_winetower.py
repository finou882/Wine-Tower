"""
compare_winetower.py
--------------------
Plot a side-by-side (or overlaid) comparison of:
  - Wine-Tower enabled  (recovery condition)
  - Wine-Tower disabled (control condition)

Single-seed usage
-----------------
  uv run python compare_winetower.py results/wt.npz results/no_wt.npz --out comparison.png

Multi-seed usage (mean ± std across seeds 42, 123, 456)
---------------------------------------------------------
  uv run python compare_winetower.py --multi-seed --out comparison_multiseed.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SEEDS = [42, 123, 456]
COLOR_WT   = "#2196F3"   # blue
COLOR_NOWT = "#F44336"   # red


def load(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def rolling(arr, window=50):
    return np.convolve(arr.astype(float), np.ones(window) / window, mode="valid")


def stack_seeds(prefix: str, seeds, key: str) -> np.ndarray:
    """Load key from multiple seed files and return (n_seeds, T) array."""
    arrays = []
    for s in seeds:
        path = f"results/{prefix}_seed{s}.npz"
        d = load(path)
        if key == "n_dead_hidden":
            # Sum up all 3 layers
            val = (d["n_dead_hidden1"] + d["n_dead_hidden2"] + d["n_dead_hidden3"]).astype(float)
        else:
            val = d[key].astype(float)
        arrays.append(val)
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays])   # (n_seeds, T)


def plot_single(args):
    wt    = load(args.wt_file)
    no_wt = load(args.no_wt_file)
    W = args.window

    ep_wt    = wt["episode"]
    ep_no_wt = no_wt["episode"]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    # Sum up dead neurons from all 3 hidden layers
    def get_total_dead(data):
        return (data["n_dead_hidden1"] + data["n_dead_hidden2"] + data["n_dead_hidden3"])

    ax1.plot(ep_no_wt, get_total_dead(no_wt),
             color=COLOR_NOWT, linewidth=1.2, label="No Wine-Tower (control, seed42)")
    ax1.plot(ep_wt, get_total_dead(wt),
             color=COLOR_WT,   linewidth=1.8, label="Wine-Tower seed42", zorder=3)

    # Extra WT runs (other seeds)
    extra_colors = ["#64B5F6", "#1565C0", "#4FC3F7", "#0288D1"]
    for i, path in enumerate(args.extra_wt or []):
        try:
            ex = load(path)
            label = path.split("/")[-1].replace(".npz", "")
            ax1.plot(ex["episode"], get_total_dead(ex),
                     color=extra_colors[i % len(extra_colors)],
                     linewidth=1.2, linestyle="--", alpha=0.8,
                     label=f"Wine-Tower {label}")
        except (FileNotFoundError, KeyError):
            print(f"Warning: {path} not found or missing keys, skipping.")

    ax1.set_ylabel("Total dead hidden neurons (H1+H2+H3)")
    ax1.set_xlabel("Episode")
    ax1.set_title("Dead Neuron Count: Wine-Tower vs Control", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    final_wt   = int(get_total_dead(wt)[-1])
    final_nowt = int(get_total_dead(no_wt)[-1])
    ax1.annotate(f"end={final_wt}",
                 xy=(ep_wt[-1], final_wt),
                 xytext=(-60, 8), textcoords="offset points",
                 color=COLOR_WT, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=COLOR_WT))
    ax1.annotate(f"end={final_nowt}",
                 xy=(ep_no_wt[-1], final_nowt),
                 xytext=(-60, 8), textcoords="offset points",
                 color=COLOR_NOWT, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=COLOR_NOWT))

    ax2 = fig.add_subplot(gs[1, 0])
    roll_wt = rolling(wt["success"], W) * 100
    ax2.plot(range(W - 1, len(ep_wt)), roll_wt, color=COLOR_WT, linewidth=1.2)
    ax2.set_title("Success rate – Wine-Tower", fontsize=9)
    ax2.set_ylabel(f"Rolling acc ({W}-ep) %")
    ax2.set_xlabel("Episode"); ax2.set_ylim(0, 50); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    roll_no = rolling(no_wt["success"], W) * 100
    ax3.plot(range(W - 1, len(ep_no_wt)), roll_no, color=COLOR_NOWT, linewidth=1.2)
    ax3.set_title("Success rate – No Wine-Tower (control)", fontsize=9)
    ax3.set_ylabel(f"Rolling acc ({W}-ep) %")
    ax3.set_xlabel("Episode"); ax3.set_ylim(0, 50); ax3.grid(True, alpha=0.3)

    plt.suptitle(
        "Wine-Tower Recovery in WTA-SNN  ·  Phase 3 (all-10-goals recall)",
        fontsize=12, y=1.01)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved to: {args.out}")


def plot_multi_seed(args):
    W = args.window

    dead_wt   = stack_seeds("wt",    SEEDS, "n_dead_hidden")
    dead_no   = stack_seeds("no_wt", SEEDS, "n_dead_hidden")
    succ_wt   = stack_seeds("wt",    SEEDS, "success")
    succ_no   = stack_seeds("no_wt", SEEDS, "success")

    T = dead_wt.shape[1]
    episodes = np.arange(T)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ── Panel 1: Dead neurons mean ± std ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for arr, color, label in [
        (dead_no, COLOR_NOWT, "No Wine-Tower (control)"),
        (dead_wt, COLOR_WT,   "Wine-Tower (recovery)"),
    ]:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax1.plot(episodes, mean, color=color, linewidth=1.5, label=label)
        ax1.fill_between(episodes, mean - std, mean + std,
                         color=color, alpha=0.18)
    ax1.set_ylabel("# Dead hidden neurons")
    ax1.set_xlabel("Episode (Phase 3)")
    ax1.set_title(f"Dead Neuron Count: mean ± std  (seeds {SEEDS})", fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Annotate final means
    for arr, color in [(dead_wt, COLOR_WT), (dead_no, COLOR_NOWT)]:
        final = arr.mean(axis=0)[-1]
        ax1.annotate(f"end={final:.1f}",
                     xy=(episodes[-1], final),
                     xytext=(-65, 8), textcoords="offset points",
                     color=color, fontsize=8,
                     arrowprops=dict(arrowstyle="->", color=color))

    # ── Panel 2/3: Rolling accuracy mean ± std ──────────────────────
    for ax, arr, color, title in [
        (fig.add_subplot(gs[1, 0]), succ_wt,   COLOR_WT,
         "Success rate – Wine-Tower"),
        (fig.add_subplot(gs[1, 1]), succ_no,   COLOR_NOWT,
         "Success rate – No Wine-Tower"),
    ]:
        rolls = np.stack([rolling(arr[i], W) * 100 for i in range(len(SEEDS))])
        mean  = rolls.mean(axis=0)
        std   = rolls.std(axis=0)
        x     = np.arange(W - 1, W - 1 + len(mean))
        ax.plot(x, mean, color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(f"Rolling acc ({W}-ep) %")
        ax.set_xlabel("Episode")
        ax.set_ylim(0, 50); ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Wine-Tower Recovery in WTA-SNN  ·  Phase 3  (n={len(SEEDS)} seeds)",
        fontsize=12, y=1.01)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Multi-seed comparison saved to: {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wt_file",  nargs="?", help="History npz WITH Wine-Tower (single-seed)")
    parser.add_argument("no_wt_file", nargs="?", help="History npz WITHOUT Wine-Tower (single-seed)")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Load results/wt_seedN.npz and results/no_wt_seedN.npz for all seeds")
    parser.add_argument("--extra-wt", nargs="+", metavar="FILE.npz",
                        help="Additional WT history files to overlay on the dead-neuron panel")
    parser.add_argument("--out", default="comparison_winetower.png")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    if args.multi_seed:
        plot_multi_seed(args)
    else:
        if not args.wt_file or not args.no_wt_file:
            parser.error("Provide wt_file and no_wt_file, or use --multi-seed")
        plot_single(args)


if __name__ == "__main__":
    main()

