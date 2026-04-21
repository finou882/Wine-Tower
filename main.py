"""
Multiple-TMaze SNN Lifelong Learning – Main Entry Point
========================================================

Usage examples
--------------
# Default training (1000 episodes, curriculum, 3→10 goals):
    uv run python main.py

# Fix specific goals (bypass curriculum):
    uv run python main.py --fixed-goals 0 2 5 7

# All 10 goals fixed from start:
    uv run python main.py --fixed-goals 0 1 2 3 4 5 6 7 8 9

# Custom hyperparameters:
    uv run python main.py --episodes 2000 --hidden 128 --wta-k 6 --lr 0.003

# Save learning curve plot:
    uv run python main.py --plot results.png
"""

import argparse
import sys
import numpy as np

from src.snn_agent.environment import MultipleTMaze, N_GOALS
from src.snn_agent.agent import SNNAgent
from src.snn_agent.curriculum import GoalCurriculum
from src.snn_agent.trainer import LifelongTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LIF+STDP SNN agent on Multiple-TMaze with recursive lifelong learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Training budget ──────────────────────────────────────────────
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Total training episodes.",
    )

    # ── Goal options ─────────────────────────────────────────────────
    parser.add_argument(
        "--fixed-goals", type=int, nargs="+", default=None,
        metavar="GOAL_ID",
        help=(
            "Fix the goal set for every episode (bypasses curriculum). "
            f"Provide 1-{N_GOALS} goal IDs from 0 to {N_GOALS-1}. "
            "E.g. --fixed-goals 0 2 5"
        ),
    )
    parser.add_argument(
        "--n-hint-goals", type=int, default=3,
        help="Number of goal cues shown per episode in curriculum phase 1/2.",
    )
    parser.add_argument(
        "--anchor-goal", type=int, default=None,
        metavar="GOAL_ID",
        help=(
            "Always include this goal ID in the hint cue set every episode. "
            "The remaining n_hint-1 slots are filled randomly. "
            "Ignored when --fixed-goals is used."
        ),
    )
    parser.add_argument(
        "--max-gap", type=int, default=50,
        help="Max look-ahead gap (in episodes) for phase-2 memory curriculum.",
    )

    # ── Network hyperparameters ───────────────────────────────────────
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden layer size.")
    parser.add_argument("--wta-k", type=int, default=4,
                        help="WTA winners per timestep.")
    parser.add_argument("--encode-steps", type=int, default=10,
                        help="Poisson encoding steps per observation.")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="STDP base learning rate.")
    parser.add_argument("--recurrent", action="store_true", default=True,
                        help="Enable recurrent hidden→hidden synapses.")
    parser.add_argument("--no-recurrent", dest="recurrent", action="store_false",
                        help="Disable recurrent hidden→hidden synapses.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # ── Training options ──────────────────────────────────────────────
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode.")
    parser.add_argument("--replay-interval", type=int, default=20,
                        help="Run replay consolidation every N episodes.")
    parser.add_argument("--verbose-every", type=int, default=50,
                        help="Print stats every N episodes.")

    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument("--plot", type=str, default=None,
                        metavar="FILE.png",
                        help="Save learning curve to this file.")
    parser.add_argument("--weight-plot", type=str, default=None,
                        metavar="FILE.png",
                        help="Save weight heatmaps (W_in, W_out, W_rec) to this file.")
    parser.add_argument("--save-model", type=str, default=None,
                        metavar="FILE.npz",
                        help="Save trained model weights to this file after training.")
    parser.add_argument("--load-model", type=str, default=None,
                        metavar="FILE.npz",
                        help="Load model weights from this file before training.")
    parser.add_argument("--start-phase", type=int, default=None,
                        choices=[1, 2, 3], metavar="PHASE",
                        help="Skip curriculum to this phase at training start (1/2/3).")
    parser.add_argument("--max-phase", type=int, default=3,
                        choices=[1, 2, 3], metavar="PHASE",
                        help="Stop training when curriculum advances past this phase.")
    parser.add_argument("--no-wine-tower", action="store_true",
                        help="Disable Wine-Tower recovery (for ablation/control experiments).")
    parser.add_argument("--save-history", type=str, default=None,
                        metavar="FILE.npz",
                        help="Save training history (dead neurons, acc, etc.) to this file.")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def validate_goals(fixed_goals, n_goals=N_GOALS):
    """Validate --fixed-goals argument."""
    if fixed_goals is None:
        return None
    for g in fixed_goals:
        if g < 0 or g >= n_goals:
            print(f"Error: goal ID {g} out of range [0, {n_goals-1}].",
                  file=sys.stderr)
            sys.exit(1)
    return sorted(set(fixed_goals))


def print_config(args, fixed_goals):
    print("=" * 60)
    print("  Multiple-TMaze  ·  LIF+STDP SNN  ·  Lifelong Learning")
    print("=" * 60)
    if fixed_goals is not None:
        print(f"  Mode         : FIXED goals {fixed_goals}")
    else:
        print(f"  Mode         : CURRICULUM  (3-hint → all-10 recall)")
        print(f"  Hint goals   : {args.n_hint_goals}")
        if args.anchor_goal is not None:
            print(f"  Anchor goal  : {args.anchor_goal} (always in cue)")
        print(f"  Max gap      : {args.max_gap} episodes")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Hidden size  : {args.hidden}  WTA-k={args.wta_k}")
    print(f"  Recurrent    : {args.recurrent}")
    print(f"  LR           : {args.lr}")
    print(f"  Encode steps : {args.encode_steps}")
    print(f"  Max steps/ep : {args.max_steps}")
    print(f"  Seed         : {args.seed}")
    print("=" * 60)


def plot_results(history, save_path):
    import matplotlib.pyplot as plt

    episodes = history["episode"]
    successes = history["success"]
    rewards = np.array(history["reward"])
    n_dead_h = history["n_dead_hidden"]
    phases = history["phase"]

    # Rolling success rate
    window = 50
    rolling_acc = np.convolve(successes,
                              np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    # --- Panel 1: success rate ---
    ax = axes[0]
    ax.plot(range(window - 1, len(episodes)), rolling_acc * 100,
            label=f"Rolling success ({window}-ep)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade phases
    phase_arr = np.array(phases)
    for ph, color in [(1, "#d4f0ff"), (2, "#fff3cd"), (3, "#d4edda")]:
        idx = np.where(phase_arr == ph)[0]
        if len(idx):
            ax.axvspan(idx[0], idx[-1], alpha=0.3, color=color,
                       label=f"Phase {ph}")
    ax.legend(loc="upper left", fontsize=8)

    # --- Panel 2: dead neurons ---
    ax = axes[1]
    ax.plot(episodes, n_dead_h, label="Dead hidden neurons", color="tomato")
    ax.plot(episodes, history["n_dead_output"],
            label="Dead output neurons", color="orange", linestyle="--")
    ax.set_ylabel("# Dead neurons")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: reward ---
    ax = axes[2]
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=0.5)
    if len(rewards) >= window:
        roll_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episodes)), roll_r,
                color="steelblue", label=f"Rolling reward ({window}-ep)")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("LIF+STDP SNN on Multiple-TMaze – Lifelong Learning", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────

def plot_weight_heatmaps(agent, save_path: str) -> None:
    """
    Plot heatmaps of W_in, W_out, and (optionally) W_rec.

    W_in  : (obs_dim, n_hidden)  – rows = input features, cols = hidden neurons
            sorted by L2-norm of each hidden neuron's weight vector
    W_out : (n_hidden, act_dim)  – rows = hidden neurons, cols = actions
    W_rec : (n_hidden, n_hidden) – recurrent connections (if present)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Input feature labels
    from src.snn_agent.environment import N_JUNCTIONS, N_GOALS
    input_labels = (
        [f"J{i}" for i in range(N_JUNCTIONS)]        # junction one-hot
        + [f"G{i}" for i in range(N_GOALS)]           # goal cue one-hot
        + ["CUE"]                                       # hint flag
    )
    action_labels = ["Left", "Right", "Fwd"]

    W_in  = agent.W_in.W.copy()    # (obs_dim, n_hidden)
    W_out = agent.W_out.W.copy()   # (n_hidden, act_dim)
    has_rec = agent.recurrent
    W_rec = agent.W_rec.W.copy() if has_rec else None

    # Sort hidden neurons by L2 norm of their W_in column (descending)
    norms = np.linalg.norm(W_in, axis=0)
    order = np.argsort(-norms)
    W_in_sorted  = W_in[:, order]
    W_out_sorted = W_out[order, :]
    dead_sorted  = agent.hidden.dead_mask[order]

    n_panels = 3 if has_rec else 2
    fig = plt.figure(figsize=(5 * n_panels, max(8, W_in.shape[1] // 6)))
    gs  = gridspec.GridSpec(1, n_panels, wspace=0.45)

    # ── Panel 1: W_in ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    vmax = np.abs(W_in_sorted).max()
    im1 = ax1.imshow(
        W_in_sorted.T,          # (n_hidden, obs_dim) → rows=hidden, cols=input
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax1.set_title(f"W_in A: weights_{timestamp}", fontsize=9)
    ax1.set_xlabel("sensor")
    ax1.set_ylabel("hidden neuron (norm order)")
    ax1.set_xticks(range(len(input_labels)))
    ax1.set_xticklabels(input_labels, rotation=60, ha="right", fontsize=7)
    fig.colorbar(im1, ax=ax1, shrink=0.6, label="weight")

    # Highlight dead neurons with a red tick on y-axis
    dead_yticks = np.where(dead_sorted)[0]
    for y in dead_yticks:
        ax1.axhline(y, color="red", linewidth=0.4, alpha=0.5)

    # ── Panel 2: W_out ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    vmax2 = np.abs(W_out_sorted).max()
    im2 = ax2.imshow(
        W_out_sorted,            # (n_hidden, act_dim)
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax2, vmax=vmax2,
        interpolation="nearest",
    )
    ax2.set_title(f"W_out: weights_{timestamp}", fontsize=9)
    ax2.set_xlabel("action")
    ax2.set_ylabel("hidden neuron (norm order)")
    ax2.set_xticks(range(len(action_labels)))
    ax2.set_xticklabels(action_labels, fontsize=8)
    fig.colorbar(im2, ax=ax2, shrink=0.6, label="weight")

    for y in dead_yticks:
        ax2.axhline(y, color="red", linewidth=0.4, alpha=0.5)

    # ── Panel 3: W_rec (optional) ────────────────────────────────────
    if has_rec:
        ax3 = fig.add_subplot(gs[2])
        W_rec_sorted = W_rec[np.ix_(order, order)]
        vmax3 = np.abs(W_rec_sorted).max()
        im3 = ax3.imshow(
            W_rec_sorted,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax3, vmax=vmax3,
            interpolation="nearest",
        )
        ax3.set_title(f"W_rec: weights_{timestamp}", fontsize=9)
        ax3.set_xlabel("hidden (pre, norm order)")
        ax3.set_ylabel("hidden (post, norm order)")
        fig.colorbar(im3, ax=ax3, shrink=0.6, label="weight")

        for y in dead_yticks:
            ax3.axhline(y, color="red", linewidth=0.4, alpha=0.4)
            ax3.axvline(y, color="red", linewidth=0.4, alpha=0.4)

    # Legend for dead neuron marker
    from matplotlib.lines import Line2D
    legend_el = [Line2D([0], [0], color="red", linewidth=1.5,
                         label=f"dead neuron ({dead_sorted.sum()}/{len(dead_sorted)})")]
    fig.legend(handles=legend_el, loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)

    plt.suptitle("SNN Weight Heatmaps  ·  LIF+STDP+WTA", y=1.01)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"Weight heatmap saved to: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    fixed_goals = validate_goals(args.fixed_goals)
    print_config(args, fixed_goals)

    np.random.seed(args.seed)

    # Build environment
    env = MultipleTMaze(n_goals=N_GOALS, fixed_goals=fixed_goals)

    # Build agent
    agent = SNNAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        n_hidden=args.hidden,
        wta_k=args.wta_k,
        encode_steps=args.encode_steps,
        recurrent=args.recurrent,
        lr=args.lr,
        seed=args.seed,
    )

    # Build curriculum
    curriculum = GoalCurriculum(
        total_episodes=args.episodes,
        n_hint_goals=args.n_hint_goals,
        max_gap=args.max_gap,
        fixed_goals=fixed_goals,
        anchor_goal=args.anchor_goal,
        seed=args.seed,
    )

    # Build trainer
    trainer = LifelongTrainer(
        agent=agent,
        env=env,
        curriculum=curriculum,
        max_steps=args.max_steps,
        replay_interval=args.replay_interval,
        verbose_every=args.verbose_every,
        wine_tower=not args.no_wine_tower,
    )

    # Load model if requested (before training)
    if args.load_model:
        agent.load(args.load_model)

    # Skip curriculum to requested phase
    if args.start_phase == 2:
        curriculum._episode = curriculum.phase1_end
    elif args.start_phase == 3:
        curriculum._episode = curriculum.phase2_end

    # Train
    history = trainer.train(total_episodes=args.episodes,
                            max_phase=args.max_phase)

    # Summary
    recent = history["success"][-100:]
    final_acc = np.mean(recent) * 100
    print(f"\nFinal accuracy (last 100 episodes): {final_acc:.1f}%")
    print(f"Dead hidden neurons at end: {agent.n_dead_hidden}/{args.hidden}")
    print(f"Dead output neurons at end: {agent.n_dead_output}/{env.act_dim}")

    if args.save_model:
        agent.save(args.save_model)

    if args.save_history:
        import pathlib
        pathlib.Path(args.save_history).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_history,
            episode=np.array(history["episode"]),
            success=np.array(history["success"]),
            reward=np.array(history["reward"]),
            n_dead_hidden=np.array(history["n_dead_hidden"]),
            n_dead_output=np.array(history["n_dead_output"]),
            phase=np.array(history["phase"]),
        )
        print(f"History saved to: {args.save_history}")

    if args.plot:
        plot_results(history, args.plot)

    if args.weight_plot:
        plot_weight_heatmaps(agent, args.weight_plot)


if __name__ == "__main__":
    main()
