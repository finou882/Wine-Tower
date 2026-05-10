"""
generate_attractor_heatmap.py
------------------------------
Visualize WTA attractor formation as a goal-condition firing-rate heatmap.

For each goal (0..N_GOALS-1), run N_EPISODES episodes with that goal fixed,
record average spike count per hidden neuron (H1/H2/H3), and plot:

  rows   = hidden neurons (sorted by overall norm, same order as fig1)
  cols   = goal condition (0..4)
  color  = mean firing rate across episodes

If attractor is strong: every column looks the same (same neurons always win).
If goal-specific: each column has a different pattern.

Usage:
    uv run python generate_attractor_heatmap.py
    uv run python generate_attractor_heatmap.py --model models/phase2_fig1_3000ep.npz --episodes 50
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.snn_agent.environment import MultipleTMaze, N_GOALS
from src.snn_agent.agent import SNNAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/phase2_fig1_3000ep.npz",
                   help="Phase 1/2 checkpoint to load.")
    p.add_argument("--episodes", type=int, default=50,
                   help="Episodes per goal condition.")
    p.add_argument("--max-steps", type=int, default=50,
                   help="Max steps per episode.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--wta-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="articles/images/fig_attractor.png",
                   help="Output PNG path.")
    return p.parse_args()


def collect_firing_rates(agent, env, goal_id, n_episodes, max_steps, rng):
    """Run n_episodes with fixed goal_id, return mean spike count per neuron."""
    n_h = agent.n_hidden
    accum = [np.zeros(n_h), np.zeros(n_h), np.zeros(n_h)]  # H1, H2, H3
    total_steps = 0

    for _ in range(n_episodes):
        obs = env.reset(active_goals=[goal_id])
        agent.reset_episode()
        for _ in range(max_steps):
            # Monkey-patch act to also record hidden spikes
            spikes = _act_and_record(agent, obs)
            for layer_idx in range(3):
                accum[layer_idx] += spikes[layer_idx]
            total_steps += 1
            action = int(np.argmax(agent._out_rates)) if agent._out_rates.sum() > 0 \
                else int(rng.integers(agent.act_dim))
            obs, _, done, _ = env.step(action)
            if done:
                break

    if total_steps == 0:
        total_steps = 1
    return [a / total_steps for a in accum]


def _act_and_record(agent, obs):
    """Run one act() call (no learning) and return spike sums for H1/H2/H3."""
    from src.snn_agent.environment import N_GOALS
    agent._out_rates[:] = 0.0
    tonic_bias_out = np.ones(agent.act_dim) * 0.02

    spike_accum = [np.zeros(agent.n_hidden) for _ in range(3)]

    for _ in range(agent.encode_steps):
        in_spikes = agent._encode(obs)

        I_h1 = agent.W_in.forward(in_spikes)
        if agent.recurrent:
            I_h1 += agent.W_rec1.forward(agent._last_h_spikes[0])
        h1_spikes = agent.hidden1.step(I_h1)

        I_h2 = agent.W_12.forward(h1_spikes)
        if agent.recurrent:
            I_h2 += agent.W_rec2.forward(agent._last_h_spikes[1])
        h2_spikes = agent.hidden2.step(I_h2)

        I_h3 = agent.W_23.forward(h2_spikes)
        if agent.recurrent:
            I_h3 += agent.W_rec3.forward(agent._last_h_spikes[2])
        h3_spikes = agent.hidden3.step(I_h3)

        I_out = agent.W_out.forward(h3_spikes) + tonic_bias_out
        out_spikes = agent.output.step(I_out)
        agent._out_rates += out_spikes.astype(float)

        spike_accum[0] += h1_spikes.astype(float)
        spike_accum[1] += h2_spikes.astype(float)
        spike_accum[2] += h3_spikes.astype(float)

        agent._last_h_spikes[0] = h1_spikes
        agent._last_h_spikes[1] = h2_spikes
        agent._last_h_spikes[2] = h3_spikes

    return spike_accum


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env = MultipleTMaze(n_goals=N_GOALS)
    agent = SNNAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        n_hidden=args.hidden,
        wta_k=args.wta_k,
        seed=args.seed,
    )
    agent.load(args.model)
    print(f"Loaded: {args.model}")
    print(f"Dead neurons: H1={agent.n_dead_hidden1}, H2={agent.n_dead_hidden2}, H3={agent.n_dead_hidden3}")

    # Collect firing rates per goal
    rates = {layer: [] for layer in range(3)}  # layer -> list of (n_h,) arrays
    for goal in range(N_GOALS):
        print(f"  Goal {goal} ...", end=" ", flush=True)
        fr = collect_firing_rates(agent, env, goal, args.episodes, args.max_steps, rng)
        for layer in range(3):
            rates[layer].append(fr[layer])
        print("done")

    # Stack: shape (N_GOALS, n_h) per layer
    mats = [np.stack(rates[l], axis=0) for l in range(3)]  # (N_GOALS, n_h)

    # Sort neurons by overall firing rate (descending) — same spirit as fig1
    layer_names = ["H1", "H2", "H3"]
    dead_masks = [agent.hidden1.dead_mask, agent.hidden2.dead_mask, agent.hidden3.dead_mask]

    fig = plt.figure(figsize=(7 * 3, 14))
    gs = gridspec.GridSpec(1, 3, wspace=0.5)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.titlesize": 16,
    })

    for li in range(3):
        mat = mats[li]  # (N_GOALS, n_h)
        order = np.argsort(-mat.mean(axis=0))  # sort by mean rate desc
        mat_s = mat[:, order]
        dead_s = dead_masks[li][order]

        ax = fig.add_subplot(gs[li])
        vmax = mat_s.max() if mat_s.max() > 0 else 1.0
        im = ax.imshow(mat_s.T, aspect="auto", cmap="hot",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{layer_names[li]}  firing rate")
        ax.set_xlabel("goal condition")
        ax.set_ylabel("neuron (mean-rate order)")
        ax.set_xticks(range(N_GOALS))
        ax.set_xticklabels([f"G{g}" for g in range(N_GOALS)])
        fig.colorbar(im, ax=ax, shrink=0.6, label="mean spike/step")

        # Cyan lines for dead neurons
        for y in np.where(dead_s)[0]:
            ax.axhline(y, color="cyan", linewidth=0.6, alpha=0.6)

    n_dead = agent.n_dead_hidden1 + agent.n_dead_hidden2 + agent.n_dead_hidden3
    plt.suptitle(
        f"WTA Attractor Heatmap — Phase 1/2 model ({args.episodes} ep/goal)\n"
        f"Uniform columns → same neurons win regardless of goal (attractor). "
        f"Dead: {n_dead}/192",
        y=1.02
    )
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
