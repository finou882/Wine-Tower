"""
Before / After Wine-Tower raster plot comparison.

(A) Before: models/no_wt_deep3_model.npz  -- trained WITHOUT Wine-Tower (Phase3)
(B) After:  models/wt_deep3_model.npz     -- trained WITH    Wine-Tower (Phase3)

Output: articles/images/fig_raster.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.snn_agent.agent import SNNAgent
from src.snn_agent.environment import MultipleTMaze


def collect_spikes(agent, env, n_env_steps=8):
    """エージェントを実行してスパイク履歴を収集する。"""
    agent.reset_episode()
    obs = env.reset(active_goals=[0, 1, 2])
    n_encode = agent.encode_steps

    history = {'input': [], 'h1': [], 'h2': [], 'h3': []}

    for _ in range(n_env_steps):
        for _ in range(n_encode):
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

            history['input'].append(in_spikes.copy())
            history['h1'].append(h1_spikes.copy())
            history['h2'].append(h2_spikes.copy())
            history['h3'].append(h3_spikes.copy())

            agent._last_h_spikes[0] = h1_spikes
            agent._last_h_spikes[1] = h2_spikes
            agent._last_h_spikes[2] = h3_spikes

        obs, _, done, _ = env.step(int(agent.rng.integers(3)))
        if done:
            break

    return {k: np.array(v) for k, v in history.items()}, n_encode


def plot_raster_column(axes, history, n_encode, n_env_steps, title, dead_masks=None):
    layers  = ['input', 'h1', 'h2', 'h3']
    labels  = ['Input (Cues)', 'Layer H1', 'Layer H2', 'Layer H3']
    colors  = ['#607D8B', '#EF5350', '#43A047', '#1E88E5']

    for i, (key, ax) in enumerate(zip(layers, axes)):
        data = history[key]
        t_idx, n_idx = np.where(data)

        # dead neuronを赤でハイライト（h1/h2/h3のみ）
        if dead_masks is not None and i >= 1:
            dm = dead_masks[i - 1]   # h1→0, h2→1, h3→2
            alive_mask = ~dm[n_idx]
            dead_mask_pts = dm[n_idx]
            if alive_mask.any():
                ax.scatter(t_idx[alive_mask], n_idx[alive_mask],
                           s=12, color=colors[i], marker='|', alpha=0.8)
            if dead_mask_pts.any():
                ax.scatter(t_idx[dead_mask_pts], n_idx[dead_mask_pts],
                           s=12, color='#FF6F00', marker='|', alpha=0.9,
                           label='Dead (revived)' if key == 'h1' else None)
        else:
            ax.scatter(t_idx, n_idx, s=12, color=colors[i], marker='|', alpha=0.8)

        ax.set_ylabel(labels[i], fontsize=8)
        ax.set_ylim(-1, data.shape[1])
        ax.tick_params(labelsize=7)

        # 環境ステップ境界の縦線
        for step in range(1, n_env_steps):
            ax.axvline(step * n_encode, color='gray', alpha=0.25,
                       linestyle='--', linewidth=0.8)

    axes[0].set_title(title, fontsize=10, fontweight='bold', pad=4)
    axes[-1].set_xlabel("Time (Simulation Time Steps / ms)", fontsize=8)


def main(out="articles/images/fig_raster.png"):
    env = MultipleTMaze(n_goals=5)

    # ── (A) Before: no Wine-Tower model ──────────────────────────────
    agent_nowt = SNNAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                          n_hidden=64, seed=42)
    agent_nowt.load("models/no_wt_deep3_model.npz")
    dead_nowt = [h.dead_mask.copy() for h in agent_nowt.hiddens]
    hist_nowt, n_enc = collect_spikes(agent_nowt, env)

    # ── (B) After: Wine-Tower model ───────────────────────────────────
    agent_wt = SNNAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        n_hidden=64, seed=42)
    agent_wt.load("models/wt_deep3_model.npz")
    dead_wt = [h.dead_mask.copy() for h in agent_wt.hiddens]
    hist_wt, _ = collect_spikes(agent_wt, env)

    n_env_steps = 8

    # ── プロット ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Spike Raster Plot: Before vs. After Wine-Tower\n"
        "(Orange ticks = neurons active at recording time that were previously dead)",
        fontsize=11, y=0.98
    )

    gs = gridspec.GridSpec(4, 2, hspace=0.55, wspace=0.35,
                           left=0.08, right=0.97, top=0.91, bottom=0.07)

    axes_nowt = [fig.add_subplot(gs[r, 0]) for r in range(4)]
    axes_wt   = [fig.add_subplot(gs[r, 1]) for r in range(4)]

    # 共通のx軸スケール
    total_steps = n_env_steps * n_enc
    for ax in axes_nowt + axes_wt:
        ax.set_xlim(0, total_steps)

    nowt_dead_total = sum(dm.sum() for dm in dead_nowt)
    wt_dead_total   = sum(dm.sum() for dm in dead_wt)

    plot_raster_column(
        axes_nowt, hist_nowt, n_enc, n_env_steps,
        f"(A) Before Wine-Tower  [dead hidden = {nowt_dead_total}]",
        dead_masks=dead_nowt,
    )
    plot_raster_column(
        axes_wt, hist_wt, n_enc, n_env_steps,
        f"(B) After Wine-Tower  [dead hidden = {wt_dead_total}]",
        dead_masks=dead_wt,
    )

    # y軸ラベルは左列のみ
    for ax in axes_wt:
        ax.set_ylabel("")

    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
