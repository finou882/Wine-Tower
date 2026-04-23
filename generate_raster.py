
import numpy as np
import matplotlib.pyplot as plt
from src.snn_agent.agent import SNNAgent
from src.snn_agent.environment import MultipleTMaze

def generate_raster_plot(output_path="articles/images/fig_raster.png"):
    env = MultipleTMaze(n_goals=5)
    agent = SNNAgent(obs_dim=env.obs_dim, act_dim=env.act_dim, n_hidden=64)
    
    # CRITICAL: Must call reset_episode to initialize _last_h_spikes
    agent.reset_episode()
    
    obs = env.reset(active_goals=[0])
    
    spike_history = [[] for _ in range(3)] # H1, H2, H3
    
    for t in range(50):
        # We simulate manually to catch spikes within encode_steps if possible, 
        # or just the last step of act(). Let's just use act() and access state.
        _ = agent.act(obs, learn=False)
        for i in range(3):
            spike_history[i].append(agent._last_h_spikes[i].copy())
            
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        data = np.array(spike_history[i]) # (T, N)
        t_idx, n_idx = np.where(data)
        ax.scatter(t_idx, n_idx, s=5, color='black', marker='|')
        ax.set_ylabel(f"Layer H{i+1}")
        ax.set_ylim(-1, 64)
        if i == 0:
            ax.set_title("Spike Raster Plot (Activity across Deep SNN Layers)")
    
    axes[-1].set_xlabel("Time steps (Environment steps)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Raster plot saved to {output_path}")

if __name__ == "__main__":
    generate_raster_plot()
