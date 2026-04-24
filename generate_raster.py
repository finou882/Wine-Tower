
import numpy as np
import matplotlib.pyplot as plt
from src.snn_agent.agent import SNNAgent
from src.snn_agent.environment import MultipleTMaze

def generate_improved_raster(output_path="articles/images/fig_raster.png"):
    env = MultipleTMaze(n_goals=5)
    agent = SNNAgent(obs_dim=env.obs_dim, act_dim=env.act_dim, n_hidden=64)
    agent.reset_episode()
    
    # Analyze just 10 environment steps (but all 100 internal micro-steps)
    n_env_steps = 10
    n_encode = agent.encode_steps
    
    # History for all micro-steps
    history = {
        'input': [],
        'h1': [],
        'h2': [],
        'h3': []
    }
    
    obs = env.reset(active_goals=[0])
    
    for _ in range(n_env_steps):
        # Manually run the encode loop that's usually inside agent.act()
        for _ in range(n_encode):
            in_spikes = agent._encode(obs)
            
            # H1
            I_h1 = agent.W_in.forward(in_spikes)
            if agent.recurrent: I_h1 += agent.W_rec1.forward(agent._last_h_spikes[0])
            h1_spikes = agent.hidden1.step(I_h1)
            
            # H2
            I_h2 = agent.W_12.forward(h1_spikes)
            if agent.recurrent: I_h2 += agent.W_rec2.forward(agent._last_h_spikes[1])
            h2_spikes = agent.hidden2.step(I_h2)
            
            # H3
            I_h3 = agent.W_23.forward(h2_spikes)
            if agent.recurrent: I_h3 += agent.W_rec3.forward(agent._last_h_spikes[2])
            h3_spikes = agent.hidden3.step(I_h3)
            
            # Record everything
            history['input'].append(in_spikes.copy())
            history['h1'].append(h1_spikes.copy())
            history['h2'].append(h2_spikes.copy())
            history['h3'].append(h3_spikes.copy())
            
            # Update recurrent state
            agent._last_h_spikes[0] = h1_spikes
            agent._last_h_spikes[1] = h2_spikes
            agent._last_h_spikes[2] = h3_spikes

        # Dummy act to advance env (not using for history)
        obs, _, done, _ = env.step(agent.rng.choice([0,1,2]))
        if done: break

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [1, 2, 2, 2]})
    
    layers = ['input', 'h1', 'h2', 'h3']
    labels = ['Input (Cues)', 'Layer H1', 'Layer H2', 'Layer H3']
    colors = ['gray', '#FF5252', '#4CAF50', '#2196F3']
    
    for i, (key, ax) in enumerate(zip(layers, axes)):
        data = np.array(history[key])
        t_idx, n_idx = np.where(data)
        ax.scatter(t_idx, n_idx, s=15, color=colors[i], marker='|', alpha=0.7)
        ax.set_ylabel(labels[i])
        
        # Add vertical lines to show environment step boundaries
        for step in range(1, n_env_steps):
            ax.axvline(step * n_encode, color='black', alpha=0.1, linestyle='--')
            
    axes[0].set_title("Detailed Spike Raster Plot (Temporal resolution = 1ms per micro-step)")
    axes[-1].set_xlabel("Time (Micro-steps / ms)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Improved raster plot saved to {output_path}")

if __name__ == "__main__":
    generate_improved_raster()
