"""
SNN Agent (Deep Architecture)
-----------------------------
Architecture:
  Input layer (encode observations as Poisson spike trains)
      ↓  STDPSynapse W_in
  Hidden layer 1 (WTA-LIF)
      ↓  STDPSynapse W_12
  Hidden layer 2 (WTA-LIF)
      ↓  STDPSynapse W_23
  Hidden layer 3 (WTA-LIF, memory/representation)
      ↓  STDPSynapse W_out
  Output layer (LIF, one neuron per action)

Action selection: argmax of output firing rates over the episode step.
Recurrent connections (W_rec1, W_rec2, W_rec3) allow short-term memory.

Reward-modulated STDP gates all weight updates.
"""

import numpy as np
from typing import List, Optional
import pathlib

from .lif import LIFLayer
from .wta import WTALayer
from .stdp import STDPSynapse


class SNNAgent:
    """
    Deep SNN agent with 3 hidden layers (LIF + STDP + WTA).

    Parameters
    ----------
    obs_dim     : int   observation dimensionality
    act_dim     : int   number of actions
    n_hidden    : int   hidden layer size (for all 3 layers)
    wta_k       : int   WTA k value (winners per step)
    encode_steps: int   number of spike-train encoding steps per observation
    recurrent   : bool  add recurrent synapses in hidden layers
    seed        : int   random seed
    """
    #easteregg: the "wine tower" replay method is inspired by the idea of using a "tower of wine" to revive dead neurons by providing them with strong, reward-modulated input during replay. The name is a playful nod to the concept of a wine tower, where each layer represents a different level of revival for the neurons.

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_hidden: int = 64,
        wta_k: int = 4,
        encode_steps: int = 10,
        recurrent: bool = True,
        lr: float = 0.005,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_hidden = n_hidden
        self.encode_steps = encode_steps
        self.rng = np.random.default_rng(seed)

        # Layers
        # dead_window: ~2000 env-steps × encode_steps
        dead_win = encode_steps * 2000
        lif_kwargs = dict(tau_m=20.0, V_rest=0.0, V_thresh=1.0,
                          t_refrac=3, dt=1.0,
                          dead_window=dead_win, dead_threshold=0.003)

        self.hidden1 = WTALayer(n_hidden, k=wta_k, **lif_kwargs)
        self.hidden2 = WTALayer(n_hidden, k=wta_k, **lif_kwargs)
        self.hidden3 = WTALayer(n_hidden, k=wta_k, **lif_kwargs)
        self.output = LIFLayer(act_dim,
                               tau_m=10.0, V_rest=0.0, V_thresh=1.0,
                               t_refrac=2, dt=1.0,
                               dead_window=dead_win, dead_threshold=0.003)

        self.hiddens = [self.hidden1, self.hidden2, self.hidden3]

        # Synapses
        stdp_common = dict(lr=lr, A_plus=0.02, A_minus=0.01,
                           tau_plus=20.0, tau_minus=20.0, dt=1.0,
                           w_min=-2.0, w_max=2.0, rng=self.rng)

        self.W_in  = STDPSynapse(obs_dim, n_hidden, w_init_std=0.3, **stdp_common)
        self.W_12  = STDPSynapse(n_hidden, n_hidden, w_init_std=0.3, **stdp_common)
        self.W_23  = STDPSynapse(n_hidden, n_hidden, w_init_std=0.3, **stdp_common)
        self.W_out = STDPSynapse(n_hidden, act_dim, w_init_std=0.5, **stdp_common)

        self.feedforward_synapses = [self.W_in, self.W_12, self.W_23, self.W_out]

        self.recurrent = recurrent
        if recurrent:
            self.W_rec1 = STDPSynapse(n_hidden, n_hidden, w_init_std=0.1, **stdp_common)
            self.W_rec2 = STDPSynapse(n_hidden, n_hidden, w_init_std=0.1, **stdp_common)
            self.W_rec3 = STDPSynapse(n_hidden, n_hidden, w_init_std=0.1, **stdp_common)
            self.recurrent_synapses = [self.W_rec1, self.W_rec2, self.W_rec3]
        else:
            self.recurrent_synapses = []

        # Rate accumulators for action selection
        self._out_rates = np.zeros(act_dim)
        self._step_count = 0

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset per-episode state (neuron voltages, traces, rate accum)."""
        for h in self.hiddens:
            h.reset()
        self.output.reset()
        
        for syn in self.feedforward_synapses:
            syn.reset_traces()
        for syn in self.recurrent_synapses:
            syn.reset_traces()
            
        self._out_rates[:] = 0.0
        self._step_count = 0

        # Store last hidden spikes for recurrent input
        self._last_h_spikes = [
            np.zeros(self.n_hidden, dtype=bool) for _ in range(3)
        ]

    # ------------------------------------------------------------------
    def _encode(self, obs: np.ndarray) -> np.ndarray:
        prob = np.clip(obs, 0.0, 1.0)
        return self.rng.random(self.obs_dim) < prob

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, reward: float = 0.0,
            learn: bool = True, epsilon: float = 0.0) -> int:
        self._out_rates[:] = 0.0
        tonic_bias_out = np.ones(self.act_dim) * 0.02

        for _ in range(self.encode_steps):
            in_spikes = self._encode(obs)

            # H1
            I_h1 = self.W_in.forward(in_spikes)
            if self.recurrent: I_h1 += self.W_rec1.forward(self._last_h_spikes[0])
            h1_spikes = self.hidden1.step(I_h1)

            # H2
            I_h2 = self.W_12.forward(h1_spikes)
            if self.recurrent: I_h2 += self.W_rec2.forward(self._last_h_spikes[1])
            h2_spikes = self.hidden2.step(I_h2)

            # H3
            I_h3 = self.W_23.forward(h2_spikes)
            if self.recurrent: I_h3 += self.W_rec3.forward(self._last_h_spikes[2])
            h3_spikes = self.hidden3.step(I_h3)

            # Output
            I_out = self.W_out.forward(h3_spikes) + tonic_bias_out
            out_spikes = self.output.step(I_out)
            self._out_rates += out_spikes.astype(float)

            # STDP updates gated by reward modulation
            if learn:
                mod = reward  # positive → LTP, negative → LTD
                self.W_in.update(in_spikes, h1_spikes, reward=mod)
                self.W_12.update(h1_spikes, h2_spikes, reward=mod)
                self.W_23.update(h2_spikes, h3_spikes, reward=mod)
                self.W_out.update(h3_spikes, out_spikes, reward=mod)
                if self.recurrent:
                    self.W_rec1.update(self._last_h_spikes[0], h1_spikes, reward=mod)
                    self.W_rec2.update(self._last_h_spikes[1], h2_spikes, reward=mod)
                    self.W_rec3.update(self._last_h_spikes[2], h3_spikes, reward=mod)

            # Update history
            self._last_h_spikes[0] = h1_spikes
            self._last_h_spikes[1] = h2_spikes
            self._last_h_spikes[2] = h3_spikes

        self._step_count += 1
        if self._out_rates.sum() == 0 or self.rng.random() < epsilon:
            action = int(self.rng.integers(self.act_dim))
        else:
            action = int(np.argmax(self._out_rates))
        return action

    # ------------------------------------------------------------------
    def wine_tower_replay(
        self,
        trajectories: list,
        wine_strength: float = 50.0,
        n_passes: int = 2,
    ) -> None:
        has_dead = any(h.dead_mask.any() for h in self.hiddens)
        if not has_dead:
            return

        tonic_out = np.ones(self.act_dim) * 0.02
        revived = [np.zeros(self.n_hidden, dtype=bool) for _ in range(3)]

        # Temporarily boost lr for revived-neuron rapid re-specialisation
        orig_lrs_ff = [syn.lr for syn in self.feedforward_synapses]
        orig_lrs_rec = [syn.lr for syn in self.recurrent_synapses]

        for syn in self.feedforward_synapses:
            syn.lr *= 10.0
        for syn in self.recurrent_synapses:
            syn.lr *= 10.0

        for _ in range(n_passes):
            W_pos_list = [np.maximum(0.0, syn.W) for syn in self.recurrent_synapses] if self.recurrent else [None, None, None]
            
            for traj in trajectories:
                self.reset_episode()
                for obs, step_reward in traj:
                    effective = max(0.0, step_reward) * 20.0
                    for _ in range(self.encode_steps):
                        in_spikes = self._encode(obs)

                        # H1
                        I_h1 = self.W_in.forward(in_spikes)
                        if self.recurrent: I_h1 += self.W_rec1.forward(self._last_h_spikes[0])
                        assist_I_h1 = None
                        if self.recurrent and self.hidden1.dead_mask.any():
                            live = self._last_h_spikes[0] & ~self.hidden1.dead_mask
                            assist_I_h1 = wine_strength * (W_pos_list[0] @ live.astype(np.float64))
                        h1_spikes = self.hidden1.step(I_h1, assist_I=assist_I_h1)
                        revived[0] |= (h1_spikes & self.hidden1.dead_mask)

                        # H2
                        I_h2 = self.W_12.forward(h1_spikes)
                        if self.recurrent: I_h2 += self.W_rec2.forward(self._last_h_spikes[1])
                        assist_I_h2 = None
                        if self.recurrent and self.hidden2.dead_mask.any():
                            live = self._last_h_spikes[1] & ~self.hidden2.dead_mask
                            assist_I_h2 = wine_strength * (W_pos_list[1] @ live.astype(np.float64))
                        h2_spikes = self.hidden2.step(I_h2, assist_I=assist_I_h2)
                        revived[1] |= (h2_spikes & self.hidden2.dead_mask)

                        # H3
                        I_h3 = self.W_23.forward(h2_spikes)
                        if self.recurrent: I_h3 += self.W_rec3.forward(self._last_h_spikes[2])
                        assist_I_h3 = None
                        if self.recurrent and self.hidden3.dead_mask.any():
                            live = self._last_h_spikes[2] & ~self.hidden3.dead_mask
                            assist_I_h3 = wine_strength * (W_pos_list[2] @ live.astype(np.float64))
                        h3_spikes = self.hidden3.step(I_h3, assist_I=assist_I_h3)
                        revived[2] |= (h3_spikes & self.hidden3.dead_mask)

                        # Output
                        I_out = self.W_out.forward(h3_spikes) + tonic_out
                        out_spikes = self.output.step(I_out)

                        # STDP
                        self.W_in.update(in_spikes, h1_spikes, reward=effective)
                        self.W_12.update(h1_spikes, h2_spikes, reward=effective)
                        self.W_23.update(h2_spikes, h3_spikes, reward=effective)
                        self.W_out.update(h3_spikes, out_spikes, reward=effective)
                        if self.recurrent:
                            self.W_rec1.update(self._last_h_spikes[0], h1_spikes, reward=effective)
                            self.W_rec2.update(self._last_h_spikes[1], h2_spikes, reward=effective)
                            self.W_rec3.update(self._last_h_spikes[2], h3_spikes, reward=effective)

                        # Update History
                        self._last_h_spikes[0] = h1_spikes
                        self._last_h_spikes[1] = h2_spikes
                        self._last_h_spikes[2] = h3_spikes

        # Restore original lr
        for syn, lr in zip(self.feedforward_synapses, orig_lrs_ff):
            syn.lr = lr
        for syn, lr in zip(self.recurrent_synapses, orig_lrs_rec):
            syn.lr = lr

        for i, h in enumerate(self.hiddens):
            if revived[i].any():
                h._spike_buf[:, revived[i]] = 0
                h._buf_filled = min(h._buf_filled, h.dead_window // 2 - 1)
                h.dead_mask[revived[i]] = False
                grace_steps = self.encode_steps * 500
                h.grace_counts[revived[i]] = grace_steps

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "W_in": self.W_in.W,
            "W_12": self.W_12.W,
            "W_23": self.W_23.W,
            "W_out": self.W_out.W,
        }
        for i, h in enumerate(self.hiddens):
            data[f"h{i+1}_spike_buf"] = h._spike_buf
            data[f"h{i+1}_buf_idx"] = np.array(h._buf_idx)
            data[f"h{i+1}_buf_filled"] = np.array(h._buf_filled)
            data[f"h{i+1}_dead_mask"] = h.dead_mask

        data["output_spike_buf"] = self.output._spike_buf
        data["output_buf_idx"] = np.array(self.output._buf_idx)
        data["output_buf_filled"] = np.array(self.output._buf_filled)
        data["output_dead_mask"] = self.output.dead_mask

        if self.recurrent:
            data["W_rec1"] = self.W_rec1.W
            data["W_rec2"] = self.W_rec2.W
            data["W_rec3"] = self.W_rec3.W

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        print(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        d = np.load(path)
        self.W_in.W = d["W_in"]
        self.W_12.W = d["W_12"]
        self.W_23.W = d["W_23"]
        self.W_out.W = d["W_out"]
        
        for i, h in enumerate(self.hiddens):
            h._spike_buf = d[f"h{i+1}_spike_buf"]
            h._buf_idx = int(d[f"h{i+1}_buf_idx"])
            h._buf_filled = int(d[f"h{i+1}_buf_filled"])
            h.dead_mask = d[f"h{i+1}_dead_mask"]
            
        self.output._spike_buf = d["output_spike_buf"]
        self.output._buf_idx = int(d["output_buf_idx"])
        self.output._buf_filled = int(d["output_buf_filled"])
        self.output.dead_mask = d["output_dead_mask"]
        
        if self.recurrent and "W_rec1" in d:
            self.W_rec1.W = d["W_rec1"]
            self.W_rec2.W = d["W_rec2"]
            self.W_rec3.W = d["W_rec3"]
        print(f"Model loaded from: {path}")

    # ------------------------------------------------------------------
    @property
    def n_dead_hidden1(self) -> int:
        return self.hidden1.n_dead

    @property
    def n_dead_hidden2(self) -> int:
        return self.hidden2.n_dead

    @property
    def n_dead_hidden3(self) -> int:
        return self.hidden3.n_dead

    @property
    def n_dead_output(self) -> int:
        return self.output.n_dead
