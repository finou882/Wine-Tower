"""
SNN Agent
----------
Architecture:
  Input layer  (encode observations as Poisson spike trains)
      ↓  STDPSynapse  W1
  Hidden layer (WTA-LIF, memory/representation)
      ↓  STDPSynapse  W2
  Output layer (LIF, one neuron per action)

Action selection: argmax of output firing rates over the episode step.
Recurrent connections in the hidden layer (optional) allow short-term memory.

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
    Small SNN agent with LIF + STDP + WTA.

    Parameters
    ----------
    obs_dim     : int   observation dimensionality
    act_dim     : int   number of actions
    n_hidden    : int   hidden layer size
    wta_k       : int   WTA k value (winners per step)
    encode_steps: int   number of spike-train encoding steps per observation
    recurrent   : bool  add recurrent synapses in hidden layer
    seed        : int   random seed
    """

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
        # dead_window: ~2000 env-steps × encode_steps so dead detection
        # only triggers after prolonged multi-episode silence.
        # dead_threshold: 0.3% — only truly never-firing neurons flagged.
        dead_win = encode_steps * 2000
        lif_kwargs = dict(tau_m=20.0, V_rest=0.0, V_thresh=1.0,
                          t_refrac=3, dt=1.0,
                          dead_window=dead_win, dead_threshold=0.003)

        self.hidden = WTALayer(n_hidden, k=wta_k, **lif_kwargs)
        self.output = LIFLayer(act_dim,
                               tau_m=10.0, V_rest=0.0, V_thresh=1.0,
                               t_refrac=2, dt=1.0,
                               dead_window=dead_win, dead_threshold=0.003)

        # Synapses
        # A_plus > A_minus: LTP-biased so reward signals can consolidate.
        # Previous 0.01/0.012 ratio caused LTD to dominate (LTD total ≈ 2× LTP),
        # preventing any learning in Phase 1/2.
        stdp_common = dict(lr=lr, A_plus=0.02, A_minus=0.01,
                           tau_plus=20.0, tau_minus=20.0, dt=1.0,
                           w_min=-2.0, w_max=2.0, rng=self.rng)

        self.W_in  = STDPSynapse(obs_dim, n_hidden,
                                 w_init_std=0.3, **stdp_common)
        self.W_out = STDPSynapse(n_hidden, act_dim,
                                 w_init_std=0.5, **stdp_common)

        self.recurrent = recurrent
        if recurrent:
            self.W_rec = STDPSynapse(n_hidden, n_hidden,
                                     w_init_std=0.1, **stdp_common)

        # Rate accumulators for action selection
        self._out_rates = np.zeros(act_dim)
        self._step_count = 0

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset per-episode state (neuron voltages, traces, rate accum)."""
        self.hidden.reset()
        self.output.reset()
        self.W_in.reset_traces()
        self.W_out.reset_traces()
        if self.recurrent:
            self.W_rec.reset_traces()
        self._out_rates[:] = 0.0
        self._step_count = 0
        # No homeostatic kick: dead neurons stay dead (intentional WTA behaviour).

        # Store last hidden spikes for recurrent input
        self._last_hidden_spikes = np.zeros(self.n_hidden, dtype=bool)

    # ------------------------------------------------------------------
    def _encode(self, obs: np.ndarray) -> np.ndarray:
        """
        Rate-code observation as Poisson spike train.
        obs values in [0,1] → Bernoulli probability per timestep.
        Returns (obs_dim,) bool spikes.
        """
        prob = np.clip(obs, 0.0, 1.0)
        return self.rng.random(self.obs_dim) < prob

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, reward: float = 0.0,
            learn: bool = True, epsilon: float = 0.0) -> int:
        """
        Encode observation, run SNN for `encode_steps` timesteps,
        accumulate eligibility traces, apply STDP gated by reward.

        Parameters
        ----------
        obs    : (obs_dim,) float  current observation
        reward : float             modulation signal for STDP
        learn  : bool              whether to apply weight updates

        Returns
        -------
        action : int
        """
        self._out_rates[:] = 0.0
        tonic_bias_out = np.ones(self.act_dim) * 0.02

        for _ in range(self.encode_steps):
            in_spikes = self._encode(obs)

            # Input → hidden current
            I_hidden = self.W_in.forward(in_spikes)
            if self.recurrent:
                I_hidden += self.W_rec.forward(self._last_hidden_spikes)

            hidden_spikes = self.hidden.step(I_hidden)
            self._last_hidden_spikes = hidden_spikes

            # Hidden → output current + tonic bias
            I_out = self.W_out.forward(hidden_spikes) + tonic_bias_out
            out_spikes = self.output.step(I_out)

            self._out_rates += out_spikes.astype(float)

            # STDP updates gated by reward modulation
            if learn:
                # Use absolute reward |r| for LTP/LTD direction from sign
                mod = reward  # positive → LTP, negative → LTD
                self.W_in.update(in_spikes, hidden_spikes, reward=mod)
                self.W_out.update(hidden_spikes, out_spikes, reward=mod)
                if self.recurrent:
                    self.W_rec.update(
                        self._last_hidden_spikes, hidden_spikes, reward=mod
                    )

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
        """
        Wine-Tower offline replay: when a live W_rec neighbour fires, inject
        a spike-triggered current (wine_strength × W_pos[d,j]) into dead
        neuron d so it crosses threshold and recovers its dead_mask.

        Uses spike events rather than continuous V_excess so current magnitude
        is sufficient:
            wine_strength(50) × W_pos(~0.07) × k_winners(16) ≈ 56 >> 20(thresh)

        STDP is gated by the original replay reward (meaningful context only).

        Parameters
        ----------
        trajectories  : list of list[(obs, reward)]
        wine_strength : spike-triggered current multiplier for dead neurons
        n_passes      : number of replay passes over each trajectory
        """
        if not self.hidden.dead_mask.any():
            return

        tonic_out = np.ones(self.act_dim) * 0.02
        revived = np.zeros(self.n_hidden, dtype=bool)

        # Temporarily boost lr for revived-neuron rapid re-specialisation
        orig_lr_in  = self.W_in.lr
        orig_lr_rec = self.W_rec.lr if self.recurrent else None
        orig_lr_out = self.W_out.lr
        self.W_in.lr  = orig_lr_in  * 10.0
        self.W_out.lr = orig_lr_out * 10.0
        if self.recurrent:
            self.W_rec.lr = orig_lr_rec * 10.0

        for _ in range(n_passes):
            W_pos = np.maximum(0.0, self.W_rec.W) if self.recurrent else None
            for traj in trajectories:
                self.reset_episode()
                for obs, step_reward in traj:
                    effective = max(0.0, step_reward) * 20.0
                    for _ in range(self.encode_steps):
                        in_spikes = self._encode(obs)
                        I_hidden = self.W_in.forward(in_spikes)
                        if self.recurrent:
                            I_hidden += self.W_rec.forward(
                                self._last_hidden_spikes
                            )

                        # --- Wine-Tower: spike-triggered injection ---
                        assist_I = None
                        if self.recurrent and self.hidden.dead_mask.any():
                            live_spikes = (
                                self._last_hidden_spikes
                                & ~self.hidden.dead_mask
                            )
                            assist_I = wine_strength * (
                                W_pos @ live_spikes.astype(np.float64)
                            )
                        # --- end Wine-Tower ---

                        hidden_spikes = self.hidden.step(
                            I_hidden, assist_I=assist_I
                        )
                        revived |= (hidden_spikes & self.hidden.dead_mask)
                        self._last_hidden_spikes = hidden_spikes

                        I_out = self.W_out.forward(hidden_spikes) + tonic_out
                        out_spikes = self.output.step(I_out)

                        self.W_in.update(in_spikes, hidden_spikes, reward=effective)
                        self.W_out.update(hidden_spikes, out_spikes, reward=effective)
                        if self.recurrent:
                            self.W_rec.update(
                                self._last_hidden_spikes,
                                hidden_spikes,
                                reward=effective,
                            )

        # Restore original lr
        self.W_in.lr  = orig_lr_in
        self.W_out.lr = orig_lr_out
        if self.recurrent:
            self.W_rec.lr = orig_lr_rec

        if revived.any():
            self.hidden._spike_buf[:, revived] = 0
            self.hidden._buf_filled = min(
                self.hidden._buf_filled,
                self.hidden.dead_window // 2 - 1,
            )
            self.hidden.dead_mask[revived] = False
            # Grant grace period: exempt revived neurons from WTA for N steps
            # encode_steps(10) × 500steps ≈ 50 episodes of protection
            grace_steps = self.encode_steps * 500
            self.hidden.grace_counts[revived] = grace_steps

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save all weight matrices and spike-history buffers to a .npz file.
        """
        data = {
            "W_in": self.W_in.W,
            "W_out": self.W_out.W,
            "hidden_spike_buf": self.hidden._spike_buf,
            "hidden_buf_idx": np.array(self.hidden._buf_idx),
            "hidden_buf_filled": np.array(self.hidden._buf_filled),
            "hidden_dead_mask": self.hidden.dead_mask,
            "output_spike_buf": self.output._spike_buf,
            "output_buf_idx": np.array(self.output._buf_idx),
            "output_buf_filled": np.array(self.output._buf_filled),
            "output_dead_mask": self.output.dead_mask,
        }
        if self.recurrent:
            data["W_rec"] = self.W_rec.W
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        print(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        """
        Restore weights and spike-history from a .npz file saved by save().
        """
        d = np.load(path)
        self.W_in.W = d["W_in"]
        self.W_out.W = d["W_out"]
        self.hidden._spike_buf = d["hidden_spike_buf"]
        self.hidden._buf_idx = int(d["hidden_buf_idx"])
        self.hidden._buf_filled = int(d["hidden_buf_filled"])
        self.hidden.dead_mask = d["hidden_dead_mask"]
        self.output._spike_buf = d["output_spike_buf"]
        self.output._buf_idx = int(d["output_buf_idx"])
        self.output._buf_filled = int(d["output_buf_filled"])
        self.output.dead_mask = d["output_dead_mask"]
        if self.recurrent and "W_rec" in d:
            self.W_rec.W = d["W_rec"]
        print(f"Model loaded from: {path}")

    # ------------------------------------------------------------------
    @property
    def n_dead_hidden(self) -> int:
        return self.hidden.n_dead

    @property
    def n_dead_output(self) -> int:
        return self.output.n_dead
