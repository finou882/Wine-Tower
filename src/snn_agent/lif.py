"""
Leaky Integrate-and-Fire (LIF) Neuron Layer
--------------------------------------------
Vectorised numpy implementation.

Key design choices
~~~~~~~~~~~~~~~~~~
* Membrane potential leaks with time constant tau_m.
* Spike when V >= V_thresh; hard reset to V_rest.
* Refractory period enforced via a countdown counter.
* Dead-neuron detection: a neuron whose spike-rate over a rolling window
  stays below `dead_threshold` is flagged as "dead" and receives a
  homeostatic depolarisation kick so it can re-engage.
  (This is the desired WTA-induced dead-neuron behaviour – once a WTA
  winner suppresses neighbours repeatedly, those neighbours may go
  permanently silent and become specialised for other stimuli.)
"""

import numpy as np
from typing import Optional


class LIFLayer:
    """
    A single layer of LIF neurons.

    Parameters
    ----------
    n_neurons : int
    tau_m     : float   membrane time constant (ms)
    V_rest    : float   resting / reset potential
    V_thresh  : float   spike threshold
    t_refrac  : int     refractory period (timesteps)
    dt        : float   simulation timestep (ms)
    dead_window : int   rolling window length for dead-neuron detection
    dead_threshold : float  spike-rate below which a neuron is "dead"
    """

    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 20.0,
        V_rest: float = -65.0,
        V_thresh: float = -50.0,
        t_refrac: int = 5,
        dt: float = 1.0,
        dead_window: int = 200,
        dead_threshold: float = 0.005,
    ):
        self.n = n_neurons
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_thresh = V_thresh
        self.t_refrac = t_refrac
        self.dt = dt
        self.dead_window = dead_window
        self.dead_threshold = dead_threshold

        self.decay = np.exp(-dt / tau_m)

        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.V = np.full(self.n, self.V_rest, dtype=np.float64)
        self.refrac_count = np.zeros(self.n, dtype=np.int32)
        # Circular buffer for spike history (used for dead-neuron detection)
        self._spike_buf = np.zeros((self.dead_window, self.n), dtype=np.uint8)
        self._buf_idx = 0
        self._buf_filled = 0  # how many steps have been written (caps at dead_window)
        self.dead_mask = np.zeros(self.n, dtype=bool)   # True = dead

    def reset(self) -> None:
        """
        Reset per-episode neuron state (voltage, refractory counters).
        The spike-history buffer and dead_mask are intentionally PRESERVED
        across episodes so that long-term dead-neuron detection works correctly.
        """
        self.V = np.full(self.n, self.V_rest, dtype=np.float64)
        self.refrac_count = np.zeros(self.n, dtype=np.int32)

    # ------------------------------------------------------------------
    def step(self, I: np.ndarray,
             assist_I: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance one timestep.

        Parameters
        ----------
        I        : (n_neurons,) float  – total input current
        assist_I : (n_neurons,) float, optional
                   Extra current injected into dead neurons only (Wine-Tower).

        Returns
        -------
        spikes : (n_neurons,) bool
        """
        # Refractory: clamp V for neurons still refractory
        in_refrac = self.refrac_count > 0

        # Scale I so that typical synaptic input drives neurons toward threshold.
        # factor = tau_m/dt so that I=1 corresponds to 1mV/step contribution.
        I_eff = I * (self.tau_m / self.dt)

        # Wine-Tower: inject extra current only into dead neurons
        if assist_I is not None and self.dead_mask.any():
            I_eff = I_eff.copy()
            I_eff[self.dead_mask] += assist_I[self.dead_mask]

        # Membrane update (only for non-refractory neurons)
        dV = (-(self.V - self.V_rest) + I_eff) * (self.dt / self.tau_m)
        self.V = np.where(in_refrac, self.V_rest, self.V + dV)

        # Spike detection
        spikes = self.V >= self.V_thresh

        # Reset spiking neurons
        self.V[spikes] = self.V_rest
        self.refrac_count[spikes] = self.t_refrac
        self.refrac_count[~spikes] = np.maximum(0, self.refrac_count[~spikes] - 1)

        # Record in circular buffer
        self._spike_buf[self._buf_idx] = spikes.astype(np.uint8)
        self._buf_idx = (self._buf_idx + 1) % self.dead_window
        self._buf_filled = min(self._buf_filled + 1, self.dead_window)

        # Update dead-neuron mask – only after buffer is at least half full
        if self._buf_filled >= self.dead_window // 2:
            mean_rate = self._spike_buf[:self._buf_filled].mean(axis=0)
            self.dead_mask = mean_rate < self.dead_threshold

        return spikes

    # ------------------------------------------------------------------
    def homeostatic_kick(self, strength: float = 5.0) -> None:
        """
        Depolarise dead neurons slightly so they can recover.
        Called externally (e.g., each episode reset).
        """
        self.V[self.dead_mask] += strength

    # ------------------------------------------------------------------
    @property
    def n_dead(self) -> int:
        return int(self.dead_mask.sum())
