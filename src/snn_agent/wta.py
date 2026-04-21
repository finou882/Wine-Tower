"""
Winner-Takes-All (WTA) competition layer
-----------------------------------------
Lateral inhibition implemented as k-WTA:
  - After each timestep, only the top-k most-depolarised neurons
    are allowed to spike; all others are clamped.
  - Neurons that never win accumulate suppression and eventually
    become "dead" (as tracked by LIFLayer.dead_mask).

This intentional dead-neuron phenomenon is desirable:
  * WTA forces specialisation → most neurons go silent on most inputs.
  * Dead neurons are *unable* to respond to unexpected stimuli,
    making the agent's behaviour brittle for out-of-distribution events.
  * Homeostatic kicks provide limited recovery, keeping the dynamics
    interesting without fully rescuing all dead neurons.
"""

import numpy as np
from .lif import LIFLayer


class WTALayer(LIFLayer):
    """
    LIF layer with k-WTA lateral inhibition.

    Parameters
    ----------
    k : int   number of winners allowed to spike per timestep (default 3)
    All other parameters forwarded to LIFLayer.
    """

    def __init__(self, n_neurons: int, k: int = 3, **lif_kwargs):
        super().__init__(n_neurons, **lif_kwargs)
        self.k = min(k, n_neurons)
        self._wta_rng = np.random.default_rng()
        # Grace period: revived neurons are exempt from WTA suppression
        # for this many timesteps after revival (adult neurogenesis grace)
        self.grace_counts = np.zeros(n_neurons, dtype=np.int32)

    # ------------------------------------------------------------------
    def step(self, I: np.ndarray, assist_I=None) -> np.ndarray:
        """
        Advance one timestep with WTA masking applied after spike detection.

        Only the top-k neurons by membrane potential are allowed to produce
        spikes. Lateral inhibition is applied as a current penalty to
        non-winner neurons to drive specialisation and dead-neuron formation.
        Grace-period neurons are always allowed to spike regardless of WTA.
        """
        spikes = super().step(I, assist_I=assist_I)

        if spikes.any():
            n_spikes = spikes.sum()
            if n_spikes > self.k:
                fired_idx = np.where(spikes)[0]
                if len(fired_idx) > self.k:
                    # Grace-period neurons fire as BONUS (outside k quota)
                    grace_mask = self.grace_counts[fired_idx] > 0
                    grace_fired = fired_idx[grace_mask]
                    normal_fired = fired_idx[~grace_mask]

                    # Normal WTA among non-grace neurons (full k slots)
                    if len(normal_fired) > self.k:
                        current_at_normal = I[normal_fired]
                        sorted_normal = normal_fired[np.argsort(-current_at_normal)]
                        normal_winners = sorted_normal[:self.k]
                        normal_losers  = sorted_normal[self.k:]
                        self.V[normal_losers] = self.V_rest
                        spikes[normal_losers] = False
                    # Grace neurons always keep their spikes (bonus winners)

        # Decrement grace counts
        self.grace_counts = np.maximum(0, self.grace_counts - 1)

        return spikes
