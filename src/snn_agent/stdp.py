"""
Spike-Timing Dependent Plasticity (STDP)
-----------------------------------------
Pair-based STDP with reward-modulated gating (R-STDP).

  ΔW_ij  = η × reward × (A_+ × pre_trace_i × post_spike_j
                        - A_- × post_trace_j × pre_spike_i)

Traces decay exponentially:
  x_pre  → x_pre  × exp(-dt/tau_+) + pre_spike
  x_post → x_post × exp(-dt/tau_-) + post_spike

Weight bounds are soft-clipped to [w_min, w_max].
"""

import numpy as np


class STDPSynapse:
    """
    Full weight matrix between two LIF layers.

    Parameters
    ----------
    n_pre      : int    number of pre-synaptic neurons
    n_post     : int    number of post-synaptic neurons
    lr         : float  base learning rate η
    A_plus     : float  LTP amplitude
    A_minus    : float  LTD amplitude
    tau_plus   : float  pre-trace decay constant (ms)
    tau_minus  : float  post-trace decay constant (ms)
    dt         : float  timestep (ms)
    w_min      : float  minimum weight
    w_max      : float  maximum weight
    w_init_std : float  std-dev for initial weight Gaussian
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        lr: float = 0.01,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        dt: float = 1.0,
        w_min: float = -1.0,
        w_max: float = 1.0,
        w_init_std: float = 0.1,
        rng: np.random.Generator = None,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.lr = lr
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.dt = dt
        self.w_min = w_min
        self.w_max = w_max

        self.decay_plus = np.exp(-dt / tau_plus)
        self.decay_minus = np.exp(-dt / tau_minus)

        rng = rng or np.random.default_rng()
        self.W = rng.normal(0.0, w_init_std, size=(n_pre, n_post))
        self.W = np.clip(self.W, w_min, w_max)

        self.x_pre = np.zeros(n_pre)
        self.x_post = np.zeros(n_post)

    # ------------------------------------------------------------------
    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        Compute post-synaptic current from pre-synaptic spikes.

        Parameters
        ----------
        pre_spikes : (n_pre,) bool

        Returns
        -------
        current : (n_post,) float
        """
        return self.W[pre_spikes].sum(axis=0)

    # ------------------------------------------------------------------
    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        reward: float = 1.0,
    ) -> None:
        """
        Apply one timestep of R-STDP.

        Parameters
        ----------
        pre_spikes  : (n_pre,)  bool
        post_spikes : (n_post,) bool
        reward      : float  modulation signal (positive=LTP favoured,
                              negative=LTD favoured, 0=no update)
        """
        # Update traces
        self.x_pre *= self.decay_plus
        self.x_pre[pre_spikes] += 1.0

        self.x_post *= self.decay_minus
        self.x_post[post_spikes] += 1.0

        # Outer products for dW
        # LTP: pre-trace × post-spike
        dW_ltp = np.outer(self.x_pre, post_spikes.astype(np.float64))
        # LTD: pre-spike × post-trace
        dW_ltd = np.outer(pre_spikes.astype(np.float64), self.x_post)

        dW = self.lr * reward * (self.A_plus * dW_ltp - self.A_minus * dW_ltd)
        self.W += dW
        np.clip(self.W, self.w_min, self.w_max, out=self.W)


    # ------------------------------------------------------------------
    def reset_traces(self) -> None:
        """Reset eligibility traces at episode boundary."""
        self.x_pre[:] = 0.0
        self.x_post[:] = 0.0
