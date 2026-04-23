"""
Multiple T-Maze Environment
---------------------------
10 goals arranged as a series of T-junctions.
Each episode the agent is told 3 candidate goal IDs upfront;
hundreds of episodes later it must recall all 10 simultaneously.

Observation:
  - One-hot junction position (T-junction index 0..N_JUNCTIONS-1)
  - One-hot cue for each candidate goal (N_GOALS bits)
  - A "hint active" flag (1 bit): 1 during cue phase, 0 during navigation

Action:  0=left, 1=right, 2=forward (straight)
"""

import numpy as np
from typing import List, Optional, Tuple

N_GOALS = 5
N_JUNCTIONS = 5          # depth of T-maze tree (2^5 = 32 leaves)
CUE_STEPS = 5            # timesteps the cue is shown before navigation starts

# Pre-compute goal→path mappings (binary tree navigation)
# Goal i -> sequence of L/R turns at each junction (0=left,1=right)
GOAL_PATHS = {}
for g in range(N_GOALS):
    # each goal maps to a unique path (use modular arithmetic over 5-bit patterns)
    bits = [(g >> j) & 1 for j in range(N_JUNCTIONS)]
    GOAL_PATHS[g] = bits   # bits[0] = turn at junction 0, etc.


class MultipleTMaze:
    """
    Multiple T-Maze with curriculum-controlled goal visibility.

    Parameters
    ----------
    n_goals : int
        Total number of goal locations (default 10).
    fixed_goals : Optional[List[int]]
        If provided, always use these goals every episode (no curriculum).
        Length must match ``n_active_goals``.
    """

    OBS_DIM = N_JUNCTIONS + N_GOALS + 1   # position one-hot + goal cues + hint flag
    ACT_DIM = 3                            # left / right / forward

    def __init__(
        self,
        n_goals: int = N_GOALS,
        fixed_goals: Optional[List[int]] = None,
    ):
        self.n_goals = n_goals
        self.fixed_goals = fixed_goals

        self.junction = 0
        self.cue_step = 0
        self.in_cue_phase = True
        self.active_goals: List[int] = []
        self.target_goal: int = 0
        self.done = False
        self.n_active_junctions = N_JUNCTIONS  # controlled externally per episode
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def set_active_goals(self, goals: List[int]) -> None:
        """Override the active goal list (used by the curriculum)."""
        self.active_goals = list(goals)

    # ------------------------------------------------------------------
    def reset(self, active_goals: Optional[List[int]] = None) -> np.ndarray:
        """
        Start a new episode.

        Parameters
        ----------
        active_goals : list[int] or None
            Goal IDs to present as cues this episode.
            Ignored if ``fixed_goals`` was set at construction.
        """
        self.junction = 0
        self.cue_step = 0
        self.in_cue_phase = True
        self.done = False
        # NOTE: n_active_junctions is NOT reset here so the trainer
        # can control it per-episode via self.env.n_active_junctions.

        if self.fixed_goals is not None:
            self.active_goals = list(self.fixed_goals)
        elif active_goals is not None:
            self.active_goals = list(active_goals)
        # else keep whatever was set via set_active_goals

        # Pick which goal the agent must reach this episode
        self.target_goal = int(self.rng.choice(self.active_goals))
        return self._observe()

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action and advance one timestep.

        Returns obs, reward, done, info.
        """
        if self.done:
            raise RuntimeError("Call reset() before step() after episode end.")

        reward = 0.0
        info: dict = {}

        if self.in_cue_phase:
            self.cue_step += 1
            if self.cue_step >= CUE_STEPS:
                self.in_cue_phase = False
            obs = self._observe()
            return obs, reward, False, info

        # Navigation phase
        correct_turn = GOAL_PATHS[self.target_goal][self.junction]
        if action == correct_turn:
            self.junction += 1
            reward = 0.1   # small shaped reward for correct step
        else:
            # Wrong turn → penalty and end
            reward = -1.0
            self.done = True
            info["outcome"] = "wrong_turn"
            return self._observe(), reward, True, info

        if self.junction >= self.n_active_junctions:
            # Reached the active depth for the target goal
            reward = 1.0
            self.done = True
            info["outcome"] = "success"
        else:
            obs = self._observe()

        if not self.done:
            return self._observe(), 0.1, False, info

        return self._observe(), reward, True, info

    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        # Junction position (one-hot; clamp to last junction during cue phase)
        pos = min(self.junction, N_JUNCTIONS - 1)
        obs[pos] = 1.0
        # Goal cues (one-hot over ALL N_GOALS positions; only active goals lit)
        for g in self.active_goals:
            obs[N_JUNCTIONS + g] = 1.0
        # Hint flag
        obs[-1] = 1.0 if self.in_cue_phase else 0.0
        return obs

    # ------------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    @property
    def act_dim(self) -> int:
        return self.ACT_DIM
