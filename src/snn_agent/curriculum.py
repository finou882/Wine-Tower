"""
Goal-Memory Curriculum
-----------------------
Implements the progressive curriculum that trains long-range memory:

Phase 1  (episodes  0  … phase1_end):
    Each episode draws 3 random goals from the pool of 10.
    The agent must navigate to the *single* correct target within the episode.

Phase 2  (episodes phase1_end … phase2_end):
    Gradually increase the "look-ahead gap":
    At episode E, the agent is shown a cue set at episode E,
    but the *target* is drawn from the cue set shown
    `gap` episodes ago (gap grows from 1 → max_gap linearly).

Phase 3  (episodes phase2_end … total_episodes):
    Full recall: the agent must remember & navigate to any of the
    10 goals presented as a batch hint at the start of this phase,
    without receiving new cues each episode.

If ``fixed_goals`` is set, the curriculum is bypassed and the fixed
goal set is used for every episode.
"""

import numpy as np
from collections import deque
from typing import Deque, List, Optional, Tuple


N_GOALS = 10


class GoalCurriculum:
    """
    Parameters
    ----------
    total_episodes  : int   total training budget
    n_hint_goals    : int   goals shown each episode in phase 1/2 (default 3)
    max_gap         : int   longest look-ahead gap in episodes (default 50)
    phase1_frac     : float fraction of budget spent in phase 1 (default 0.40)
    phase2_frac     : float fraction of budget spent in phase 2 (default 0.30)
    fixed_goals     : list[int] or None
                       if given, bypass curriculum; always present these goals.
    anchor_goal     : int or None
                       if given, this goal ID is always included in the hint
                       cue set (the remaining n_hint-1 slots are random).
    seed            : int
    """

    def __init__(
        self,
        total_episodes: int = 1000,
        n_hint_goals: int = 3,
        max_gap: int = 50,
        phase1_frac: float = 0.40,
        phase2_frac: float = 0.30,
        fixed_goals: Optional[List[int]] = None,
        anchor_goal: Optional[int] = None,
        seed: int = 42,
    ):
        self.total = total_episodes
        self.n_hint = n_hint_goals
        self.max_gap = max_gap
        self.fixed_goals = fixed_goals
        self.anchor_goal = anchor_goal

        p1 = int(total_episodes * phase1_frac)
        p2 = p1 + int(total_episodes * phase2_frac)
        self.phase1_end = p1
        self.phase2_end = p2

        self.rng = np.random.default_rng(seed)

        # History buffer for look-ahead curriculum
        self._history: Deque[List[int]] = deque(maxlen=max_gap + 10)
        self._episode = 0

    # ------------------------------------------------------------------
    def _current_phase(self) -> int:
        e = self._episode
        if e < self.phase1_end:
            return 1
        elif e < self.phase2_end:
            return 2
        else:
            return 3

    # ------------------------------------------------------------------
    def _current_gap(self) -> int:
        """Linearly increase gap from 1 → max_gap across phase 2."""
        phase2_len = max(1, self.phase2_end - self.phase1_end)
        progress = (self._episode - self.phase1_end) / phase2_len
        return max(1, int(progress * self.max_gap))

    # ------------------------------------------------------------------
    def _sample_cues(self) -> List[int]:
        """
        Sample n_hint goal IDs.
        If anchor_goal is set, it is always included;
        remaining slots are filled from the other N_GOALS-1 goals.
        """
        if self.anchor_goal is not None:
            others = [g for g in range(N_GOALS) if g != self.anchor_goal]
            fill = list(self.rng.choice(others,
                                        size=self.n_hint - 1,
                                        replace=False))
            cues = [self.anchor_goal] + fill
            self.rng.shuffle(cues)   # randomise position within cue
            return cues
        return list(self.rng.choice(N_GOALS, size=self.n_hint, replace=False))

    # ------------------------------------------------------------------
    def next_episode_goals(self) -> Tuple[List[int], int]:
        """
        Decide which goals to show as cues and which single target to pursue.

        Returns
        -------
        cue_goals  : list[int]   goal IDs the agent sees as hints this episode
        target     : int         goal ID the agent must actually reach
        """
        if self.fixed_goals is not None:
            target = int(self.rng.choice(self.fixed_goals))
            self._episode += 1
            return list(self.fixed_goals), target

        phase = self._current_phase()

        if phase == 1:
            cues = self._sample_cues()
            self._history.append(cues)
            # If anchor_goal is set, always train toward it in Phase 1/2
            # so WTA specialises neurons for that goal before Phase 3.
            target = self.anchor_goal if self.anchor_goal is not None else int(self.rng.choice(cues))

        elif phase == 2:
            gap = self._current_gap()
            cues = self._sample_cues()
            self._history.append(cues)
            if self.anchor_goal is not None:
                target = self.anchor_goal
            elif len(self._history) > gap:
                old_cues = list(self._history)[-gap - 1]
                target = int(self.rng.choice(old_cues))
            else:
                target = int(self.rng.choice(cues))

        else:   # phase 3 – all 10 goals active
            cues = list(range(N_GOALS))
            self._history.append(cues)
            target = int(self.rng.integers(N_GOALS))

        self._episode += 1
        return cues, target

    # ------------------------------------------------------------------
    @property
    def episode(self) -> int:
        return self._episode

    @property
    def phase(self) -> int:
        return self._current_phase()

    def phase_description(self) -> str:
        p = self._current_phase()
        if p == 1:
            return f"Phase1(3-goal cue, ep {self._episode}/{self.phase1_end})"
        elif p == 2:
            gap = self._current_gap()
            return (f"Phase2(look-ahead gap={gap}, "
                    f"ep {self._episode}/{self.phase2_end})")
        else:
            return f"Phase3(all-10-goals recall, ep {self._episode})"
