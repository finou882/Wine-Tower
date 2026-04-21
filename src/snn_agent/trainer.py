"""
Recursive Lifelong Learning Trainer
-------------------------------------
Trains the SNN agent on Multiple-TMaze using the goal-memory curriculum.

"Recursive lifelong learning" here means:
  * The weight matrices are *never reset* between episodes.
  * STDP continuously rewrites synaptic weights based on the reward signal.
  * The curriculum progressively increases task demands, forcing the network
    to consolidate earlier memories while learning new ones (catastrophic
    forgetting is expected and studied rather than mitigated).
  * Every `replay_interval` episodes a short "replay pass" re-exposes the
    agent to a random subset of previously seen goal sequences (offline
    consolidation, analogous to hippocampal replay during sleep).
"""

import numpy as np
from typing import List, Optional, Dict

from .environment import MultipleTMaze
from .agent import SNNAgent
from .curriculum import GoalCurriculum


class LifelongTrainer:
    """
    Parameters
    ----------
    agent           : SNNAgent
    env             : MultipleTMaze
    curriculum      : GoalCurriculum
    max_steps       : int   max steps per episode
    replay_interval : int   every N episodes replay memory buffer
    replay_size     : int   number of goal sequences kept in replay buffer
    verbose_every   : int   print stats every N episodes
    epsilon_start   : float initial exploration rate (Phase 1 start)
    epsilon_end     : float final exploration rate (Phase 3 end)
    """

    def __init__(
        self,
        agent: SNNAgent,
        env: MultipleTMaze,
        curriculum: GoalCurriculum,
        max_steps: int = 50,
        replay_interval: int = 20,
        replay_size: int = 100,
        verbose_every: int = 50,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        wine_tower: bool = True,
    ):
        self.agent = agent
        self.env = env
        self.curriculum = curriculum
        self.max_steps = max_steps
        self.replay_interval = replay_interval
        self.verbose_every = verbose_every

        # Replay memory: list of (cue_goals, target)
        self._replay_buf: List[tuple] = []
        self._replay_size = replay_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self._wine_tower_enabled = wine_tower
        self._wine_tower_interval = 20   # run Wine-Tower pass every N episodes

        # Statistics
        self.history: Dict[str, List] = {
            "episode": [],
            "success": [],
            "reward": [],
            "n_dead_hidden": [],
            "n_dead_output": [],
            "phase": [],
        }

    # ------------------------------------------------------------------
    def _epsilon(self, ep: int, total: int) -> float:
        """Linear decay from epsilon_start to epsilon_end."""
        t = ep / max(1, total - 1)
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def _run_episode(
        self,
        cue_goals: List[int],
        target: Optional[int] = None,
        learn: bool = True,
        epsilon: float = 0.0,
    ) -> tuple:
        """
        Run one episode.

        Returns (total_reward, trajectory) where trajectory is the list of
        (obs, reward) pairs from the forward pass (used by Wine-Tower replay).

        Reward propagation strategy:
          * Run a forward pass collecting (obs, reward) pairs.
          * Then replay the trajectory with STDP updates gated by rewards.
          * Per-step shaped rewards (+0.1 per correct junction) propagate
            immediately; terminal reward (+1 / -1) is broadcast over last steps.

        Returns terminal reward (1.0 = success, -1.0 = failure, accumulated).
        """
        obs = self.env.reset(active_goals=cue_goals)
        if target is not None:
            self.env.target_goal = target

        self.agent.reset_episode()

        # Forward collection pass (no STDP to keep traces clean)
        trajectory: List[tuple] = []   # (obs, reward_after_action)
        total_reward = 0.0
        terminal_reward = 0.0

        for _ in range(self.max_steps):
            action = self.agent.act(obs, reward=0.0, learn=False, epsilon=epsilon)
            prev_obs = obs.copy()
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            trajectory.append((prev_obs, reward))
            if done:
                terminal_reward = reward
                break

        # STDP update pass: phase-aware reward shaping.
        #
        # Phase 1/2: clip reward to [0, +∞) — pure positive reinforcement.
        #   Failure (-1.0) is ignored so LTD never suppresses learning.
        #   Even 0.4% success rate gives enough LTP signal to consolidate
        #   goal-anchor neurons (each correct junction = +0.1 signal).
        # Phase 3: full reward including -1.0 penalty.
        #   Novel goals → frequent failure → LTD dominates → cascade.
        if learn:
            phase = self.curriculum.phase
            self.agent.reset_episode()
            for step_obs, step_reward in trajectory:
                if step_reward > 0:
                    effective = step_reward * 20.0   # LTP強め
                else:
                    effective = step_reward * 2.0    # LTD弱め（崩壊防止）
                self.agent.act(step_obs, reward=effective, learn=True)

        return total_reward, trajectory

    # ------------------------------------------------------------------
    def _replay_pass(self) -> None:
        """Re-run a sample of stored (cue, target) pairs without logging."""
        if not self._replay_buf:
            return
        n = min(len(self._replay_buf), 5)
        indices = np.random.default_rng().choice(len(self._replay_buf),
                                                  size=n, replace=False)
        for idx in indices:
            cues, target, _, _r = self._replay_buf[int(idx)]
            self._run_episode(cues, target=target, learn=True)

    # ------------------------------------------------------------------
    def _wine_tower_pass(
        self,
        wine_strength: float = 50.0,
        n_passes: int = 2,
    ) -> None:
        """
        Wine-Tower offline replay: spike-triggered current injection into dead
        hidden neurons via positive W_rec connections from live neighbours.
        Only runs when dead neurons are present.
        """
        if not self.agent.hidden.dead_mask.any():
            return
        if not self._replay_buf:
            return
        # Select top-K highest-reward episodes for targeted replay
        k = min(len(self._replay_buf), 5)
        sorted_buf = sorted(self._replay_buf, key=lambda x: x[3], reverse=True)
        top_trajs = [t for _, _, t, _ in sorted_buf[:k]]
        self.agent.wine_tower_replay(
            top_trajs, wine_strength=wine_strength, n_passes=n_passes
        )

    # ------------------------------------------------------------------
    def train(self, total_episodes: int,
              max_phase: int = 3) -> Dict[str, List]:
        """
        Run the full training loop.

        Parameters
        ----------
        max_phase : int  stop training when curriculum advances past this phase

        Returns
        -------
        history dict with per-episode stats.
        """
        for ep in range(total_episodes):
            cue_goals, target = self.curriculum.next_episode_goals()
            phase = self.curriculum.phase

            # Stop early if phase exceeds max_phase
            if phase > max_phase:
                break
            # Junction curriculum (Phase 1 only):
            # Slow start: stay at 1-2 junctions for the first half of Phase 1
            # so STDP consolidates goal-anchor neurons before difficulty ramps.
            #   ep 0   → 600:  1 junction  (P=33%)
            #   ep 600 → 900:  2 junctions (P=11%)
            #   ep 900 → 1050: 3 junctions (P=3.7%)
            #   ep1050 → 1200: 4-5 junctions
            if phase == 1:
                phase1_len = max(1, self.curriculum.phase1_end)
                progress = ep / phase1_len  # 0.0 → 1.0
                # Quadratic ramp: spends more time at low difficulty
                n_junc = max(1, round(1 + (progress ** 2) * (5 - 1)))
            else:
                n_junc = 5
            self.env.n_active_junctions = n_junc

            # Phase-aware WTA-k:
            # Phase 1/2 – large k (half of hidden) so most neurons survive
            #   long enough for STDP to specialise on the anchor goal.
            # Phase 3  – moderate k (12.5% of hidden) so Wine-Tower recovery
            #   can compete with WTA-induced death and demonstrate revival.
            n_hidden = self.agent.n_hidden
            self.agent.hidden.k = n_hidden // 2 if phase <= 2 else max(1, n_hidden // 8)

            # Phase-aware homeostatic kick:
            # Phase 1/2 – gently depolarise dead neurons so STDP has time to
            #   learn before WTA causes irreversible collapse.
            # Phase 3  – mild kick to let Wine-Tower have a chance to revive
            #   dead neurons via neighbour-voltage leaking + STDP.
            if phase <= 2:
                self.agent.hidden.homeostatic_kick(strength=0.4)
                self.agent.output.homeostatic_kick(strength=0.2)
            else:
                self.agent.hidden.homeostatic_kick(strength=0.1)

            eps = self._epsilon(ep, total_episodes)
            reward, traj = self._run_episode(cue_goals, target=target, learn=True, epsilon=eps)
            success = reward >= 1.0   # only terminal +1 counts as success

            # Store in replay buffer (FIFO) – includes trajectory and reward for Wine-Tower
            self._replay_buf.append((list(cue_goals), target, traj, reward))
            if len(self._replay_buf) > self._replay_size:
                self._replay_buf.pop(0)

            # Record stats
            self.history["episode"].append(ep)
            self.history["success"].append(int(success))
            self.history["reward"].append(reward)
            self.history["n_dead_hidden"].append(self.agent.n_dead_hidden)
            self.history["n_dead_output"].append(self.agent.n_dead_output)
            self.history["phase"].append(self.curriculum.phase)

            # Periodic replay
            if (ep + 1) % self.replay_interval == 0:
                self._replay_pass()

            # Wine-Tower recovery: Phase 3 only, every 5 episodes
            # Phase 1/2 uses homeostatic_kick instead (simpler, faster)
            if self._wine_tower_enabled and phase == 3 and (ep + 1) % 5 == 0:
                self._wine_tower_pass()

            # Logging
            if (ep + 1) % self.verbose_every == 0:
                window = 50
                recent = self.history["success"][max(0, ep - window):]
                acc = np.mean(recent) * 100
                nd_h = self.agent.n_dead_hidden
                nd_o = self.agent.n_dead_output
                phase_desc = self.curriculum.phase_description()
                print(
                    f"[Ep {ep+1:5d}/{total_episodes}] "
                    f"acc(last{window})={acc:5.1f}%  "
                    f"dead_hidden={nd_h:3d}  dead_out={nd_o}  "
                    f"| {phase_desc}"
                )

        return self.history
