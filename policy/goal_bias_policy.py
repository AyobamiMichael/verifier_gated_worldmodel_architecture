import numpy as np
from typing import Tuple, List

class GoalBiasPolicy:
    def __init__(self, env, goal_bias_prob: float = 0.7, seed: int = 42):
        self.env = env
        self.p = goal_bias_prob
        self.rng = np.random.default_rng(seed)

        self.actions = [0, 1, 2, 3]  # up, down, left, right
    
     # -------------------------
    # Select Action
    # -------------------------
    def select_action(self, state: Tuple[int, int]) -> int:
        if self.rng.random() < self.p:
            return self._goal_directed_action(state)
        else:
            return self.rng.choice(self.actions)
    
     # -------------------------
    # Goal-Directed Action
    # -------------------------
    def _goal_directed_action(self, state: Tuple[int, int]) -> int:
        current_distance = self.env.manhattan_distance(state)

        best_actions: List[int] = []

        for action in self.actions:
            next_state = self._simulate(state, action)
            dist = self.env.manhattan_distance(next_state)

            if dist < current_distance:
                best_actions.append(action)

        # If multiple good actions → choose randomly
        if best_actions:
            return self.rng.choice(best_actions)

        # If no improving action → fallback random
        return self.rng.choice(self.actions)

    # -------------------------
    # Simulate 1-step (true env dynamics)
    # -------------------------
    def _simulate(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        x, y = state

        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1

        x = np.clip(x, 0, self.env.size - 1)
        y = np.clip(y, 0, self.env.size - 1)

        return (x, y)

    