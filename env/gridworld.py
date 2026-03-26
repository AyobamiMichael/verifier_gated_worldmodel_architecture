import numpy as np
from typing import Tuple, List, Dict

class GridWorld:
    def __init__(self,
                 size: int = 10,
                 hazard_positions: List[Tuple[int, int]] = None,
                 goal_position: Tuple[int, int] = (9, 9),
                 start_position: Tuple[int, int] = None,
                 seed: int = 42,

                 ):
                  
                  self.size = size
                  self.rng = np.random.default_rng(seed)

                  self.goal_position = goal_position
                  self.hazard_positions = set(hazard_positions or [])

                  self.start_position = start_position
                  self.agent_position = None
                  

                  self.action_space = [0, 1, 2, 3] # Up, Down, Left, Right  

                  # Hidden mode
                  self.hidden_mode = self.rng.integers(0, 2) 

    
    #  --------------------------
    # Reset Environment
    #  --------------------------

    def reset(self) -> Tuple[int, int]:
        if self.start_position is not None:
               self.agent_position = self.start_position
        else:
            self.agent_position = self._random_safe_position()

        return self.agent_position
    
    # -------------------------
    # Step Function
    # -------------------------
    def step(self, action: int):
        next_position = self._transition(self.agent_position, action)

        reward = self._compute_reward(next_position)
        done = self._is_done(next_position)

        constraint = self._constraint(next_position)

        self.agent_position = next_position

        info = {
            "constraint": constraint,
            "is_hazard": next_position in self.hazard_positions,
        }

        return next_position, reward, done, info
            
    
     # -------------------------
    # Transition Function (True P)
    # -------------------------
    def _transition(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        x, y = state

        # Stochasticity: 10% random action(aleatoric uncertainty)
        #if self.rng.random() < 0.1:
           #  action = self.rng.choice([0,1,2,3])
         
        # Hidden mode flips controls(epistemic misspecification), systematic bias where all the gates fails
        #if self.hidden_mode == 1:
                 #action = self._flip_action(action)


        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1

        # boundary check
        x = np.clip(x, 0, self.size - 1)
        y = np.clip(y, 0, self.size - 1)

        return (x, y)
    
     # -------------------------
    # Reward Function
    # -------------------------
    def _compute_reward(self, state: Tuple[int, int]) -> float:
        if state == self.goal_position:
            return 10.0
        elif state in self.hazard_positions:
            return -5.0
        else:
            return -0.1
     # -------------------------
    # Constraint Function C(s)
    # -------------------------
    def _constraint(self, state: Tuple[int, int]) -> int:
        return 1 if state in self.hazard_positions else 0
    
      # -------------------------
    # Done Condition
    # -------------------------
    def _is_done(self, state: Tuple[int, int]) -> bool:
        return state == self.goal_position
    
      # -------------------------
    # Utility: Safe Random Start
    # -------------------------
    def _random_safe_position(self) -> Tuple[int, int]:
        while True:
            pos = (
                self.rng.integers(0, self.size),
                self.rng.integers(0, self.size),
            )
            if pos not in self.hazard_positions and pos != self.goal_position:
                return pos
    
    
    # -------------------------
    # Public Constraint Checker (for verifier)
    # -------------------------
    def is_hazard(self, state: Tuple[int, int]) -> bool:
        return state in self.hazard_positions
    

     # -------------------------
    # Distance to Goal (for policy)
    # -------------------------
    def manhattan_distance(self, state: Tuple[int, int]) -> int:
        return abs(state[0] - self.goal_position[0]) + abs(state[1] - self.goal_position[1])

   
      # -------------------------
    # Render (Optional Debug)
    # -------------------------
    def render(self):
        grid = np.full((self.size, self.size), ".", dtype=str)

        for (x, y) in self.hazard_positions:
            grid[x, y] = "H"

        gx, gy = self.goal_position
        grid[gx, gy] = "G"

        ax, ay = self.agent_position
        grid[ax, ay] = "A"

        print("\n".join(" ".join(row) for row in grid))
        print()

    # This is for systematic bias(epistemic misspecification), represents a hidden mode where controls are flipped 
    def _flip_action(self, action):
        mapping = {
             0: 1,  # up -> down
             1: 0,
             2: 3,  # left -> right
             3: 2
                  }
        return mapping[action]