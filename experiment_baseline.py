# experiment_baseline.py
from env.gridworld import GridWorld
from experiments.runner import ExperimentRunner
from models.load_model import load_world_model
from policy.goal_bias_policy import GoalBiasPolicy
from verifier.hybrid_verifier import HybridVerifier


model = load_world_model("models/world_model.pth")

env = GridWorld(
    size=10,
    hazard_positions=[(3, 3), (3, 4), (4, 3)],
    goal_position=(9, 9),
    seed=42        # fixed seed for reproducibility
)

policy = GoalBiasPolicy(env, seed=42)
verifier = HybridVerifier(env=env, world_model=model)
runner = ExperimentRunner(env, policy, verifier)
results = runner.run(num_episodes=100)

print("Baseline Results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}")