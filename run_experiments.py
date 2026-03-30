
from env.gridworld import GridWorld
from models.ensemble import EnsembleModel
from models.load_model import load_world_model
from policy.goal_bias_policy import GoalBiasPolicy
from experiments.runner import ExperimentRunner
from training.train_critic import train_critic
from main_train_world_model import prepare_data
from verifier.hybrid_verifier import HybridVerifier


# load world model

#model = load_world_model()

env = GridWorld(
    size=10,
    hazard_positions=[(3, 3), (3, 4), (4, 3)],
    goal_position = (9, 9),
    seed = 42
)

# ensemble(Uncertainty estimation)

models = [load_world_model(f"models/world_model_{i}.pth") for i in range(3)]
ensemble = EnsembleModel(models)

# Critic(Risk estimation)
env, policy, states, actions, next_states = prepare_data()
critic = train_critic(states, actions, next_states, env)

policy = GoalBiasPolicy(env,  seed =42)

verifier = HybridVerifier(env=env, world_model=models[0], ensemble_model=ensemble, critic=critic)
runner = ExperimentRunner(env, policy, verifier)
results = runner.run(num_episodes=100)

print("Baseline Results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}")