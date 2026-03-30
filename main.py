from env.gridworld import GridWorld
from models.load_model import load_world_model
from policy.goal_bias_policy import GoalBiasPolicy
from training.collect_data import collect_transitions, to_numpy
from training.train_world_model import train_world_model
from verifier.hybrid_verifier import HybridVerifier
from experiments.runner import ExperimentRunner




model = load_world_model("models/world_model.pth")

env = GridWorld(
    size=10,
    hazard_positions=[(3, 3), (3, 4), (4, 3)],
    goal_position=(9, 9),
    seed=42        # fixed seed for reproducibility
)

policy = GoalBiasPolicy(env, seed=42)
verifier = HybridVerifier(env=env, world_model=model)
#runner = ExperimentRunner(env, policy, verifier)


verifier = HybridVerifier(
    env=env,
    world_model=model,
    ensemble_model=None,  # add later
    critic=None,          # add later
)

state = env.reset()


for step in range(20):
    action = policy.select_action(state)

    safe, info = verifier.is_safe(state, action)

    print(f"Step {step} | Action {action} | Safe: {safe}")
    print(info)
    

    if safe:
        state, _, done, _ = env.step(action)
    else:
        print("Action blocked by verifier")




"""
print("Collecting data...")

env = GridWorld(
    size=10,
    hazard_positions=[(3, 3), (3, 4), (4, 3)],
    goal_position=(9, 9),
)

policy = GoalBiasPolicy(env)

data = collect_transitions(env, policy, num_steps=3000)

states, actions, next_states = to_numpy(data)

print("Training world model...")

model = train_world_model(states, actions, next_states)

print("Training complete.")



verifier = HybridVerifier(
    env=env,
    world_model=model,
    ensemble_model=None,  # add later
    critic=None,          # add later
)

state = env.reset()

for step in range(20):
    action = policy.select_action(state)

    safe, info = verifier.is_safe(state, action)

    print(f"Step {step} | Action {action} | Safe: {safe}")
    print(info)

    if safe:
        state, _, done, _ = env.step(action)
    else:
        print("Action blocked by verifier")
"""