
import torch
from env.gridworld import GridWorld
from policy.goal_bias_policy import GoalBiasPolicy
from training.collect_data import collect_transitions, to_numpy
from training.train_world_model import bootstrap_sample, train_world_model




def prepare_data():
    
    env = GridWorld(size=10,
                hazard_positions=[(3,3),(3,4),(4,3)],
                goal_position=(9,9))

    policy = GoalBiasPolicy(env)

    data = collect_transitions(env, policy, num_steps=3000)
    states, actions, next_states = to_numpy(data)
    
    return env, policy, states, actions, next_states



models = []
if __name__ == "__main__":
    print("Training pipeline...")
    env, policy, states, actions, next_states = prepare_data()

    for i in range(3):
        torch.manual_seed(i)

        s, a, ns = bootstrap_sample(states, actions, next_states)
        model = train_world_model(
            s, a, ns,
            save_path=f"models/world_model_{i}.pth"
       )
    models.append(model)

