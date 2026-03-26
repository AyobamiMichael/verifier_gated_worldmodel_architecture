import numpy as np


def collect_transitions(env, policy, num_steps=5000):
    data = []

    state = env.reset()

    for _ in range(num_steps):
        action = policy.select_action(state)
        next_state, reward, done, info = env.step(action)

        data.append({
            "state": state,
            "action": action,
            "next_state": next_state,
            "constraint": info["constraint"]
        })

        state = next_state
        # Dataset bias(OOD)
        #if state[0] >= 6 or state[1] >= 6:
           # continue

        if done:
            state = env.reset()

    return data


def to_numpy(data):
    states = np.array([d["state"] for d in data])
    actions = np.array([d["action"] for d in data])
    next_states = np.array([d["next_state"] for d in data])

    return states, actions, next_states