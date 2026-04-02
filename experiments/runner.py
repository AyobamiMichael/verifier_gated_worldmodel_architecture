import numpy as np


class ExperimentRunner:
    def __init__(self, env, policy, verifier):
        self.env = env
        self.policy = policy
        self.verifier = verifier

    # -------------------------
    # Run Single Episode
    # -------------------------
    def run_episode(self, max_steps=50, condition="None"):
        #state = self.env.reset()
        if condition == "ood":
              self.env.reset()
              self.env.agent_position = (7, 7)
              state = self.env.agent_position
        else:
             state = self.env.reset()


        metrics = {
            "violations": 0,
            "blocked": 0,
            "total": 0,
            "false_negatives": 0,
            "false_positives": 0,
        }

        for _ in range(max_steps):
            action = self.policy.select_action(state)

            safe, _ = self.verifier.is_safe(state, action)
            if safe:
                next_state, _, done, info = self.env.step(action)
            else:
                next_state = state
                done = False
                info = {"constraint": 0}  # Assume constraint info is 0 when blocked

            # Ground truth next state (real env)
            #next_state, _, done, info = self.env.step(action)

            is_violation = info["constraint"] == 1

            # -------------------------
            # Metrics accounting
            # -------------------------
            metrics["total"] += 1

            if not safe:
                metrics["blocked"] += 1

                # False positive: safe but blocked
                if not is_violation:
                    metrics["false_positives"] += 1

                continue  # action not executed

            # If action allowed → we already stepped env
            if is_violation:
                metrics["violations"] += 1
                metrics["false_negatives"] += 1  # unsafe but allowed

            state = next_state

            if done:
                break

        return metrics

    # -------------------------
    # Run Multiple Episodes
    # -------------------------
    def run(self, num_episodes=50, condition="None"):
        if condition == "ood":
              self.env.reset()
              self.env.agent_position = (7, 7)
              state = self.env.agent_position
        else:
             state = self.env.reset()

        aggregate = {
            "violations": 0,
            "blocked": 0,
            "total": 0,
            "false_negatives": 0,
            "false_positives": 0,
        }

        for _ in range(num_episodes):
            ep_metrics = self.run_episode()

            for k in aggregate:
                aggregate[k] += ep_metrics[k]

        # -------------------------
        # Normalize
        # -------------------------
        results = {
            "violation_rate": aggregate["violations"] / aggregate["total"],
            "block_rate": aggregate["blocked"] / aggregate["total"],
            "false_negative_rate": aggregate["false_negatives"] / aggregate["total"],
            "false_positive_rate": aggregate["false_positives"] / aggregate["total"],
        }
        print(f"Results for condition {condition}: {results}")

        return results

