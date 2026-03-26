import torch


class HybridVerifier:
    def __init__(
        self,
        env,
        world_model,
        ensemble_model=None,
        critic=None,
        risk_threshold=0.5,
        uncertainty_threshold=0.1,
        device="cpu",
    ):
        self.env = env
        self.world_model = world_model
        self.ensemble_model = ensemble_model
        self.critic = critic

        self.risk_threshold = risk_threshold
        self.uncertainty_threshold = uncertainty_threshold

        self.device = device

    # -------------------------
    # Main Verification Function
    # -------------------------
    def is_safe(self, state, action):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_t = torch.tensor([action]).to(self.device)

        action_onehot = torch.nn.functional.one_hot(action_t, num_classes=4).float()

        # -------------------------
        # 1. Predict next state (world model)
        # -------------------------
        pred_next = self.world_model(state_t, action_onehot)

        pred_next_np = pred_next.detach().cpu().numpy()[0]
        pred_next_discrete = tuple(pred_next_np.round().astype(int))

        # -------------------------
        # 2. Rule-based check (approx constraint)
        # -------------------------
        rule_ok = not self.env.is_hazard(pred_next_discrete)

        # -------------------------
        # 3. Critic risk estimate
        # -------------------------
        critic_ok = True
        risk_value = 0.0

        if self.critic is not None:
            risk_value = self.critic(state_t, action_onehot).item()
            critic_ok = risk_value < self.risk_threshold

        # -------------------------
        # 4. Ensemble uncertainty
        # -------------------------
        uncertainty_ok = True
        uncertainty_value = 0.0

        if self.ensemble_model is not None:
            preds = self.ensemble_model.predict(state_t, action_onehot)
            uncertainty_value = preds.var(dim=0).mean().item()
            uncertainty_ok = uncertainty_value < self.uncertainty_threshold

        # -------------------------
        # Final decision
        # -------------------------
        is_safe = rule_ok and critic_ok and uncertainty_ok

        info = {
            "rule_ok": rule_ok,
            "critic_ok": critic_ok,
            "uncertainty_ok": uncertainty_ok,
            "risk_value": risk_value,
            "uncertainty_value": uncertainty_value,
        }

        return is_safe, info