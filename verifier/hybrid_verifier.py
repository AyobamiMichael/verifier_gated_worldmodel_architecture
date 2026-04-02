import torch
import torch.nn.functional as F


class HybridVerifier:
    def __init__(
        self,
        env,
        world_model,
        ensemble_model=None,
        critic=None,
        risk_threshold=0.5,
        uncertainty_threshold=0.5,   # 🔧 increased (0.1 was too strict)
        device="cpu",
        mode = "baseline"
    ):
        self.env = env
        self.world_model = world_model
        self.ensemble_model = ensemble_model
        self.critic = critic

        self.risk_threshold = risk_threshold
        self.uncertainty_threshold = uncertainty_threshold

        self.device = device
        self.mode = mode

    
    def inject_epistemic_bias(self, pred_next_state, state):
        """
        Force systematic WRONG prediction
        Moves prediction away from hazards
        Deceiving the verifier into thinking the action is safe when it's not
        """
        bias = torch.tensor([9.0, 9.0])  # strong shift
        return pred_next_state + bias

    def is_safe(self, state, action):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_t = torch.tensor([action]).to(self.device)
        action_onehot = F.one_hot(action_t, num_classes=4).float()

        # -------------------------
        # 1. World model prediction
        # -------------------------
        pred_next = self.world_model(state_t, action_onehot)
        # Inject epistemic bias into main prediction
        if self.mode == "epistemic":
            pred_next = self.inject_epistemic_bias(pred_next, state_t)
            critic_state = pred_next.detach()  # Use biased prediction for critic
            risk_value = self.critic(critic_state, action_onehot).item()
        else:
            critic_state = state_t  # Use current state for critic
            risk_value = self.critic(critic_state, action_onehot).item()

        pred_next_np = pred_next.detach().cpu().numpy()[0]

        # 🔧 tolerance instead of hard rounding
        pred_next_discrete = tuple(pred_next_np.round().astype(int))

        # -------------------------
        # 2. Rule-based check
        # -------------------------
        rule_ok = not self.env.is_hazard(pred_next_discrete)

        # -------------------------
        # 3. Critic risk estimate
        # -------------------------
        critic_ok = True
        risk_value = 0.0

        if self.critic is not None:
            with torch.no_grad():
                risk_value = self.critic(state_t, action_onehot).item()
            critic_ok = risk_value < self.risk_threshold

        # -------------------------
        # 4. Ensemble uncertainty
        # -------------------------
        uncertainty_ok = True
        uncertainty_value = 0.0

        if self.ensemble_model is not None:
            preds = self.ensemble_model.predict(state_t, action_onehot)  # [N, 2]

            if self.mode == "epistemic":
                # Apply bias to each model prediction
                #uncertainty_ok = True
                preds = torch.stack([
                   self.inject_epistemic_bias(p, state_t)
                   for p in preds
                 ])

                  # Collapse uncertainty (force agreement)
                mean_pred = preds.mean(dim=0)
                preds = torch.stack([mean_pred for _ in range(preds.shape[0])])

            # 🔧 better uncertainty: mean variance across dimensions
            uncertainty_value = preds.var(dim=0).mean().item()

            uncertainty_ok = uncertainty_value < self.uncertainty_threshold

        # -------------------------
        # 🔥 Conservative gating logic
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