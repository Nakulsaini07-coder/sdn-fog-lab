
"""
Phase 5 - Cell 26A
AI-driven five-action SDN decision bridge.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from stable_baselines3 import DQN, PPO

from ids_preprocessing_adapter_v1 import transform_raw_features
from observation_contract_v1 import build_observation, FIELDS


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results/p5_c26"
RESULTS.mkdir(parents=True, exist_ok=True)

ACTIONS = [
    "ALLOW",
    "INSPECT",
    "RATE_LIMIT",
    "REDIRECT",
    "BLOCK",
]

TRUST_LEVELS = [
    0.10,
    0.25,
    0.30,
    0.50,
    0.75,
    0.90,
]


class HardenedIDS(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(70, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):

        return self.network(x)


def zero_trust(action, risk, trust):

    assert action in range(5)
    assert 0 <= risk <= 1
    assert 0 <= trust <= 1

    if trust < 0.20 and risk > 0.70:

        return 4, "MANDATORY_BLOCK"

    if (
        trust < 0.35
        and risk > 0.55
        and action == 0
    ):

        return 1, "ESCALATE_TO_INSPECT"

    return action, "NO_OVERRIDE"


def make_observation(risk, trust):

    values = {
        "traffic_intensity": 0.50,
        "queue_utilization": 0.40,
        "estimated_latency": 0.35,
        "bandwidth_utilization": 0.45,
        "ids_risk_score": risk,
        "device_trust_score": trust,
        "service_health": 0.90,
        "active_security_policy": 0.00,
    }

    provenance = {
        field: "SYNTHETIC"
        for field in FIELDS
    }

    provenance["ids_risk_score"] = "IDS_INFERENCE"
    provenance["active_security_policy"] = "POLICY_STATE"

    obs = build_observation(values, provenance)

    assert obs.shape == (8,)
    assert np.isfinite(obs).all()

    return obs


def load_models():

    ids_path = (
        ROOT / "models/ids/"
        "mlp_iid_adversarially_trained_v1.pt"
    )

    actual_hash = hashlib.sha256(
        ids_path.read_bytes()
    ).hexdigest()

    expected_hash = (
        "d2c802de79157875e371ef4874e0d103f1"
        "e9dcf04db387f203d233bcd015a090"
    )

    assert actual_hash == expected_hash, (
        "Hardened IDS checkpoint mismatch"
    )

    checkpoint = torch.load(
        ids_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["input_dim"] == 70
    assert float(checkpoint["threshold"]) == 0.52

    ids = HardenedIDS()

    ids.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    ids.eval()

    preprocessor = joblib.load(
        ROOT / "models/ids/preprocessor_iid_v1.joblib"
    )

    assert len(
        preprocessor["raw_feature_columns"]
    ) == 78

    assert len(
        preprocessor["selected_feature_columns"]
    ) == 70

    models = {
        "DQN": DQN.load(
            str(
                ROOT / "models/rl/"
                "dqn_digital_twin_best_v1.zip"
            ),
            device="cpu",
        ),
        "PPO": PPO.load(
            str(
                ROOT / "models/rl/"
                "ppo_digital_twin_best_v1.zip"
            ),
            device="cpu",
        ),
    }

    for model in models.values():

        assert model.observation_space.shape == (8,)
        assert model.action_space.n == 5

    print("[PASS] Frozen AI models loaded")

    return ids, preprocessor, models, actual_hash


def generate_ids_scores(ids, preprocessor):

    fixture = (
        ROOT / "fixtures/"
        "p5_c23c_verified_raw_samples_v1.json"
    )

    cohort = json.loads(fixture.read_text())

    assert cohort["cell"] == "P5_C23C_STEP3"
    assert cohort["status"] == "PASS"
    assert cohort["split"] == "IID_TEST"
    assert cohort["full_label_alignment_passed"] is True

    samples = cohort["records"]

    assert len(samples) == 69

    records = []

    for sample in samples:

        assert sample["iid_split"] == "TEST"

        x = transform_raw_features(
            sample["raw_features"],
            preprocessor,
        )

        assert x.shape == (1, 70)

        with torch.no_grad():

            risk = float(
                torch.sigmoid(
                    ids(torch.from_numpy(x).float())
                ).item()
            )

        assert np.isfinite(risk)
        assert 0 <= risk <= 1

        records.append({
            "global_row_id": int(
                sample["global_row_id"]
            ),
            "attack_type_evaluation_only":
                sample["attack_type"],
            "ground_truth_evaluation_only":
                int(sample["binary_label"]),
            "ids_risk": risk,
        })

    print(
        f"[PASS] {len(records)} held-out IDS scores generated"
    )

    return records


def make_decision(
    record,
    trust,
    action,
    model_name,
    decision_source,
):

    risk = record["ids_risk"]

    final, reason = zero_trust(
        action,
        risk,
        trust,
    )

    observation = make_observation(risk, trust)

    return {
        **record,
        "model": model_name,
        "decision_source": decision_source,
        "device_trust_synthetic": trust,
        "observation": observation.tolist(),
        "requested_action_id": action,
        "requested_action": ACTIONS[action],
        "final_action_id": final,
        "final_action": ACTIONS[final],
        "zero_trust_reason": reason,
        "action_changed": action != final,
    }


def main():

    print("=" * 75)
    print("P5_C26A | AI-DRIVEN FIVE-ACTION DECISION BRIDGE")
    print("=" * 75)

    torch.set_num_threads(1)

    ids, preprocessor, models, checkpoint_hash = (
        load_models()
    )

    records = generate_ids_scores(
        ids,
        preprocessor,
    )

    natural = []

    # Natural deterministic RL decisions.
    for record in records:

        for trust in TRUST_LEVELS:

            obs = make_observation(
                record["ids_risk"],
                trust,
            )

            for model_name, model in models.items():

                predicted, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                action = int(
                    np.asarray(predicted).item()
                )

                natural.append(
                    make_decision(
                        record,
                        trust,
                        action,
                        model_name,
                        "NATURAL_RL",
                    )
                )

    assert len(natural) == 69 * 6 * 2

    print(
        f"[PASS] {len(natural)} natural RL decisions"
    )

    summary = {}

    for model_name in models:

        subset = [
            r for r in natural
            if r["model"] == model_name
        ]

        requested = Counter(
            r["requested_action"] for r in subset
        )

        final = Counter(
            r["final_action"] for r in subset
        )

        reasons = Counter(
            r["zero_trust_reason"] for r in subset
        )

        summary[model_name] = {
            "decisions": len(subset),
            "requested_actions": {
                a: requested[a] for a in ACTIONS
            },
            "final_actions": {
                a: final[a] for a in ACTIONS
            },
            "zero_trust_reasons": dict(reasons),
            "action_changes": sum(
                r["action_changed"] for r in subset
            ),
        }

        print("\nMODEL:", model_name)

        print(
            "Requested:",
            summary[model_name]["requested_actions"],
        )

        print(
            "Final:",
            summary[model_name]["final_actions"],
        )

        print("Zero trust:", dict(reasons))

    # Select by IDS score, not ground-truth label.
    safe = min(
        records,
        key=lambda r: r["ids_risk"],
    )

    high = max(
        records,
        key=lambda r: r["ids_risk"],
    )

    assert safe["ids_risk"] < 0.55
    assert high["ids_risk"] > 0.70

    # Controlled action coverage.
    # These are NOT naturally predicted RL actions.
    controlled = []

    for action in range(5):

        decision = make_decision(
            safe,
            0.90,
            action,
            None,
            "CONTROLLED_ACTION_INJECTION",
        )

        assert decision["final_action_id"] == action

        controlled.append(decision)

    assert {
        r["final_action"] for r in controlled
    } == set(ACTIONS)

    print("[PASS] All five controlled action paths")

    # Explicit zero-trust intervention tests.
    override_tests = []

    for action in range(5):

        decision = make_decision(
            high,
            0.10,
            action,
            None,
            "CONTROLLED_ZERO_TRUST_TEST",
        )

        assert decision["final_action"] == "BLOCK"
        assert (
            decision["zero_trust_reason"]
            == "MANDATORY_BLOCK"
        )

        override_tests.append(decision)

    inspect_test = make_decision(
        high,
        0.30,
        0,
        None,
        "CONTROLLED_ZERO_TRUST_TEST",
    )

    assert inspect_test["final_action"] == "INSPECT"

    assert (
        inspect_test["zero_trust_reason"]
        == "ESCALATE_TO_INSPECT"
    )

    override_tests.append(inspect_test)

    print("[PASS] Six zero-trust intervention tests")

    # Select representative natural decisions
    # for downstream SDN verification.
    selected = []

    for model_name in models:

        for action in ACTIONS:

            candidates = [
                r for r in natural
                if r["model"] == model_name
                and r["final_action"] == action
            ]

            if candidates:

                selected.append(candidates[0])

                print(
                    "[SELECTED]",
                    model_name,
                    action,
                )

    plan = []

    for index, decision in enumerate(
        selected + controlled + override_tests,
        start=1,
    ):

        plan.append({
            "scenario_id": f"C26_{index:03d}",
            **decision,
            "sdn_command": {
                "action": decision["final_action"],
            },
        })

    evidence = {
        "cell": "P5_C26A",
        "status": "PASS",
        "ids_checkpoint_sha256": checkpoint_hash,
        "cohort_size": len(records),
        "trust_levels_synthetic": TRUST_LEVELS,
        "natural_decision_count": len(natural),
        "controlled_action_count": len(controlled),
        "override_test_count": len(override_tests),
        "labels_used_as_model_inputs": False,
        "live_traffic_feature_extraction": False,
        "sdn_dataplane_verified": False,
        "summary": summary,
    }

    outputs = {
        "ids_scores.json": records,
        "natural_rl_decisions.json": natural,
        "controlled_action_tests.json": controlled,
        "zero_trust_tests.json": override_tests,
        "sdn_execution_plan.json": plan,
        "decision_bridge_summary.json": evidence,
    }

    for filename, data in outputs.items():

        path = RESULTS / filename

        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

        print("[SAVED]", filename)

    print("\nSDN execution plan size:", len(plan))

    print("=" * 75)
    print("P5_C26A: AI DECISION BRIDGE — PASS")
    print("=" * 75)


if __name__ == "__main__":

    main()
