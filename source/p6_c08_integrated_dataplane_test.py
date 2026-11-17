"""Phase 6: replay frozen AI security decisions in an isolated Mininet lab."""

import hashlib
import json
import math
import time

from collections import Counter
from pathlib import Path

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.log import setLogLevel

import p5_c25_five_action_test as test

from p5_c26_integrated_dataplane_test import verify_action


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results/p6_c08"
EVIDENCE = RESULTS / "dataplane"

EVIDENCE.mkdir(parents=True, exist_ok=True)

PLAN_FILE = (
    ROOT / "fixtures/p6_sdn_enforcement_test_plan_v1.json"
)

EXPECTED_PLAN_SHA256 = "7148af2e693804236e41f73fbfcc5574a599e01acbc47bc23d7ff1f123537d48"

AUDITED_BASE_COMMIT = "81129c2a8b93449b5b4a96a204296826263793e9"

EXPECTED_SOURCE_HASHES = {'source/p5_c25_five_action_controller.py': 'b1f438d45aa956d719ed9759fa3e055f7ebec5e4a9cb19662bfe9c6639048956', 'source/p5_c25_five_action_test.py': '02da168c9b439c06ea79b8624c80d2ef4a7d4b2c3b3851eb3a2f39b070ca943d', 'source/p5_c26_integrated_dataplane_test.py': '8548c6e101328cff098068c9f56cbf71cd6ff3fc89409017af47b0187dd3c38e'}

CRITERIA = {'ALLOW': {'packet_loss_pct': 0.0}, 'INSPECT': {'packet_loss_pct': 0.0, 'minimum_mirrored_packets': 5}, 'RATE_LIMIT': {'configured_rate_kbps': 1000, 'minimum_baseline_mbps': 8.0, 'minimum_limited_mbps': 0.5, 'maximum_limited_mbps': 1.3, 'minimum_reduction_pct': 80.0}, 'REDIRECT': {'destination_packet_loss_pct': 100.0, 'minimum_diverted_packets': 5}, 'BLOCK': {'packet_loss_pct': 100.0}, 'RECOVERY': {'packet_loss_pct': 0.0}}

ACTIONS = [
    "ALLOW",
    "INSPECT",
    "RATE_LIMIT",
    "REDIRECT",
    "BLOCK",
]

records = []

baseline_metrics = {}
recovery_metrics = {}


def sha256_file(path):

    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_sources():

    for filename, expected in EXPECTED_SOURCE_HASHES.items():

        path = ROOT / filename

        assert path.is_file(), filename
        assert sha256_file(path) == expected, filename

        print("[PASS] Frozen source:", filename)

    assert sha256_file(PLAN_FILE) == EXPECTED_PLAN_SHA256

    print("[PASS] Frozen Phase-6 plan hash.")


def load_plan():

    document = json.loads(PLAN_FILE.read_text())

    assert document["scenario_count"] == 15

    plan = document["scenarios"]

    assert len(plan) == 15

    assert len({
        row["scenario_id"] for row in plan
    }) == 15

    assert len({
        row["global_row_id"] for row in plan
    }) == 15

    assert Counter(
        row["final_action"] for row in plan
    ) == {action: 3 for action in ACTIONS}

    for row in plan:

        assert row["decision_source"] == (
            "FROZEN_PHASE6_AI_DECISION"
        )

        assert row["inference_reproduced"] is True
        assert row["forced_rl_action"] is False

        assert row["ids_epsilon"] == 0.02
        assert row["rl_epsilon"] == 0.10

        observation = row["observation"]

        assert len(observation) == 8

        assert all(
            math.isfinite(x) and 0 <= x <= 1
            for x in observation
        )

        assert observation[7] == 0.5

        risk = row["ids_risk"]
        trust = row["observed_trust"]

        assert observation[4] == risk
        assert observation[5] == trust

        requested = row["requested_action_id"]

        assert requested in range(5)

        assert ACTIONS[requested] == row["requested_action"]

        # Independent policy contract verification.

        if trust < 0.20 and risk > 0.70:

            expected = 4
            reason = "MANDATORY_BLOCK"

        elif (
            trust < 0.35
            and risk > 0.55
            and requested == 0
        ):

            expected = 1
            reason = "MANDATORY_INSPECT"

        else:

            expected = requested
            reason = "PASSTHROUGH"

        assert row["final_action_id"] == expected

        assert row["final_action"] == ACTIONS[expected]

        assert row["expected_action"] == ACTIONS[expected]

        assert row["zero_trust_override"] == (
            expected != requested
        )

        assert row["zero_trust_reason"] == reason

    assert sum(
        row["final_action"] == "INSPECT"
        and row["zero_trust_override"]
        for row in plan
    ) == 3

    assert any(
        row["final_action"] == "BLOCK"
        and row["zero_trust_override"]
        for row in plan
    )

    print("[PASS] All 15 frozen scenarios validated.")

    return plan


def save_summary(status, error=None):

    summary = {

        "cell": "P6_C08C3",

        "status": status,

        "audited_base_commit": AUDITED_BASE_COMMIT,

        "test_plan_sha256": EXPECTED_PLAN_SHA256,

        "planned_scenarios": 15,

        "executed_scenarios": len(records),

        "passed_scenarios": sum(
            r["status"] == "PASS" for r in records
        ),

        "failed_scenarios": sum(
            r["status"] == "FAIL" for r in records
        ),

        "action_distribution": dict(Counter(
            r["final_action"] for r in records
        )),

        "zero_trust_override_cases": sum(
            r["zero_trust_override"] for r in records
        ),

        "baseline": baseline_metrics,

        "recovery": recovery_metrics,

        "acceptance_criteria": CRITERIA,

        "new_ai_inference_executed": False,

        "forced_rl_actions": 0,

        "traffic_replay_mode": "SYNTHETIC_LAB_PROBE",

        "live_traffic_feature_extraction": False,

        "scenarios": records,

        "error": error,
    }

    (RESULTS / "integrated_dataplane_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )


def verify_additional_criteria(action, metrics):

    # Retain all assertions from the historical C26B harness.

    if action == "RATE_LIMIT":

        assert metrics["configured_rate_kbps"] == 1000

        assert metrics["baseline_mbps"] >= 8.0

        assert 0.5 <= metrics["limited_mbps"] <= 1.3

        assert metrics["throughput_reduction_pct"] >= 80.0

    elif action == "INSPECT":

        assert metrics["packet_loss_pct"] == 0.0

        assert metrics["mirrored_packets"] >= 5

    elif action == "REDIRECT":

        assert metrics["destination_packet_loss_pct"] == 100.0

        assert metrics["diverted_packets"] >= 5

    elif action == "ALLOW":

        assert metrics["packet_loss_pct"] == 0.0

    elif action == "BLOCK":

        assert metrics["packet_loss_pct"] == 100.0

    else:

        raise ValueError(action)


def main():

    print("=" * 75)
    print("P6_C08C3 | PHASE-6 SDN ENFORCEMENT VERIFICATION")
    print("=" * 75)

    setLogLevel("info")

    net = None

    try:

        verify_frozen_sources()

        plan = load_plan()

        net = Mininet(controller=None, build=False)

        net.addController(
            "c0",
            controller=RemoteController,
            ip="127.0.0.1",
            port=6653,
        )

        switch = net.addSwitch(
            "s1",
            cls=OVSSwitch,
            protocols="OpenFlow13",
            failMode="secure",
        )

        hosts = []

        for i in range(1, 5):

            host = net.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24",
                mac=f"00:00:00:00:00:{i:02x}",
            )

            net.addLink(host, switch, port2=i)

            hosts.append(host)

        net.build()
        net.start()

        h1, h2, h3, h4 = hosts

        for src in hosts:

            for dst in hosts:

                if src != dst:

                    src.setARP(dst.IP(), dst.MAC())

        test.wait_until(
            lambda: "NORMAL" in test.ovs(
                "dump-flows", "s1"
            )
        )

        print("[PASS] SDN controller connected.")

        (EVIDENCE / "switch_ports.txt").write_text(
            test.ovs("show", "s1")
        )

        (EVIDENCE / "meters.txt").write_text(
            test.ovs("dump-meters", "s1")
        )

        # Establish administrative baseline.

        test.RESULTS = EVIDENCE / "baseline"
        test.RESULTS.mkdir(exist_ok=True)

        test.apply_action("ALLOW")

        loss = test.ping_loss(h1)

        assert loss == 0

        baseline = test.throughput_test(
            h1, h2, "BASELINE"
        )

        assert baseline >= 8.0

        baseline_metrics.update({
            "packet_loss_pct": loss,
            "throughput_mbps": baseline,
            "status": "PASS",
        })

        print("[PASS] Network baseline:", baseline_metrics)

        # Execute the frozen AI-generated final decisions.

        for index, scenario in enumerate(plan, 1):

            scenario_id = scenario["scenario_id"]
            action = scenario["final_action"]

            test.RESULTS = EVIDENCE / scenario_id
            test.RESULTS.mkdir(exist_ok=True)

            print("\n" + "-" * 75)

            print(f"[SCENARIO {index}/15] {scenario_id}")

            print("IDS:", scenario["ids_model"])
            print("RL:", scenario["rl_model"])
            print("Risk:", scenario["ids_risk"])
            print("Trust:", scenario["observed_trust"])

            print(
                "Decision:",
                scenario["requested_action"],
                "->",
                action,
            )

            result = dict(scenario)

            started = time.monotonic()

            try:

                metrics = verify_action(
                    action,
                    h1, h2, h3, h4,
                    baseline,
                )

                result["dataplane_metrics"] = metrics

                verify_additional_criteria(
                    action, metrics
                )

                result["status"] = "PASS"

                print(
                    f"[PASS] {scenario_id}: "
                    f"{action} -> {metrics}"
                )

            except Exception as exc:

                result.update({
                    "status": "FAIL",
                    "error": repr(exc),
                })

                print(f"[FAIL] {scenario_id}: {exc!r}")

            result["verification_duration_seconds"] = (
                time.monotonic() - started
            )

            records.append(result)

            (test.RESULTS / "scenario_result.json").write_text(
                json.dumps(result, indent=2)
            )

            save_summary("IN_PROGRESS")

        # Restore administrative connectivity.

        test.RESULTS = EVIDENCE / "recovery"
        test.RESULTS.mkdir(exist_ok=True)

        test.apply_action("ALLOW")

        recovery_loss = test.ping_loss(h1)

        recovery_metrics.update({
            "packet_loss_pct": recovery_loss,
            "status": "PASS" if recovery_loss == 0 else "FAIL",
        })

        (test.RESULTS / "recovery.json").write_text(
            json.dumps(recovery_metrics, indent=2)
        )

        assert recovery_loss == 0

        assert len(records) == 15

        assert all(
            row["status"] == "PASS"
            for row in records
        )

        verify_frozen_sources()

        save_summary("PASS")

        print("\n" + "=" * 75)

        print("P6_C08C3: ALL 15 SCENARIOS VERIFIED — PASS")

        print("=" * 75)

    except Exception as exc:

        save_summary("FAIL", repr(exc))

        raise

    finally:

        if net is not None:

            net.stop()


if __name__ == "__main__":

    main()
