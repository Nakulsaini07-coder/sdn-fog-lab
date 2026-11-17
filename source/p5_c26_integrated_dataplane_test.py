
"""
Phase 5 - Cell 26B
Integrated AI-driven five-action dataplane verification.

Reuses the verified C25B networking test functions.
"""

import json
import time
from collections import Counter
from pathlib import Path

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.log import setLogLevel

import p5_c25_five_action_test as test


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results/p5_c26"

EVIDENCE = RESULTS / "dataplane"
EVIDENCE.mkdir(parents=True, exist_ok=True)

PLAN_FILE = RESULTS / "sdn_execution_plan.json"

SUMMARY_FILE = RESULTS / "integrated_dataplane_summary.json"

records = []


def save_summary(status):

    summary = {
        "cell": "P5_C26B",
        "status": status,
        "planned_scenarios": 18,
        "executed_scenarios": len(records),
        "passed_scenarios": sum(
            r["status"] == "PASS" for r in records
        ),
        "failed_scenarios": sum(
            r["status"] == "FAIL" for r in records
        ),
        "source_distribution": dict(Counter(
            r["decision_source"]
            for r in records
        )),
        "action_distribution": dict(Counter(
            r["final_action"]
            for r in records
        )),
        "live_traffic_feature_extraction": False,
        "scenarios": records,
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2)
    )


def load_plan():

    plan = json.loads(PLAN_FILE.read_text())

    assert len(plan) == 18

    sources = Counter(
        r["decision_source"] for r in plan
    )

    assert sources == {
        "NATURAL_RL": 7,
        "CONTROLLED_ACTION_INJECTION": 5,
        "CONTROLLED_ZERO_TRUST_TEST": 6,
    }

    assert len({
        r["scenario_id"] for r in plan
    }) == len(plan)

    for scenario in plan:

        assert scenario["final_action"] in {
            "ALLOW",
            "INSPECT",
            "RATE_LIMIT",
            "REDIRECT",
            "BLOCK",
        }

        assert (
            scenario["sdn_command"]["action"]
            == scenario["final_action"]
        )

    print("[PASS] 18-scenario AI execution plan validated")

    return plan


def verify_action(action, h1, h2, h3, h4, baseline):

    test.apply_action(action)

    if action == "ALLOW":

        loss = test.ping_loss(h1)

        assert loss == 0

        return {
            "packet_loss_pct": loss
        }

    if action == "INSPECT":

        loss, count = test.capture_test(
            h1, h2, h3, "INSPECT"
        )

        assert loss == 0
        assert count >= 5

        return {
            "packet_loss_pct": loss,
            "mirrored_packets": count,
        }

    if action == "RATE_LIMIT":

        limited = test.throughput_test(
            h1, h2, "RATE_LIMIT"
        )

        meter_stats = test.ovs(
            "meter-stats", "s1"
        )

        (
            test.RESULTS / "meter_stats.txt"
        ).write_text(meter_stats)

        assert 0.5 <= limited <= 1.3
        assert limited < baseline * 0.25

        return {
            "configured_rate_kbps": 1000,
            "baseline_mbps": baseline,
            "limited_mbps": limited,
            "throughput_reduction_pct": (
                100 * (1 - limited / baseline)
            ),
        }

    if action == "REDIRECT":

        loss, count = test.capture_test(
            h1, h2, h4, "REDIRECT"
        )

        assert loss == 100
        assert count >= 5

        return {
            "destination_packet_loss_pct": loss,
            "diverted_packets": count,
        }

    if action == "BLOCK":

        loss = test.ping_loss(h1)

        assert loss == 100

        return {
            "packet_loss_pct": loss
        }

    raise ValueError(
        f"Unsupported action: {action}"
    )


def main():

    print("=" * 75)
    print("P5_C26B | INTEGRATED AI-DRIVEN SDN VERIFICATION")
    print("=" * 75)

    plan = load_plan()

    setLogLevel("info")

    net = Mininet(
        controller=None,
        build=False,
    )

    try:

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

            net.addLink(
                host, switch, port2=i
            )

            hosts.append(host)

        net.build()
        net.start()

        h1, h2, h3, h4 = hosts

        for src in hosts:
            for dst in hosts:
                if src != dst:
                    src.setARP(
                        dst.IP(),
                        dst.MAC(),
                    )

        test.wait_until(
            lambda: "NORMAL" in test.ovs(
                "dump-flows", "s1"
            )
        )

        print("[PASS] SDN controller connected")

        (EVIDENCE / "switch_ports.txt").write_text(
            test.ovs("show", "s1")
        )

        (EVIDENCE / "meters.txt").write_text(
            test.ovs("dump-meters", "s1")
        )

        # Establish baseline network performance.
        test.RESULTS = EVIDENCE / "baseline"
        test.RESULTS.mkdir(exist_ok=True)

        test.apply_action("ALLOW")

        assert test.ping_loss(h1) == 0

        baseline = test.throughput_test(
            h1, h2, "BASELINE"
        )

        assert baseline >= 8.0

        print(
            f"[PASS] Baseline: {baseline:.4f} Mbps"
        )

        # Execute every AI-derived and controlled scenario.
        for index, scenario in enumerate(plan, 1):

            scenario_id = scenario["scenario_id"]
            action = scenario["final_action"]

            test.RESULTS = EVIDENCE / scenario_id
            test.RESULTS.mkdir(exist_ok=True)

            print("\n" + "-" * 75)

            print(
                f"[SCENARIO {index}/{len(plan)}] "
                f"{scenario_id}"
            )

            print(
                "Source:",
                scenario["decision_source"]
            )

            print("Model:", scenario["model"])
            print("IDS risk:", scenario["ids_risk"])

            print(
                "Requested:",
                scenario["requested_action"]
            )

            print("Final:", action)

            print(
                "Zero trust:",
                scenario["zero_trust_reason"]
            )

            started = time.monotonic()

            result = {
                **scenario,
            }

            try:

                metrics = verify_action(
                    action,
                    h1, h2, h3, h4,
                    baseline,
                )

                result.update({
                    "status": "PASS",
                    "dataplane_metrics": metrics,
                    "verification_duration_seconds":
                        time.monotonic() - started,
                })

                print(
                    f"[PASS] {scenario_id}: "
                    f"{action} -> {metrics}"
                )

            except Exception as exc:

                result.update({
                    "status": "FAIL",
                    "error": repr(exc),
                })

                print(
                    f"[FAIL] {scenario_id}: "
                    f"{repr(exc)}"
                )

            records.append(result)

            (
                test.RESULTS / "scenario_result.json"
            ).write_text(
                json.dumps(result, indent=2)
            )

            save_summary("IN_PROGRESS")

        # Restore network connectivity.
        test.RESULTS = EVIDENCE / "recovery"
        test.RESULTS.mkdir(exist_ok=True)

        test.apply_action("ALLOW")

        recovery_loss = test.ping_loss(h1)

        (test.RESULTS / "recovery.json").write_text(
            json.dumps({
                "packet_loss_pct": recovery_loss,
                "status": (
                    "PASS"
                    if recovery_loss == 0
                    else "FAIL"
                ),
            }, indent=2)
        )

        assert recovery_loss == 0

        print("[PASS] Administrative recovery")

        assert len(records) == len(plan)

        assert all(
            r["status"] == "PASS"
            for r in records
        ), "One or more integrated scenarios failed"

        save_summary("PASS")

        print("\n" + "=" * 75)
        print("P5_C26B: ALL 18 SCENARIOS VERIFIED — PASS")
        print("=" * 75)

    except Exception:

        save_summary("FAIL")
        raise

    finally:

        net.stop()


if __name__ == "__main__":

    main()
