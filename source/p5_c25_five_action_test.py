
"""
Phase 5 - Cell 25B
Five-action OpenFlow dataplane verification.
"""

import json
import os
import re
import time
import uuid
import subprocess
from pathlib import Path

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.log import setLogLevel


RESULTS = Path("results/p5_c25").resolve()
RESULTS.mkdir(parents=True, exist_ok=True)

COMMAND = Path("/tmp/p5-c25-command.json")
ACK = Path("/tmp/p5-c25-ack.json")

results = {}


def ovs(*args):
    return subprocess.check_output(
        ["ovs-ofctl", "-O", "OpenFlow13", *args],
        text=True,
        stderr=subprocess.STDOUT,
    )


def wait_until(predicate, timeout=90):

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        if predicate():
            return

        time.sleep(0.5)

    raise RuntimeError("Verification timeout")


def apply_action(action):

    request_id = str(uuid.uuid4())

    command = {
        "request_id": request_id,
        "action": action,
    }

    temp = COMMAND.with_suffix(".tmp")
    temp.write_text(json.dumps(command))
    os.replace(temp, COMMAND)

    def acknowledged():

        if not ACK.exists():
            return False

        ack = json.loads(ACK.read_text())

        if ack.get("request_id") != request_id:
            return False

        if ack.get("status") == "ERROR":
            raise RuntimeError(str(ack))

        return ack.get("status") == "SENT"

    wait_until(acknowledged)

    expected = {
        "ALLOW": "output:2",
        "INSPECT": "output:2,output:3",
        "RATE_LIMIT": "meter:25,output:2",
        "REDIRECT": "output:4",
        "BLOCK": "drop",
    }[action]

    def flow_installed():

        flows = ovs("dump-flows", "s1")

        return any(
            "cookie=0x25a" in line
            and line.strip().endswith("actions=" + expected)
            for line in flows.splitlines()
        )

    wait_until(flow_installed, timeout=20)

    flows = ovs("dump-flows", "s1")

    (RESULTS / f"{action}_flows.txt").write_text(flows)

    print(f"[OK] Installed action: {action}", flush=True)


def ping_loss(host):

    output = host.cmd("ping -c 5 -W 1 10.0.0.2")

    print(output, flush=True)

    match = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)

    if not match:
        raise RuntimeError("Cannot parse ping result")

    return float(match.group(1))


def capture_test(source, destination, observer, action):

    pcap = RESULTS / f"{action}.pcap"

    capture = observer.popen([
        "tcpdump",
        "-U",
        "-n",
        "-i",
        observer.defaultIntf().name,
        "-w",
        str(pcap),
        "icmp and src host 10.0.0.1 "
        "and dst host 10.0.0.2",
    ], stdout=subprocess.DEVNULL,
       stderr=subprocess.PIPE)

    try:

        time.sleep(1.5)

        if capture.poll() is not None:
            raise RuntimeError("Packet capture failed to start")

        loss = ping_loss(source)

        time.sleep(0.5)

    finally:

        if capture.poll() is None:
            capture.send_signal(2)

        capture.communicate(timeout=10)

    packets = subprocess.check_output(
        ["tcpdump", "-nn", "-r", str(pcap)],
        text=True,
        stderr=subprocess.DEVNULL,
    )

    count = len(packets.splitlines())

    print(f"[CAPTURE] {action}: {count} packets", flush=True)

    return loss, count


def throughput_test(h1, h2, label):

    server = h2.popen(
        ["iperf3", "-s", "-1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:

        time.sleep(1)

        raw = h1.cmd(
            "iperf3 -c 10.0.0.2 "
            "-u -b 10M -l 1200 -t 10 -J"
        )

        (RESULTS / f"{label}_iperf.json").write_text(raw)

        data = json.loads(raw)

        if "error" in data:
            raise RuntimeError(data["error"])

        received = data["end"]["sum_received"]
        mbps = received["bits_per_second"] / 1_000_000

        print(
            f"[THROUGHPUT] {label}: {mbps:.4f} Mbps",
            flush=True,
        )

        return mbps

    finally:

        if server.poll() is None:
            server.terminate()

        server.wait(timeout=10)


def record(action, **metrics):

    results[action] = {
        "status": "PASS",
        **metrics,
    }

    save_results()

    print(f"[PASS] {action}: {metrics}", flush=True)


def save_results():

    (RESULTS / "verification_results.json").write_text(
        json.dumps(results, indent=2)
    )


def main():

    setLogLevel("info")

    net = Mininet(controller=None, build=False)

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

            net.addLink(host, switch, port2=i)
            hosts.append(host)

        net.build()
        net.start()

        h1, h2, h3, h4 = hosts

        # Static ARP avoids unrelated broadcast traffic.
        for src in hosts:
            for dst in hosts:
                if src != dst:
                    src.setARP(dst.IP(), dst.MAC())

        wait_until(
            lambda: "NORMAL" in ovs("dump-flows", "s1")
        )

        print("[OK] Controller connected", flush=True)

        ports = ovs("show", "s1")
        print(ports)

        (RESULTS / "switch_ports.txt").write_text(ports)

        meters = ovs("dump-meters", "s1")
        print(meters)

        (RESULTS / "meters.txt").write_text(meters)

        # 1. ALLOW
        apply_action("ALLOW")

        loss = ping_loss(h1)

        assert loss == 0, "ALLOW failed"

        baseline = throughput_test(h1, h2, "ALLOW")

        assert baseline >= 8.0, "Baseline throughput too low"

        record(
            "ALLOW",
            packet_loss_pct=loss,
            throughput_mbps=baseline,
        )

        # 2. INSPECT
        apply_action("INSPECT")

        loss, count = capture_test(
            h1, h2, h3, "INSPECT"
        )

        assert loss == 0, "INSPECT interrupted delivery"
        assert count >= 5, "INSPECT mirroring failed"

        record(
            "INSPECT",
            packet_loss_pct=loss,
            mirrored_packets=count,
        )

        # 3. RATE_LIMIT
        apply_action("RATE_LIMIT")

        limited = throughput_test(h1, h2, "RATE_LIMIT")

        meter_stats = ovs("meter-stats", "s1")

        (RESULTS / "rate_limit_meter_stats.txt").write_text(
            meter_stats
        )

        assert 0.5 <= limited <= 1.3, (
            f"Unexpected limited throughput: {limited}"
        )

        assert limited < baseline * 0.25

        record(
            "RATE_LIMIT",
            configured_rate_kbps=1000,
            baseline_mbps=baseline,
            limited_mbps=limited,
            throughput_reduction_pct=(
                100 * (1 - limited / baseline)
            ),
        )

        # 4. REDIRECT
        apply_action("REDIRECT")

        loss, count = capture_test(
            h1, h2, h4, "REDIRECT"
        )

        assert loss == 100, "Original destination reachable"
        assert count >= 5, "Quarantine diversion failed"

        record(
            "REDIRECT",
            original_destination_loss_pct=loss,
            diverted_packets=count,
        )

        # 5. BLOCK
        apply_action("BLOCK")

        loss = ping_loss(h1)

        assert loss == 100, "BLOCK failed"

        record("BLOCK", packet_loss_pct=loss)

        # 6. RECOVERY
        apply_action("ALLOW")

        loss = ping_loss(h1)

        assert loss == 0, "Recovery failed"

        record("RECOVERY", packet_loss_pct=loss)

        print("=" * 70)
        print("P5_C25B: ALL FIVE ACTIONS VERIFIED — PASS")
        print("=" * 70)

    except Exception as exc:

        results["failure"] = {
            "status": "FAIL",
            "error": str(exc),
        }

        raise

    finally:

        save_results()
        net.stop()


if __name__ == "__main__":
    main()
