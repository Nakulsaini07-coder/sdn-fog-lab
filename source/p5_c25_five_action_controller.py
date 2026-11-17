
"""
Phase 5 - Cell 25A
Five-action OpenFlow 1.3 SDN controller.
"""

import json
import os

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import (
    CONFIG_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
)
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib import hub


COMMAND_FILE = "/tmp/p5-c25-command.json"
ACK_FILE = "/tmp/p5-c25-ack.json"

COOKIE = 0x25A
METER_ID = 25
RATE_KBPS = 1000

ACTION_MAP = {
    0: "ALLOW",
    1: "INSPECT",
    2: "RATE_LIMIT",
    3: "REDIRECT",
    4: "BLOCK",
}


class FiveActionController(app_manager.OSKenApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.datapath = None
        self.last_request_id = None
        self.worker = hub.spawn(self.command_loop)

        self.logger.info("P5_C25: CONTROLLER_INITIALIZED")

    @set_ev_cls(
        ofp_event.EventOFPSwitchFeatures,
        CONFIG_DISPATCHER,
    )
    def switch_features_handler(self, ev):

        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        # Default forwarding.
        actions = [
            parser.OFPActionOutput(ofp.OFPP_NORMAL)
        ]

        dp.send_msg(
            parser.OFPFlowMod(
                datapath=dp,
                priority=0,
                match=parser.OFPMatch(),
                instructions=[
                    parser.OFPInstructionActions(
                        ofp.OFPIT_APPLY_ACTIONS,
                        actions,
                    )
                ],
            )
        )

        # Remove any previous experiment meter.
        dp.send_msg(
            parser.OFPMeterMod(
                datapath=dp,
                command=ofp.OFPMC_DELETE,
                meter_id=METER_ID,
            )
        )

        # Install a 1 Mbps rate-limiting meter.
        dp.send_msg(
            parser.OFPMeterMod(
                datapath=dp,
                command=ofp.OFPMC_ADD,
                flags=ofp.OFPMF_KBPS | ofp.OFPMF_STATS,
                meter_id=METER_ID,
                bands=[
                    parser.OFPMeterBandDrop(
                        rate=RATE_KBPS,
                        burst_size=0,
                    )
                ],
            )
        )

        self.datapath = dp

        self.logger.info(
            "P5_C25: SWITCH_CONNECTED dpid=%s",
            dp.id,
        )

    def enforce_action(self, action):

        dp = self.datapath

        if dp is None:
            raise RuntimeError("Switch not connected")

        if action not in ACTION_MAP.values():
            raise ValueError(f"Unsupported action: {action}")

        ofp = dp.ofproto
        parser = dp.ofproto_parser

        # Target experimental IPv4 traffic.
        match = parser.OFPMatch(
            in_port=1,
            eth_type=0x0800,
            ipv4_src="10.0.0.1",
            ipv4_dst="10.0.0.2",
        )

        # Remove previous experimental policy.
        dp.send_msg(
            parser.OFPFlowMod(
                datapath=dp,
                cookie=COOKIE,
                cookie_mask=0xFFFFFFFFFFFFFFFF,
                table_id=0,
                command=ofp.OFPFC_DELETE,
                out_port=ofp.OFPP_ANY,
                out_group=ofp.OFPG_ANY,
                match=parser.OFPMatch(),
            )
        )

        instructions = []

        if action == "ALLOW":

            actions = [
                parser.OFPActionOutput(2)
            ]

        elif action == "INSPECT":

            # Original destination + inspection host.
            actions = [
                parser.OFPActionOutput(2),
                parser.OFPActionOutput(3),
            ]

        elif action == "RATE_LIMIT":

            instructions.append(
                parser.OFPInstructionMeter(METER_ID)
            )

            actions = [
                parser.OFPActionOutput(2)
            ]

        elif action == "REDIRECT":

            # Divert traffic to quarantine port.
            actions = [
                parser.OFPActionOutput(4)
            ]

        elif action == "BLOCK":

            actions = []

        if actions:
            instructions.append(
                parser.OFPInstructionActions(
                    ofp.OFPIT_APPLY_ACTIONS,
                    actions,
                )
            )

        dp.send_msg(
            parser.OFPFlowMod(
                datapath=dp,
                cookie=COOKIE,
                table_id=0,
                priority=200,
                command=ofp.OFPFC_ADD,
                match=match,
                instructions=instructions,
            )
        )

        dp.send_msg(parser.OFPBarrierRequest(dp))

        self.logger.info(
            "P5_C25: ACTION_SENT action=%s",
            action,
        )

    def write_ack(self, payload):

        temp_file = ACK_FILE + ".tmp"

        with open(temp_file, "w") as f:
            json.dump(payload, f, indent=2)

        os.replace(temp_file, ACK_FILE)

    def command_loop(self):

        while True:

            try:

                if (
                    self.datapath is not None
                    and os.path.exists(COMMAND_FILE)
                ):

                    with open(COMMAND_FILE) as f:
                        command = json.load(f)

                    request_id = command["request_id"]

                    if request_id != self.last_request_id:

                        self.last_request_id = request_id

                        action = command["action"]

                        try:

                            if isinstance(action, int):
                                action = ACTION_MAP[action]

                            action = str(action).upper()

                            self.enforce_action(action)

                            self.write_ack({
                                "request_id": request_id,
                                "action": action,
                                "status": "SENT",
                            })

                        except Exception as exc:

                            self.write_ack({
                                "request_id": request_id,
                                "status": "ERROR",
                                "error": str(exc),
                            })

            except Exception as exc:

                self.logger.error(
                    "P5_C25: COMMAND_ERROR %s", exc
                )

            hub.sleep(0.25)

    @set_ev_cls(
        ofp_event.EventOFPBarrierReply,
        MAIN_DISPATCHER,
    )
    def barrier_reply_handler(self, ev):

        self.logger.info(
            "P5_C25: BARRIER_REPLY xid=%s",
            ev.msg.xid,
        )

    @set_ev_cls(
        ofp_event.EventOFPErrorMsg,
        [CONFIG_DISPATCHER, MAIN_DISPATCHER],
    )
    def error_handler(self, ev):

        self.logger.error(
            "P5_C25: OPENFLOW_ERROR type=%s code=%s",
            ev.msg.type,
            ev.msg.code,
        )
