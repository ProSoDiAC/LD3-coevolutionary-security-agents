"""Capability catalogue for the simulator.

Attackers get reconnaissance / exploitation / lateral-movement capabilities;
defenders get monitoring / isolation / deception capabilities. Strategy is NOT
encoded here - only the raw capabilities and their costs.
"""

from __future__ import annotations

from coevsec.core.tool import ToolRegistry, ToolSpec
from coevsec.core.types import Role

_HOST_PARAM = {
    "type": "object",
    "properties": {"host": {"type": "string", "description": "target host name"}},
    "required": ["host"],
}
_NO_PARAM = {"type": "object", "properties": {}}


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()

    # --- Attacker capabilities -------------------------------------------------
    reg.register(ToolSpec(
        name="recon",
        description="Scan the network reachable from your current footholds to reveal hosts and open services. Low noise.",
        roles=(Role.ATTACKER,),
        parameters=_NO_PARAM,
        cost=1.0,
        category="recon",
    ))
    reg.register(ToolSpec(
        name="probe_service",
        description="Probe a known host to discover its vulnerabilities. Medium noise.",
        roles=(Role.ATTACKER,),
        parameters=_HOST_PARAM,
        cost=1.5,
        category="recon",
    ))
    reg.register(ToolSpec(
        name="exploit",
        description="Attempt to exploit a discovered vulnerability on a reachable host to gain a foothold. High noise.",
        roles=(Role.ATTACKER,),
        parameters={
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "vuln": {"type": "string", "description": "vulnerability id"},
            },
            "required": ["host"],
        },
        cost=3.0,
        category="exploit",
    ))
    reg.register(ToolSpec(
        name="harvest_credentials",
        description=(
            "Harvest credentials from a SERVICE host you already control "
            "(layer=service, e.g. svc_00). The gateway has no credentials — "
            "do not harvest there. May unlock the data store. Medium noise."
        ),
        roles=(Role.ATTACKER,),
        parameters=_HOST_PARAM,
        cost=2.0,
        category="exploit",
    ))
    reg.register(ToolSpec(
        name="lateral_move",
        description="Use harvested credentials to move to a deeper reachable host. Medium noise.",
        roles=(Role.ATTACKER,),
        parameters=_HOST_PARAM,
        cost=2.0,
        category="exploit",
    ))
    reg.register(ToolSpec(
        name="exfiltrate",
        description="Exfiltrate data from the data store (the objective). Requires a foothold on it. High noise.",
        roles=(Role.ATTACKER,),
        parameters=_HOST_PARAM,
        cost=3.0,
        category="exfiltrate",
    ))
    reg.register(ToolSpec(
        name="wait",
        description="Take no offensive action this turn; lets suspicion decay. Zero noise.",
        roles=(Role.ATTACKER, Role.DEFENDER),
        parameters=_NO_PARAM,
        cost=0.1,
        category="wait",
    ))
    reg.register(ToolSpec(
        name="communicate",
        description="Send a message to another agent (content is free text).",
        roles=(Role.ATTACKER, Role.DEFENDER),
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["to", "content"],
        },
        cost=0.5,
        category="communicate",
    ))

    # --- Defender capabilities -------------------------------------------------
    reg.register(ToolSpec(
        name="monitor",
        description="Increase monitoring sensitivity on a host, improving detection of noisy activity there.",
        roles=(Role.DEFENDER,),
        parameters=_HOST_PARAM,
        cost=1.0,
        category="monitor",
    ))
    reg.register(ToolSpec(
        name="investigate",
        description="Investigate a host; if suspicion is high, confirm a compromise (detection).",
        roles=(Role.DEFENDER,),
        parameters=_HOST_PARAM,
        cost=1.5,
        category="investigate",
    ))
    reg.register(ToolSpec(
        name="isolate",
        description="Isolate a host, evicting attacker footholds and blocking traffic through it. High operational cost if the host was clean (false positive).",
        roles=(Role.DEFENDER,),
        parameters=_HOST_PARAM,
        cost=3.0,
        category="isolate",
    ))
    reg.register(ToolSpec(
        name="rotate_credentials",
        description="Rotate credentials on a host, invalidating any the attacker harvested there.",
        roles=(Role.DEFENDER,),
        parameters=_HOST_PARAM,
        cost=2.0,
        category="rotate",
    ))
    reg.register(ToolSpec(
        name="deploy_decoy",
        description="Deploy a honeypot/decoy host in a layer. Attacker interaction with it is immediately detected and wastes their effort.",
        roles=(Role.DEFENDER,),
        parameters={
            "type": "object",
            "properties": {"layer": {"type": "string", "enum": ["gateway", "service", "datastore"]}},
            "required": ["layer"],
        },
        cost=2.5,
        category="decoy",
    ))
    reg.register(ToolSpec(
        name="block",
        description="Block the attacker's origin at the gateway, evicting all gateway footholds.",
        roles=(Role.DEFENDER,),
        parameters=_NO_PARAM,
        cost=2.5,
        category="isolate",
    ))
    return reg
