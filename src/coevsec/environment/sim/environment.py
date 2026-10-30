"""Simulator environment: the state machine tying world + tools + detection.

Implements the :class:`~coevsec.environment.base.CyberEnvironment` contract with
partial observability. Attackers chain recon -> probe -> exploit -> harvest ->
lateral -> exfiltrate; defenders monitor/investigate/isolate/rotate/decoy/block.
Detection is a noisy, seedable accumulation process, so stealthy strategies
measurably reduce the detection rate.
"""

from __future__ import annotations

import math
import random
from typing import Any

from coevsec.core.config import EnvironmentConfig
from coevsec.core.types import Action, Event, Observation, Role, StepResult
from coevsec.environment.base import CyberEnvironment
from coevsec.environment.sim.tools import build_tool_registry
from coevsec.environment.sim.world import (
    Credential,
    Host,
    LAYERS,
    Vulnerability,
    WorldState,
)


class SimEnvironment(CyberEnvironment):
    def __init__(self, cfg: EnvironmentConfig, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self.det = cfg.detector
        self.tools = build_tool_registry()
        wait = self.tools.get("wait")
        if wait is not None:
            wait.cost = self.det.wait_cost
        self._rng = rng or random.Random(0)
        self.roles: dict[str, Role] = {}
        self.world = WorldState()
        # Per-attacker knowledge under partial observability.
        self.knowledge: dict[str, dict[str, Any]] = {}
        self.cost: dict[str, float] = {}
        self.detection_cost: float = 0.0
        self.false_positives: int = 0
        self.offensive_attempted: bool = False
        self.first_compromise_turn: int | None = None
        self.first_foothold_turn: int | None = None
        self.service_foothold_host: str | None = None
        self.first_detection_turn: int | None = None
        self.first_detection_reason: str | None = None
        self.recovery_turn: int | None = None
        self.max_kill_chain_phase: int = 0
        self._events: list[Event] = []
        self._phase_rank = {
            "recon": 1, "probe_service": 2, "exploit": 3, "harvest_credentials": 4,
            "lateral_move": 5, "exfiltrate": 6,
        }

    # ------------------------------------------------------------------ setup
    def register_agent(self, agent_id: str, role: Role) -> None:
        self.roles[agent_id] = role
        self.cost.setdefault(agent_id, 0.0)
        if role == Role.ATTACKER:
            self.knowledge.setdefault(
                agent_id, {"hosts": set(), "vulns": {}, "creds": set()}
            )

    def reset(self, seed: int, profile_seed: int | None = None) -> None:
        # ``_rng`` drives per-turn stochastic draws (exploit success, detector)
        # and varies every episode; ``_struct_rng`` draws the cross-episode
        # structure (vulnerability difficulties, and hence the easiest path) and
        # recurs when ``profile_seed`` does.
        self._rng = random.Random(seed)
        # With no profile seed the structural draws share the episode stream, so
        # a single-regime (IID) run is byte-for-byte identical to one without
        # this split. A profile seed uses an independent stream that recurs.
        self._struct_rng = self._rng if profile_seed is None else random.Random(profile_seed)
        self.world = WorldState()
        self.cost = {a: 0.0 for a in self.roles}
        self.detection_cost = 0.0
        self.false_positives = 0
        self.offensive_attempted = False
        self.first_compromise_turn = None
        self.first_foothold_turn = None
        self.service_foothold_host = None
        self.first_detection_turn = None
        self.first_detection_reason = None
        self.recovery_turn = None
        self.max_kill_chain_phase = 0
        self._events = []
        for a, r in self.roles.items():
            if r == Role.ATTACKER:
                self.knowledge[a] = {"hosts": set(), "vulns": {}, "creds": set()}

        # Gateway.
        gw = Host(name="gw", layer="gateway")
        gw.vulnerabilities = self._make_vulns("gw")
        self.world.hosts["gw"] = gw

        # Service hosts, each carrying a credential that unlocks the data store.
        for i in range(self.cfg.hosts):
            name = f"svc_{i:02d}"
            h = Host(name=name, layer="service")
            h.vulnerabilities = self._make_vulns(name)
            h.credentials = [Credential(cid=f"{name}_key", unlocks="datastore")]
            self.world.hosts[name] = h

        # Data store: reachable from the service layer; also has a hard vuln so a
        # direct exploit is possible but less attractive than the credential path.
        ds = Host(name="datastore", layer="datastore")
        ds.vulnerabilities = [Vulnerability(vid="datastore_v0", difficulty=0.85)]
        self.world.hosts["datastore"] = ds

    def _make_vulns(self, host: str) -> list[Vulnerability]:
        out = []
        for j in range(self.cfg.vulnerabilities_per_host):
            out.append(
                Vulnerability(vid=f"{host}_v{j}", difficulty=round(self._struct_rng.uniform(0.2, 0.7), 2))
            )
        return out

    # -------------------------------------------------------------- utilities
    def _attacker_can_reach(self, agent_id: str, host: Host) -> bool:
        if host.isolated:
            return False
        idx = LAYERS.index(host.layer)
        if idx == 0:
            return True
        upper = LAYERS[idx - 1]
        return self.world.attacker_reached_layer(agent_id, upper)

    def _add_noise(self, host: Host, noise: float) -> None:
        host.suspicion += noise * (1.0 + 2.0 * host.monitor_level)

    def _emit(self, ev: Event) -> None:
        self._events.append(ev)

    def _record_kill_chain(self, tool: str) -> None:
        rank = self._phase_rank.get(tool, 0)
        if rank > self.max_kill_chain_phase:
            self.max_kill_chain_phase = rank

    def _mark_foothold(self, host: "Host | None" = None) -> None:
        if self.first_foothold_turn is None:
            self.first_foothold_turn = self.world.turn
        if host is not None and host.layer == "service" and self.service_foothold_host is None:
            self.service_foothold_host = host.name

    def _mark_detection(self, reason: str) -> None:
        if self.first_detection_turn is None:
            self.first_detection_turn = self.world.turn
            self.first_detection_reason = reason

    # ----------------------------------------------------------------- step
    def step(self, action: Action) -> StepResult:
        role = self.roles.get(action.agent_id, Role.ATTACKER)
        tool = self.tools.get(action.tool)
        events: list[Event] = []
        if tool is None or role not in tool.roles:
            obs = self.observe(action.agent_id)
            return StepResult(observation=obs, events=events, success=False, cost=0.1,
                              detail={"error": f"invalid tool for role: {action.tool}"})

        errs = tool.validate_params(action.params)
        if errs:
            obs = self.observe(action.agent_id)
            return StepResult(observation=obs, events=events, success=False, cost=0.1,
                              detail={"error": f"invalid params: {errs}"})

        action_cost = tool.cost
        if action.tool == "wait":
            action_cost = self.det.wait_cost
        self.cost[action.agent_id] = self.cost.get(action.agent_id, 0.0) + action_cost
        if role == Role.DEFENDER and tool.category in {"monitor", "investigate", "isolate", "rotate", "decoy"}:
            self.detection_cost += tool.cost

        handler = getattr(self, f"_do_{action.tool}", None)
        if handler is None:
            success, detail = False, {"error": "unimplemented tool"}
        else:
            success, detail = handler(action)

        self._record_kill_chain(action.tool)
        obs = self.observe(action.agent_id)
        return StepResult(observation=obs, events=list(self._events[-8:]), success=success,
                          cost=action_cost, detail=detail)

    # ----------------------------------------------- attacker action handlers
    def _do_recon(self, action: Action) -> tuple[bool, dict]:
        aid = action.agent_id
        self.offensive_attempted = True
        reachable = [h for h in self.world.hosts.values() if self._attacker_can_reach(aid, h)]
        for h in reachable:
            self.knowledge[aid]["hosts"].add(h.name)
            self._add_noise(h, 0.2)
        self._emit(Event(kind="recon", source=aid, noise=0.2,
                         detail={"revealed": [h.name for h in reachable]}))
        return True, {"revealed": [h.name for h in reachable]}

    def _do_probe_service(self, action: Action) -> tuple[bool, dict]:
        aid = action.agent_id
        self.offensive_attempted = True
        host = self.world.hosts.get(action.params["host"])
        if host is None or host.name not in self.knowledge[aid]["hosts"]:
            return False, {"error": "unknown host"}
        if not self._attacker_can_reach(aid, host):
            return False, {"error": "host not reachable"}
        if host.is_decoy:
            self._trigger_decoy(aid, host)
            return False, {"decoy": True}
        found = []
        for v in host.vulnerabilities:
            if self._rng.random() < 0.8:
                v.discovered_by.add(aid)
                self.knowledge[aid]["vulns"].setdefault(host.name, set()).add(v.vid)
                found.append(v.vid)
        self._add_noise(host, 0.5)
        self._emit(Event(kind="probe", source=aid, target=host.name, noise=0.5,
                         detail={"found": found}))
        return True, {"found": found}

    def _do_exploit(self, action: Action) -> tuple[bool, dict]:
        aid = action.agent_id
        self.offensive_attempted = True
        host = self.world.hosts.get(action.params["host"])
        if host is None or not self._attacker_can_reach(aid, host):
            return False, {"error": "host not reachable"}
        if host.is_decoy:
            self._trigger_decoy(aid, host)
            return False, {"decoy": True}
        known = self.knowledge[aid]["vulns"].get(host.name, set())
        vid = action.params.get("vuln")
        vuln = next((v for v in host.vulnerabilities if v.vid == vid), None)
        if vuln is None:
            vuln = next((v for v in host.vulnerabilities if v.vid in known), None)
        if vuln is None:
            self._add_noise(host, 0.6)
            self._emit(Event(kind="exploit_fail", source=aid, target=host.name, noise=0.6))
            return False, {"error": "no known vulnerability"}
        success = self._rng.random() < (0.9 * (1.0 - vuln.difficulty))
        self._add_noise(host, 1.0)
        if success:
            host.footholds.add(aid)
            self._mark_foothold(host)
            if host.layer == "datastore" and self.first_compromise_turn is None:
                self.first_compromise_turn = self.world.turn
            self._emit(Event(kind="exploit", source=aid, target=host.name, noise=1.0,
                             detail={"vuln": vuln.vid}))
            return True, {"foothold": host.name}
        self._emit(Event(kind="exploit_fail", source=aid, target=host.name, noise=1.0))
        return False, {"error": "exploit failed"}

    def _do_harvest_credentials(self, action: Action) -> tuple[bool, dict]:
        aid = action.agent_id
        self.offensive_attempted = True
        host = self.world.hosts.get(action.params["host"])
        if host is None or aid not in host.footholds:
            return False, {"error": "no foothold on host"}
        harvested = []
        for c in host.credentials:
            c.harvested_by.add(aid)
            if c.valid:
                self.knowledge[aid]["creds"].add(c.cid)
                harvested.append(c.cid)
        self._add_noise(host, 0.5)
        self._emit(Event(kind="harvest", source=aid, target=host.name, noise=0.5,
                         detail={"creds": harvested}))
        return bool(harvested), {"creds": harvested}

    def _do_lateral_move(self, action: Action) -> tuple[bool, dict]:
        aid = action.agent_id
        self.offensive_attempted = True
        host = self.world.hosts.get(action.params["host"])
        if host is None or not self._attacker_can_reach(aid, host):
            return False, {"error": "host not reachable"}
        if host.is_decoy:
            self._trigger_decoy(aid, host)
            return False, {"decoy": True}
        # Need a valid harvested credential unlocking this host.
        has_cred = any(
            c.unlocks == host.name and c.valid and c.cid in self.knowledge[aid]["creds"]
            for h in self.world.hosts.values() for c in h.credentials
        )
        if not has_cred:
            return False, {"error": "no valid credential"}
        host.footholds.add(aid)
        self._mark_foothold(host)
        if host.layer == "datastore" and self.first_compromise_turn is None:
            self.first_compromise_turn = self.world.turn
        self._add_noise(host, 0.4)
        self._emit(Event(kind="lateral", source=aid, target=host.name, noise=0.4))
        return True, {"foothold": host.name}

    def _do_exfiltrate(self, action: Action) -> tuple[bool, dict]:
        aid = action.agent_id
        self.offensive_attempted = True
        ds = self.world.datastore()
        if aid not in ds.footholds or ds.isolated:
            return False, {"error": "no foothold on data store"}
        self._add_noise(ds, 1.0)
        self.world.exfiltrated = True
        self._emit(Event(kind="exfiltrate", source=aid, target="datastore", noise=1.0))
        return True, {"exfiltrated": True}

    def _do_wait(self, action: Action) -> tuple[bool, dict]:
        self._emit(Event(kind="wait", source=action.agent_id, noise=0.0))
        return True, {}

    def _do_communicate(self, action: Action) -> tuple[bool, dict]:
        # The message bus is owned by the interaction layer; here we only log it.
        self._emit(Event(kind="communicate", source=action.agent_id,
                         target=action.params.get("to"), noise=0.05,
                         detail={"content": action.params.get("content", "")[:200]}))
        return True, {}

    def _trigger_decoy(self, aid: str, host: Host) -> None:
        host.suspicion += 5.0
        self.world.detections += 1
        self._mark_detection("decoy")
        self._emit(Event(kind="detection", source="defender", target=host.name, noise=0.0,
                         detail={"reason": "decoy_interaction", "attacker": aid}))

    # ---------------------------------------------- defender action handlers
    def _do_monitor(self, action: Action) -> tuple[bool, dict]:
        host = self.world.hosts.get(action.params["host"])
        if host is None:
            return False, {"error": "unknown host"}
        host.monitor_level = min(1.0, host.monitor_level + 0.5)
        self._emit(Event(kind="monitor", source=action.agent_id, target=host.name, noise=0.0))
        return True, {"monitor_level": host.monitor_level}

    def _do_investigate(self, action: Action) -> tuple[bool, dict]:
        host = self.world.hosts.get(action.params["host"])
        if host is None:
            return False, {"error": "unknown host"}
        threshold = self.det.investigate_threshold - host.monitor_level
        if host.suspicion >= threshold and host.footholds:
            self.world.detections += 1
            self._mark_detection("investigation")
            self._emit(Event(kind="detection", source=action.agent_id, target=host.name,
                             detail={"reason": "investigation"}))
            return True, {"detected": True, "compromised_by": list(host.footholds)}
        self._emit(Event(kind="investigate", source=action.agent_id, target=host.name))
        return True, {"detected": False}

    def _do_isolate(self, action: Action) -> tuple[bool, dict]:
        host = self.world.hosts.get(action.params["host"])
        if host is None:
            return False, {"error": "unknown host"}
        false_positive = not host.footholds
        was_compromised = bool(host.footholds)
        host.isolated = True
        host.footholds.clear()
        if host.layer == "datastore" and was_compromised and self.recovery_turn is None:
            self.recovery_turn = self.world.turn
        detail = {"isolated": host.name, "false_positive": false_positive}
        if false_positive:
            self.false_positives += 1
            self.detection_cost += 3.0  # operational cost of a wrong isolation
        self._emit(Event(kind="isolate", source=action.agent_id, target=host.name,
                         detail=detail))
        return True, detail

    def _do_rotate_credentials(self, action: Action) -> tuple[bool, dict]:
        host = self.world.hosts.get(action.params["host"])
        if host is None:
            return False, {"error": "unknown host"}
        for c in host.credentials:
            c.valid = False
        self._emit(Event(kind="rotate", source=action.agent_id, target=host.name))
        return True, {"rotated": host.name}

    def _do_deploy_decoy(self, action: Action) -> tuple[bool, dict]:
        layer = action.params["layer"]
        name = f"decoy_{layer}_{len([h for h in self.world.hosts if h.startswith('decoy')])}"
        h = Host(name=name, layer=layer, is_decoy=True)
        h.vulnerabilities = [Vulnerability(vid=f"{name}_v0", difficulty=0.1)]
        self.world.hosts[name] = h
        self._emit(Event(kind="decoy", source=action.agent_id, target=name,
                         detail={"layer": layer}))
        return True, {"decoy": name}

    def _do_block(self, action: Action) -> tuple[bool, dict]:
        evicted = []
        for h in self.world.hosts_in_layer("gateway"):
            if h.footholds:
                evicted.extend(h.footholds)
                h.footholds.clear()
            h.isolated = True
        self._emit(Event(kind="block", source=action.agent_id, detail={"evicted": evicted}))
        return True, {"evicted": evicted}

    # ----------------------------------------------------------- turn control
    def end_turn(self) -> list[Event]:
        """Apply decay and noisy auto-detection; advance the turn counter."""
        new_events: list[Event] = []
        lam = self.det.suspicion_lambda
        s_decay = self.det.suspicion_decay
        m_decay = self.det.monitor_decay
        for h in self.world.hosts.values():
            if h.footholds and h.suspicion > 0:
                p = 1.0 - math.exp(-lam * h.suspicion)
                if self._rng.random() < p:
                    self.world.detections += 1
                    self._mark_detection("automatic")
                    ev = Event(kind="detection", source="ids", target=h.name,
                               detail={"reason": "auto", "suspicion": round(h.suspicion, 2)})
                    self._emit(ev)
                    new_events.append(ev)
            if h.footholds and not self.det.decay_while_foothold:
                pass
            else:
                h.suspicion *= s_decay
            h.monitor_level *= m_decay
        # Benign background traffic on one random uncompromised host.
        if self.det.benign_noise > 0:
            clean = [h for h in self.world.hosts.values()
                     if not h.footholds and not h.is_decoy]
            if clean:
                self._rng.choice(clean).suspicion += self._rng.uniform(0, self.det.benign_noise)
        self.world.turn += 1
        return new_events

    # --------------------------------------------------------------- observe
    def observe(self, agent_id: str) -> Observation:
        role = self.roles.get(agent_id, Role.ATTACKER)
        if role == Role.ATTACKER:
            return self._observe_attacker(agent_id)
        return self._observe_defender(agent_id)

    def _observe_attacker(self, aid: str) -> Observation:
        k = self.knowledge.get(aid, {"hosts": set(), "vulns": {}, "creds": set()})
        footholds = [h.name for h in self.world.hosts.values() if aid in h.footholds]
        known = []
        for name in sorted(k["hosts"]):
            h = self.world.hosts.get(name)
            if h is None:
                continue
            known.append({
                "host": name,
                "layer": h.layer,
                "reachable": self._attacker_can_reach(aid, h),
                "isolated": h.isolated,
                "known_vulns": sorted(k["vulns"].get(name, set())),
                "foothold": aid in h.footholds,
            })
        data = {
            "role": "attacker",
            "turn": self.world.turn,
            "footholds": footholds,
            "credentials": sorted(k["creds"]),
            "known_hosts": known,
            "objective": "exfiltrate data from the data store while avoiding detection",
        }
        text = self._render_attacker_text(data)
        # Explicit environment fact that LLMs otherwise miss.
        text += (
            "\nNote: only SERVICE hosts hold credentials; the gateway does not. "
            "After owning gw, recon/probe/exploit a svc_* host, then harvest there."
        )
        return Observation(agent_id=aid, turn=self.world.turn, data=data, text=text)

    def _observe_defender(self, aid: str) -> Observation:
        hosts = []
        for h in self.world.hosts.values():
            entry = {"host": h.name, "layer": h.layer, "isolated": h.isolated,
                     "monitored": round(h.monitor_level, 2)}
            # Suspicion is only visible where monitoring is active (partial obs).
            if h.monitor_level > 0:
                entry["suspicion"] = round(h.suspicion, 2)
            hosts.append(entry)
        data = {
            "role": "defender",
            "turn": self.world.turn,
            "hosts": hosts,
            "detections": self.world.detections,
            "objective": "protect the data store while minimising cost and false positives",
        }
        text = self._render_defender_text(data)
        return Observation(agent_id=aid, turn=self.world.turn, data=data, text=text)

    @staticmethod
    def _render_attacker_text(data: dict) -> str:
        lines = [f"Turn {data['turn']}. Objective: {data['objective']}",
                 f"Footholds: {data['footholds'] or 'none'}",
                 f"Credentials: {data['credentials'] or 'none'}",
                 "Known hosts:"]
        if not data["known_hosts"]:
            lines.append("  (none - run recon)")
        for h in data["known_hosts"]:
            lines.append(
                f"  {h['host']} [{h['layer']}] reachable={h['reachable']} "
                f"isolated={h['isolated']} vulns={h['known_vulns']} foothold={h['foothold']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_defender_text(data: dict) -> str:
        lines = [f"Turn {data['turn']}. Objective: {data['objective']}",
                 f"Confirmed detections so far: {data['detections']}", "Hosts:"]
        for h in data["hosts"]:
            s = f"  {h['host']} [{h['layer']}] isolated={h['isolated']} monitor={h['monitored']}"
            if "suspicion" in h:
                s += f" suspicion={h['suspicion']}"
            lines.append(s)
        return "\n".join(lines)

    # ------------------------------------------------------------- accessors
    def is_compromised(self, asset: str) -> bool:
        h = self.world.hosts.get(asset)
        return bool(h and h.footholds)

    def objective_met(self) -> bool:
        return self.world.exfiltrated

    def state_snapshot(self) -> dict:
        return {
            "turn": self.world.turn,
            "exfiltrated": self.world.exfiltrated,
            "detections": self.world.detections,
            "hosts": {
                name: {
                    "layer": h.layer, "isolated": h.isolated,
                    "suspicion": round(h.suspicion, 3),
                    "footholds": sorted(h.footholds), "is_decoy": h.is_decoy,
                }
                for name, h in self.world.hosts.items()
            },
        }

    def episode_summary(self) -> dict:
        attacker_cost = sum(c for a, c in self.cost.items() if self.roles.get(a) == Role.ATTACKER)
        defender_cost = sum(c for a, c in self.cost.items() if self.roles.get(a) == Role.DEFENDER)
        recovery_time = None
        if self.first_compromise_turn is not None:
            end = self.recovery_turn if self.recovery_turn is not None else self.world.turn
            recovery_time = max(0, end - self.first_compromise_turn)
        exfil = self.world.exfiltrated
        detected = self.world.detections > 0
        if exfil:
            termination = "exfiltrated"
        elif not any(h.footholds for h in self.world.hosts.values()):
            termination = "contained"
        else:
            termination = "budget"
        return {
            "attack_success": exfil,
            "attack_success_undetected": exfil and not detected,
            "attempted_attack": self.offensive_attempted,
            # Engagement: the attacker gained at least one foothold. The
            # automatic detector only fires on compromised hosts, so an
            # attacker that never engages cannot be detected by it.
            "foothold_gained": self.first_foothold_turn is not None,
            "time_to_foothold": self.first_foothold_turn,
            "service_foothold_host": self.service_foothold_host,
            "detected": detected,
            "detections": self.world.detections,
            "attacker_cost": round(attacker_cost, 2),
            "defender_cost": round(defender_cost, 2),
            "detection_cost": round(self.detection_cost, 2),
            "false_positives": self.false_positives,
            "recovery_time": recovery_time,
            "termination_reason": termination,
            "turns_used": self.world.turn,
            "time_to_detect": self.first_detection_turn,
            "first_detection_reason": self.first_detection_reason,
            "isolated_hosts": sorted(h.name for h in self.world.hosts.values() if h.isolated),
            "kill_chain_phase_at_end": self.max_kill_chain_phase,
        }
