"""Policy interface, LLM policy with tight schemas, and hybrid LLM+heuristic.

A policy maps an observation to an action. Heuristic and LLM policies share the
same interface. The hybrid policy lets the LLM choose freely but falls back to
the heuristic when the LLM is stuck (repeated recon, invalid params, etc.), so
B3–B5 baselines remain scientifically honest about LLM contribution while still
reaching the objective often enough to measure adaptation.
"""

from __future__ import annotations

import abc
import json
import random
from collections import Counter
from typing import Any

from coevsec.agents.shared.memory import Memory
from coevsec.core.tool import ToolSpec
from coevsec.core.types import Action, Observation, Role, StepResult


class Policy(abc.ABC):
    def __init__(self, agent_id: str, role: Role, adaptation: str, rng: random.Random) -> None:
        self.agent_id = agent_id
        self.role = role
        self.adaptation = adaptation
        self.rng = rng

    @abc.abstractmethod
    def choose(self, obs: Observation, tools: list[ToolSpec], memory: Memory) -> Action: ...

    def observe_result(self, action: Action, result: StepResult) -> None:
        return

    def reset_episode(self) -> None:
        return

    def end_episode(self, summary: dict[str, Any]) -> None:
        return

    def strategy_signature(self) -> dict[str, float]:
        return {}


TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "params": {"type": "object"},
        "rationale": {"type": "string"},
        "strategy": {"type": "string"},
    },
    "required": ["tool", "params"],
}


def _legal_attacker_hints(obs: Observation) -> list[str]:
    """Capability-constrained next steps (not a hard-coded strategy script)."""
    d = obs.data
    footholds = set(d.get("footholds", []))
    creds = set(d.get("credentials", []))
    known = d.get("known_hosts", [])
    hints: list[str] = []
    if not known:
        hints.append('{"tool":"recon","params":{}}')
        return hints
    if "datastore" in footholds:
        hints.append('{"tool":"exfiltrate","params":{"host":"datastore"}}')
        return hints
    ds = next((h for h in known if h.get("host") == "datastore"), None)
    if ds and ds.get("reachable") and not ds.get("foothold") and creds:
        hints.append('{"tool":"lateral_move","params":{"host":"datastore"}}')
        return hints
    # Credentials live on service hosts only — never harvest the gateway.
    for h in known:
        if h.get("foothold") and h.get("layer") == "service" and not creds:
            hints.append(
                f'{{"tool":"harvest_credentials","params":{{"host":"{h["host"]}"}}}}'
            )
            return hints
    for h in known:
        if h.get("reachable") and not h.get("foothold") and h.get("known_vulns"):
            hints.append(
                f'{{"tool":"exploit","params":{{"host":"{h["host"]}",'
                f'"vuln":"{h["known_vulns"][0]}"}}}}'
            )
            break
    for h in known:
        if h.get("reachable") and not h.get("foothold") and not h.get("known_vulns"):
            hints.append(f'{{"tool":"probe_service","params":{{"host":"{h["host"]}"}}}}')
            break
    if footholds and not any(h.get("foothold") and h.get("layer") == "service" for h in known):
        hints.insert(0, '{"tool":"recon","params":{}}')
    elif creds and ds is None:
        # Credentials in hand but the data store not yet discovered: recon from
        # the service foothold is the progressing step.
        hints.insert(0, '{"tool":"recon","params":{}}')
    if not hints:
        hints.append('{"tool":"recon","params":{}}')
    return hints[:5]


def _legal_defender_hints(obs: Observation) -> list[str]:
    hosts = obs.data.get("hosts", [])
    hints: list[str] = []
    hot = sorted(
        [h for h in hosts if h.get("suspicion", 0) > 0],
        key=lambda h: h.get("suspicion", 0),
        reverse=True,
    )
    if hot:
        hints.append(f'{{"tool":"investigate","params":{{"host":"{hot[0]["host"]}"}}}}')
        hints.append(f'{{"tool":"isolate","params":{{"host":"{hot[0]["host"]}"}}}}')
    unmon = [h for h in hosts if h.get("monitored", 0) <= 0 and not h.get("isolated")]
    unmon = sorted(unmon, key=lambda h: {"datastore": 0, "service": 1, "gateway": 2}.get(h["layer"], 9))
    if unmon:
        hints.append(f'{{"tool":"monitor","params":{{"host":"{unmon[0]["host"]}"}}}}')
    hints.append('{"tool":"deploy_decoy","params":{"layer":"service"}}')
    hints.append('{"tool":"wait","params":{}}')
    return hints[:5]


def _autofill_params(tool: str, params: dict[str, Any], obs: Observation) -> dict[str, Any]:
    """Repair common LLM mistakes (missing/wrong host/vuln) from the observation."""
    out = dict(params or {})
    known = {h["host"]: h for h in obs.data.get("known_hosts", [])}
    hosts_def = {h["host"]: h for h in obs.data.get("hosts", [])}
    footholds = set(obs.data.get("footholds", []))

    def _service_foothold() -> str | None:
        for h in known.values():
            if h.get("foothold") and h.get("layer") == "service":
                return h["host"]
        for name in footholds:
            meta = known.get(name)
            if meta and meta.get("layer") == "service":
                return name
        return None

    def _pick_host() -> str | None:
        if tool == "harvest_credentials":
            return _service_foothold()
        if tool in {"exfiltrate", "lateral_move"}:
            if "datastore" in known or "datastore" in hosts_def:
                return "datastore"
            return out.get("host")
        if tool in {"probe_service", "exploit"}:
            # Keep the model's own choice when it is a reachable, unowned host;
            # only repair a missing or unusable one.
            chosen = known.get(out.get("host"))
            if chosen and chosen.get("reachable") and not chosen.get("foothold"):
                return chosen["host"]
            for h in known.values():
                if h.get("reachable") and not h.get("foothold"):
                    return h["host"]
            if out.get("host") in known:
                return out["host"]
            return None
        if out.get("host") in known or out.get("host") in hosts_def:
            return out["host"]
        for h in known.values():
            if tool in {"monitor", "investigate", "isolate", "rotate_credentials"}:
                return h["host"]
        if hosts_def:
            return next(iter(hosts_def))
        if known:
            return next(iter(known))
        return None

    if tool in {
        "probe_service", "exploit", "harvest_credentials", "lateral_move", "exfiltrate",
        "monitor", "investigate", "isolate", "rotate_credentials",
    }:
        host = _pick_host()
        if host:
            out["host"] = host
        elif tool == "harvest_credentials":
            out.pop("host", None)
    if tool == "exploit":
        host = out.get("host")
        if host and host in known and not out.get("vuln"):
            vulns = known[host].get("known_vulns") or []
            if vulns:
                out["vuln"] = vulns[0]
    if tool == "deploy_decoy" and "layer" not in out:
        out["layer"] = "service"
    if tool == "communicate":
        peers = obs.data.get("peers") or []
        if peers and "to" not in out:
            out["to"] = peers[0].get("id", "")
        out.setdefault("content", "status update")
    if tool in {"recon", "wait", "block"}:
        return {}
    return out


class LLMPolicy(Policy):
    """LLM tool selection with capability hints, schema repair and stuck detection."""

    def __init__(
        self, agent_id, role, adaptation, rng, backend, goal: str, *,
        use_cot: bool = False, sample_rng: random.Random | None = None,
    ) -> None:
        super().__init__(agent_id, role, adaptation, rng)
        self.backend = backend
        # Dedicated stream for per-call sampling seeds, so seeding the LLM does
        # not perturb the heuristic draws that share ``rng`` in hybrid mode.
        self.sample_rng = sample_rng
        self.goal = goal
        self.use_cot = use_cot
        self._recent_categories: list[str] = []
        self._recent_tools: list[str] = []
        self.last_tokens = 0
        self.token_remaining = 20000
        self.last_was_repaired = False
        self.repair_count = 0
        self.stuck = False
        self.last_reasoning = ""
        self.last_origin = "model"

    def reset_episode(self) -> None:
        self._recent_categories = []
        self._recent_tools = []
        self.stuck = False
        self.last_was_repaired = False
        self.repair_count = 0
        self.last_tokens = 0
        self.last_reasoning = ""
        self.last_origin = "model"

    def _schema_for(self, tools: list[ToolSpec]) -> dict[str, Any]:
        schema = dict(TOOL_CALL_SCHEMA)
        required = list(schema["required"])
        if self.use_cot:
            # "reasoning" is listed first so constrained decoding generates it
            # before the tool/params fields, giving the model a short scratchpad
            # without leaving JSON-schema-constrained decoding. It is parsed out
            # and never reaches the validator or the executed Action.
            props = {"reasoning": {"type": "string"}, **schema["properties"]}
            required = ["reasoning", *required]
        else:
            props = dict(schema["properties"])
        props["tool"] = {"type": "string", "enum": [t.name for t in tools]}
        schema = {**schema, "properties": props, "required": required}
        return schema

    def _system_prompt(self, tools: list[ToolSpec]) -> str:
        cap_lines = []
        for t in tools:
            req = (t.parameters or {}).get("required") or []
            props = (t.parameters or {}).get("properties") or {}
            param_bits = []
            for name, spec in props.items():
                mark = "required" if name in req else "optional"
                param_bits.append(f"{name} ({mark})")
            param_txt = (", ".join(param_bits)) if param_bits else "none"
            cap_lines.append(f"- {t.name} [params: {param_txt}]: {t.description}")

        if self.role == Role.ATTACKER:
            chain = (
                "Capability constraints (you must satisfy prerequisites):\n"
                "1) recon to discover hosts\n"
                "2) probe_service(host) on a known reachable host\n"
                "3) exploit(host,vuln) to gain a foothold\n"
                "4) After owning the gateway, recon again, then probe/exploit a SERVICE host "
                "(svc_00, svc_01, …). Do NOT harvest_credentials on the gateway — it has none.\n"
                "5) harvest_credentials ONLY on a service-layer foothold\n"
                "6) lateral_move(host=datastore) with a valid credential\n"
                "7) exfiltrate(host=datastore) to win\n"
                "Always set params.host for probe/exploit/harvest/lateral/exfiltrate. "
                "Do not loop recon once hosts are known — progress the chain.\n"
            )
        else:
            chain = (
                "Capability constraints:\n"
                "Monitor deep assets (datastore/service) early. "
                "Investigate when suspicion is visible. "
                "Isolate confirmed hot hosts. "
                "Always pass params.host for monitor/investigate/isolate/rotate.\n"
            )

        if self.use_cot:
            response_instr = (
                "Respond with ONLY one JSON object, no markdown. Fill \"reasoning\" "
                "first with 1-2 short sentences on what you observe and why it "
                "implies the chosen tool, then commit to the remaining fields:\n"
                '{"reasoning":"<short>","tool":"<name>","params":{...},'
                '"rationale":"<short>","strategy":"<short>"}'
            )
        else:
            response_instr = (
                "Respond with ONLY one JSON object, no markdown:\n"
                '{"tool":"<name>","params":{...},"rationale":"<short>","strategy":"<short>"}'
            )
        return (
            f"You are an autonomous {self.role.value} agent in a sandboxed cyber "
            f"environment.\nObjective: {self.goal}\n"
            "Partial observability: use only what you see.\n"
            f"{chain}\n"
            "Available capabilities:\n" + "\n".join(cap_lines) + "\n\n"
            + response_instr
        )

    def choose(self, obs: Observation, tools: list[ToolSpec], memory: Memory) -> Action:
        by_name = {t.name: t for t in tools}
        hints = (
            _legal_attacker_hints(obs)
            if self.role == Role.ATTACKER
            else _legal_defender_hints(obs)
        )
        system = self._system_prompt(tools)
        mem_ctx = memory.context()
        user = obs.text
        if mem_ctx:
            user += "\n\n" + mem_ctx
        user += "\n\nLegal/progressing example actions (prefer one of these if applicable):\n"
        user += "\n".join(f"  {h}" for h in hints)

        # Curriculum: first action of an episode with no knowledge must recon.
        if self.role == Role.ATTACKER and not obs.data.get("known_hosts") and "recon" in by_name:
            if not self._recent_tools:
                action = Action(
                    agent_id=self.agent_id, tool="recon", params={},
                    rationale="curriculum: initial recon", strategy="recon",
                )
                self.last_origin = "curriculum"
                self._record(action, by_name)
                return action

        schema = self._schema_for(tools)
        cap = max(
            64,
            min(
                getattr(self.backend, "max_tokens", 1024),
                self.token_remaining,
            ),
        )
        if self.use_cot:
            # Reasoning tokens are generated in addition to the structured
            # fields, so the per-decision budget needs headroom or the JSON
            # gets truncated mid-object.
            cap = min(cap + 96, self.token_remaining)
        call_seed = self.sample_rng.randrange(2**31) if self.sample_rng is not None else None
        resp = self.backend.complete(
            system, user, json_schema=schema, max_tokens=cap, seed=call_seed,
        )
        self.last_tokens = resp.total_tokens
        tc = resp.tool_call
        self.last_reasoning = tc.reasoning if tc is not None else ""
        self.last_was_repaired = False

        if tc is None or tc.tool not in by_name:
            # Prefer first legal hint tool.
            tool_name = "recon" if self.role == Role.ATTACKER else "wait"
            try:
                hinted = json.loads(hints[0]) if hints else {}
                tool_name = hinted.get("tool", tool_name)
                params = hinted.get("params", {})
            except json.JSONDecodeError:
                params = {}
            if tool_name not in by_name:
                tool_name = tools[0].name
                params = {}
            action = Action(
                agent_id=self.agent_id, tool=tool_name, params=params,
                target=params.get("host"), rationale="fallback/parse", strategy="fallback",
            )
            self.stuck = True
            self.last_origin = "backend"
            self._record(action, by_name)
            return action

        params = _autofill_params(tc.tool, tc.params or {}, obs)
        errs = by_name[tc.tool].validate_params(params)
        if errs:
            self.last_was_repaired = True
            self.repair_count += 1
            # Fall through to first hint if still invalid.
            try:
                hinted = json.loads(hints[0])
                params = _autofill_params(hinted["tool"], hinted.get("params", {}), obs)
                tool_name = hinted["tool"] if hinted["tool"] in by_name else tc.tool
            except (json.JSONDecodeError, KeyError, IndexError):
                tool_name = tc.tool
            errs2 = by_name.get(tool_name, by_name[tc.tool]).validate_params(params)
            if errs2:
                params = {}
                tool_name = "wait" if "wait" in by_name else tc.tool
            action = Action(
                agent_id=self.agent_id, tool=tool_name, params=params,
                target=params.get("host"),
                rationale=f"repaired:{errs[0]}", strategy=tc.strategy or "repair",
            )
            self.stuck = True
            self.last_origin = "repair"
            self._record(action, by_name)
            return action

        action = Action(
            agent_id=self.agent_id, tool=tc.tool, params=params,
            target=params.get("host") or params.get("to"),
            rationale=tc.rationale, strategy=tc.strategy,
        )
        # Stuck if last 3 actions were recon while hosts already known,
        # or if harvest was requested without a service foothold.
        no_service_foothold = not any(
            h.get("foothold") and h.get("layer") == "service"
            for h in obs.data.get("known_hosts", [])
        )
        bad_harvest = action.tool == "harvest_credentials" and (
            not action.params.get("host")
            or (obs.data.get("known_hosts") and no_service_foothold)
            or any(
                h.get("host") == action.params.get("host") and h.get("layer") != "service"
                for h in obs.data.get("known_hosts", [])
            )
        )
        self.stuck = bad_harvest or (
            self.role == Role.ATTACKER
            and len(self._recent_tools) >= 2
            and all(t == "recon" for t in self._recent_tools[-2:])
            and bool(obs.data.get("known_hosts"))
            and action.tool == "recon"
        )
        self.last_origin = "model"
        self._record(action, by_name)
        return action

    def _record(self, action: Action, by_name: dict[str, ToolSpec]) -> None:
        cat = by_name.get(action.tool).category if action.tool in by_name else "misc"
        self._recent_categories = (self._recent_categories + [cat])[-16:]
        self._recent_tools = (self._recent_tools + [action.tool])[-16:]

    def strategy_signature(self) -> dict[str, float]:
        if not self._recent_categories:
            return {}
        n = len(self._recent_categories)
        return {f"cat_{k}": v / n for k, v in Counter(self._recent_categories).items()}


class HybridPolicy(Policy):
    """LLM primary with heuristic takeover when the LLM is stuck or off-hint.

    When ``force_legal_hints`` is true (default for hybrid baselines), any LLM
    action whose tool is not among the capability-constrained legal hints is
    replaced by the heuristic. This keeps LLM free choice *within* progressing
    actions while preventing gateway-harvest / recon-loop failures.
    """

    def __init__(
        self,
        llm: LLMPolicy,
        heuristic: Policy,
        *,
        stall_limit: int = 2,
        force_legal_hints: bool = True,
    ) -> None:
        super().__init__(llm.agent_id, llm.role, llm.adaptation, llm.rng)
        self.llm = llm
        self.heuristic = heuristic
        self.stall_limit = stall_limit
        self.force_legal_hints = force_legal_hints
        self._stall = 0
        self._heuristic_steps = 0
        self._llm_steps = 0
        self._gate_illegal = 0
        self._gate_stall = 0
        self._gate_progress = 0
        self._llm_repairs = 0
        self._origin: Counter = Counter()
        self.last_origin = "model"
        self.last_tokens = 0
        self._last_action_was_llm = False

    @property
    def last_reasoning(self) -> str:
        """CoT scratchpad for the last step, only when the LLM proposal executed."""
        return self.llm.last_reasoning if self._last_action_was_llm else ""

    def reset_episode(self) -> None:
        self.llm.reset_episode()
        self.heuristic.reset_episode()
        self._stall = 0
        self._heuristic_steps = 0
        self._llm_steps = 0
        self._gate_illegal = 0
        self._gate_stall = 0
        self._gate_progress = 0
        self._llm_repairs = 0
        self._origin = Counter()
        self.last_origin = "model"
        self.last_tokens = 0
        self._last_action_was_llm = False

    def gate_stats(self) -> dict[str, float]:
        total = max(1, self._llm_steps + self._heuristic_steps)
        return {
            "llm_steps": float(self._llm_steps),
            "heuristic_steps": float(self._heuristic_steps),
            "hybrid_llm_frac": self._llm_steps / total,
            "hybrid_heur_frac": self._heuristic_steps / total,
            "gate_interventions": float(self._heuristic_steps),
            "gate_illegal": float(self._gate_illegal),
            "gate_stall": float(self._gate_stall),
            "gate_progress": float(self._gate_progress),
            "llm_repairs": float(getattr(self.llm, "repair_count", 0)),
            # Provenance of executed steps. "model" is a proposal the model made
            # and that ran unchanged; "curriculum" the forced initial recon;
            # "repair" a proposal that had to be parameter-repaired; "backend" a
            # step taken because the model returned nothing usable; the
            # "controller_*" categories are heuristic takeovers by reason.
            "origin": {k: float(v) for k, v in self._origin.items()},
        }

    def end_episode(self, summary: dict[str, Any]) -> None:
        self.llm.end_episode(summary)
        self.heuristic.end_episode(summary)

    def observe_result(self, action: Action, result: StepResult) -> None:
        self.llm.observe_result(action, result)
        self.heuristic.observe_result(action, result)

    def choose(self, obs: Observation, tools: list[ToolSpec], memory: Memory) -> Action:
        try:
            action = self.llm.choose(obs, tools, memory)
        except RuntimeError:
            # The backend is unreachable or exhausted its retries: fall back to
            # the heuristic for this step rather than aborting the whole run.
            action = self.heuristic.choose(obs, tools, memory)
            action.rationale = f"hybrid/heuristic:{action.rationale}"
            action.strategy = f"hybrid|{action.strategy}"
            self._heuristic_steps += 1
            self._gate_illegal += 1
            self._stall = 0
            self.last_tokens = 0
            self._last_action_was_llm = False
            self.last_origin = "backend"
            self._origin["backend"] += 1
            return action
        self.last_tokens = getattr(self.llm, "last_tokens", 0)
        use_heur = False
        gate_reason = ""

        legal_tools: set[str] = set()
        if self.force_legal_hints:
            hints = (
                _legal_attacker_hints(obs)
                if self.role == Role.ATTACKER
                else _legal_defender_hints(obs)
            )
            for h in hints:
                try:
                    legal_tools.add(json.loads(h).get("tool", ""))
                except json.JSONDecodeError:
                    continue
            if legal_tools and action.tool not in legal_tools:
                use_heur = True
                gate_reason = "illegal"

        if getattr(self.llm, "stuck", False) or getattr(self.llm, "last_was_repaired", False):
            self._stall += 1
        else:
            self._stall = max(0, self._stall - 1)
        if self._stall >= self.stall_limit:
            use_heur = True
            gate_reason = gate_reason or "stall"

        has_service_foothold = any(
            h.get("foothold") and h.get("layer") == "service"
            for h in obs.data.get("known_hosts", [])
        )
        if (
            self.role == Role.ATTACKER
            and self._llm_steps >= 3
            and not has_service_foothold
            and not obs.data.get("credentials")
            and action.tool in {"recon", "wait", "probe_service", "harvest_credentials"}
        ):
            use_heur = True
            gate_reason = gate_reason or "progress"
        if (
            self.role == Role.ATTACKER
            and self._llm_steps >= 6
            and not obs.data.get("footholds")
        ):
            use_heur = True
            gate_reason = gate_reason or "progress"

        if use_heur:
            action = self.heuristic.choose(obs, tools, memory)
            action.rationale = f"hybrid/heuristic:{action.rationale}"
            action.strategy = f"hybrid|{action.strategy}"
            self._heuristic_steps += 1
            if gate_reason == "illegal":
                self._gate_illegal += 1
            elif gate_reason == "stall":
                self._gate_stall += 1
            elif gate_reason == "progress":
                self._gate_progress += 1
            self._stall = 0
            # Do not zero last_tokens: the LLM backend call already happened and
            # already spent real tokens even though its proposal is discarded here.
            self._last_action_was_llm = False
            self.last_origin = f"controller_{gate_reason or 'stall'}"
            self._origin[self.last_origin] += 1
        else:
            self._llm_steps += 1
            self.last_origin = getattr(self.llm, "last_origin", "model")
            self._origin[self.last_origin] += 1
            self._last_action_was_llm = True
        return action

    def strategy_signature(self) -> dict[str, float]:
        sig = self.llm.strategy_signature()
        total = max(1, self._llm_steps + self._heuristic_steps)
        sig["hybrid_llm_frac"] = self._llm_steps / total
        sig["hybrid_heur_frac"] = self._heuristic_steps / total
        for k, v in self.heuristic.strategy_signature().items():
            sig.setdefault(k, v)
        return sig
