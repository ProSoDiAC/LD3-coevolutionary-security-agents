"""Heuristic attacker policy with tunable, adaptable stealth.

The kill chain (recon -> probe -> exploit -> harvest -> lateral -> exfiltrate) is
a *capability* ordering, not a prescribed strategy: the strategic content is the
stealth/pacing behaviour, which adapts to inferred detection through scripted
update rules. This gives a purely heuristic agent measurable, if programmed,
strategy drift without an LLM, so the co-evolution machinery can be validated
offline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from coevsec.agents.shared.memory import Memory
from coevsec.agents.shared.policy import Policy
from coevsec.core.tool import ToolSpec
from coevsec.core.types import Action, AdaptationLevel, Observation, Role, StepResult


class AttackerHeuristicPolicy(Policy):
    def __init__(
        self,
        agent_id,
        adaptation,
        rng,
        *,
        stealth: float = 0.4,
        aggression: float = 0.6,
        disable_stealth_pacing: bool = False,
        abandon_failed_hosts: bool = False,
    ):
        super().__init__(agent_id, Role.ATTACKER, adaptation, rng)
        self.base_stealth = float(stealth)
        self.stealth = float(stealth)
        self.aggression = float(aggression)
        self.disable_stealth_pacing = disable_stealth_pacing
        # Legacy behaviour (ablation only): skip failed hosts even when no
        # alternative exists, which strands the attacker at the single gateway.
        self.abandon_failed_hosts = abandon_failed_hosts
        self._cooldown = 0
        self._prev_footholds: set[str] = set()
        self._failed_hosts: set[str] = set()
        self._recent_categories: list[str] = []

    # ----------------------------------------------------------- adaptation
    def _adapts(self) -> bool:
        return self.adaptation in (AdaptationLevel.EPISODIC.value, AdaptationLevel.PERSISTENT.value)

    def reset_episode(self) -> None:
        self.stealth = self.base_stealth
        self._cooldown = 0
        self._prev_footholds = set()
        self._failed_hosts = set()
        self._recent_categories = []

    def end_episode(self, summary: dict[str, Any]) -> None:
        if self.adaptation != AdaptationLevel.PERSISTENT.value:
            return
        detected = bool(summary.get("detected"))
        success = bool(summary.get("attack_success"))
        if detected and not success:
            self.base_stealth = min(1.0, self.base_stealth + 0.15)
        elif success and not detected:
            self.base_stealth = max(0.0, self.base_stealth - 0.05)
        elif not success and not detected:
            # Timed out without being caught: be more aggressive next episode.
            self.base_stealth = max(0.0, self.base_stealth - 0.10)
        else:
            # Succeeded but still detected: the objective was reached, but the
            # attempt was noisy enough to be caught. A milder correction than
            # the pure-failure case (same direction, half the magnitude).
            self.base_stealth = min(1.0, self.base_stealth + 0.05)

    # -------------------------------------------------------------- choose
    def choose(self, obs: Observation, tools: list[ToolSpec], memory: Memory) -> Action:
        by_name = {t.name: t for t in tools}
        d = obs.data
        footholds = set(d.get("footholds", []))
        creds = set(d.get("credentials", []))
        known = d.get("known_hosts", [])

        # Infer a defensive response (a foothold we held has vanished) and react.
        if self._adapts() and self._prev_footholds - footholds:
            self.stealth = min(1.0, self.stealth + 0.2)
        self._prev_footholds = set(footholds)

        # Stealth pacing: after a high-noise action, idle to let suspicion decay.
        if not self.disable_stealth_pacing and self._cooldown > 0:
            self._cooldown -= 1
            return self._act("wait", by_name, rationale="pace attack (stealth)",
                             strategy=f"stealth={self.stealth:.2f}")

        # Opportunistic information sharing when communication is available.
        # Agents are never told to form coalitions; they only use the communicate
        # capability. Coalition formation is detected post-hoc.
        peers = [p for p in obs.data.get("peers", []) if p.get("role") == "attacker"]
        if peers and "communicate" in by_name and self.rng.random() < 0.12:
            dest = self.rng.choice(peers)["id"]
            host_names = [h.get("host") for h in known if h.get("host")]
            return self._act(
                "communicate", by_name,
                params={"to": dest, "content": f"hosts={host_names}"},
                rationale="share reconnaissance", strategy="information sharing",
            )

        action = self._plan(footholds, creds, known, by_name)
        return action

    def _plan(self, footholds, creds, known, by_name) -> Action:
        khosts = {h["host"]: h for h in known}

        # 1. On the data store -> exfiltrate.
        if "datastore" in footholds and "exfiltrate" in by_name:
            self._arm_cooldown()
            return self._act("exfiltrate", by_name, params={"host": "datastore"},
                             rationale="objective in reach", strategy="exfiltrate")

        # 2. Data store reachable and we hold a credential -> lateral move.
        ds = khosts.get("datastore")
        if ds and ds.get("reachable") and not ds.get("foothold") and creds:
            return self._act("lateral_move", by_name, params={"host": "datastore"},
                             rationale="use harvested credentials", strategy="credentialed lateral")

        # 3. Hold a service foothold but no credentials yet -> harvest.
        svc_footholds = [h for h in known if h.get("foothold") and h["layer"] == "service"]
        if svc_footholds and not creds:
            tgt = svc_footholds[0]["host"]
            return self._act("harvest_credentials", by_name, params={"host": tgt},
                             rationale="obtain credentials to reach data store", strategy="harvest")

        # 4. Reachable host with a known vulnerability and no foothold -> exploit.
        # Adaptive agents prefer hosts that have not failed this episode
        # (diversification) but retry a failed one when no alternative exists.
        targets = [h for h in known
                   if h.get("reachable") and not h.get("foothold") and h.get("known_vulns")]
        if self._adapts():
            fresh = [h for h in targets if h["host"] not in self._failed_hosts]
            targets = fresh if (fresh or self.abandon_failed_hosts) else targets
        if targets:
            h = targets[0]
            self._arm_cooldown()
            return self._act("exploit", by_name,
                             params={"host": h["host"], "vuln": h["known_vulns"][0]},
                             rationale="known vulnerability available", strategy="exploit")

        # 5. Reachable known host without known vulns -> probe.
        for h in known:
            if h.get("reachable") and not h.get("foothold") and not h.get("known_vulns"):
                return self._act("probe_service", by_name, params={"host": h["host"]},
                                 rationale="discover vulnerabilities", strategy="recon")

        # 6. Nothing actionable known -> reconnoitre (also reveals newly reachable layers).
        return self._act("recon", by_name, rationale="expand knowledge", strategy="recon")

    def _arm_cooldown(self) -> None:
        if self.disable_stealth_pacing:
            return
        # Higher stealth => more idle turns between noisy actions.
        self._cooldown = int(round(self.stealth * 2))

    def _act(self, name, by_name, params=None, rationale="", strategy="") -> Action:
        if name not in by_name:
            name = "wait" if "wait" in by_name else next(iter(by_name))
        cat = by_name[name].category
        self._recent_categories = (self._recent_categories + [cat])[-16:]
        return Action(agent_id=self.agent_id, tool=name, params=params or {},
                      target=(params or {}).get("host"), rationale=rationale, strategy=strategy)

    def observe_result(self, action: Action, result: StepResult) -> None:
        if not self._adapts():
            return
        if result.detail.get("decoy"):
            self.stealth = min(1.0, self.stealth + 0.25)
        if action.tool == "exploit" and not result.success and action.target:
            self._failed_hosts.add(action.target)

    def checkpoint(self) -> dict[str, float]:
        return {
            "base_stealth": self.base_stealth,
            "stealth": self.stealth,
            "aggression": self.aggression,
        }

    def load_checkpoint(self, state: dict[str, float]) -> None:
        self.base_stealth = float(state.get("base_stealth", self.base_stealth))
        self.stealth = float(state.get("stealth", self.stealth))
        self.aggression = float(state.get("aggression", self.aggression))

    def strategy_signature(self) -> dict[str, float]:
        sig = {"stealth": self.stealth}
        if self._recent_categories:
            n = len(self._recent_categories)
            for k, v in Counter(self._recent_categories).items():
                sig[f"cat_{k}"] = v / n
        return sig
