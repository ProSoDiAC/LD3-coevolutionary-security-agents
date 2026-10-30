"""Heuristic defender policy with tunable, adaptable vigilance.

Like the attacker, the defender's strategic content is not a fixed script but a
vigilance level governing how eagerly it monitors deep layers, investigates,
isolates, rotates credentials and deploys decoys - and how that vigilance adapts
to breaches across an episode (episodic) and across a run (persistent).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from coevsec.agents.shared.memory import Memory
from coevsec.agents.shared.policy import Policy
from coevsec.core.tool import ToolSpec
from coevsec.core.types import Action, AdaptationLevel, Observation, Role

# Defenders prioritise protecting the deepest assets first.
_PRIORITY = {"datastore": 0, "service": 1, "gateway": 2}


class DefenderHeuristicPolicy(Policy):
    def __init__(
        self, agent_id, adaptation, rng, *, vigilance: float = 0.4,
        disable_investigate: bool = False, disable_decoy: bool = False,
    ):
        super().__init__(agent_id, Role.DEFENDER, adaptation, rng)
        # Ablation switches for the defender's detection channels.
        self.disable_investigate = disable_investigate
        self.disable_decoy = disable_decoy
        self.base_vigilance = float(vigilance)
        self.vigilance = float(vigilance)
        self._prev_detections = 0
        self._decoy_deployed = False
        self._recent_categories: list[str] = []
        self._peak_suspicion = 0.0

    def _adapts(self) -> bool:
        return self.adaptation in (AdaptationLevel.EPISODIC.value, AdaptationLevel.PERSISTENT.value)

    def reset_episode(self) -> None:
        self.vigilance = self.base_vigilance
        self._prev_detections = 0
        self._decoy_deployed = False
        self._recent_categories = []
        self._peak_suspicion = 0.0

    def end_episode(self, summary: dict[str, Any]) -> None:
        if self.adaptation != AdaptationLevel.PERSISTENT.value:
            return
        if summary.get("attack_success"):
            self.base_vigilance = min(1.0, self.base_vigilance + 0.15)
        else:
            self.base_vigilance = max(0.1, self.base_vigilance - 0.03)

    def choose(self, obs: Observation, tools: list[ToolSpec], memory: Memory) -> Action:
        by_name = {t.name: t for t in tools}
        d = obs.data
        hosts = d.get("hosts", [])
        detections = int(d.get("detections", 0))

        # Pre-detection channel: react to a rising suspicion signal (available
        # from monitoring, ahead of any confirmed detection) rather than only
        # to a detection that has already happened. Since higher vigilance
        # lowers susp_threshold below and can make investigate() itself
        # trigger the first detection (world.detections += 1 when
        # suspicion >= investigate_threshold - monitor_level), this gives L1/L2
        # defenders a mechanism that can plausibly raise the first-detection
        # probability, unlike the purely reactive channel below.
        max_susp = max((h.get("suspicion", 0.0) for h in hosts), default=0.0)
        if self._adapts() and max_susp > self._peak_suspicion + 0.4:
            self.vigilance = min(1.0, self.vigilance + 0.1)
        self._peak_suspicion = max(self._peak_suspicion * 0.9, max_susp)

        new_detection = detections > self._prev_detections
        if new_detection and self._adapts():
            self.vigilance = min(1.0, self.vigilance + 0.2)
        self._prev_detections = detections

        susp_threshold = 1.2 - self.vigilance
        suspicious = sorted(
            [h for h in hosts if h.get("suspicion", 0.0) >= susp_threshold],
            key=lambda h: h.get("suspicion", 0.0), reverse=True,
        )

        # 1. React to a fresh detection: isolate the hottest host, rotate creds.
        if new_detection:
            target = suspicious[0]["host"] if suspicious else "datastore"
            if "isolate" in by_name:
                return self._act("isolate", by_name, params={"host": target},
                                 rationale="respond to detection", strategy="dynamic isolation")

        # 2. Confirmed-hot host under monitoring -> investigate to confirm.
        if suspicious and "investigate" in by_name and not self.disable_investigate:
            return self._act("investigate", by_name, params={"host": suspicious[0]["host"]},
                             rationale="confirm suspected compromise", strategy="investigate")

        # 3. Ensure monitoring coverage of the most valuable unmonitored hosts.
        unmonitored = sorted(
            [h for h in hosts if h.get("monitored", 0.0) <= 0.0 and not h.get("isolated")],
            key=lambda h: _PRIORITY.get(h["layer"], 9),
        )
        if unmonitored and self.rng.random() < 0.5 + 0.5 * self.vigilance:
            return self._act("monitor", by_name, params={"host": unmonitored[0]["host"]},
                             rationale="expand monitoring to deep assets", strategy="adaptive monitoring")

        # 4. Proactive deception when vigilant.
        if (not self._decoy_deployed and "deploy_decoy" in by_name and not self.disable_decoy
                and self.rng.random() < 0.3 * self.vigilance):
            self._decoy_deployed = True
            return self._act("deploy_decoy", by_name, params={"layer": "service"},
                             rationale="lure and expose the attacker", strategy="decoy deployment")

        # 5. Periodic credential rotation on the service layer.
        if "rotate_credentials" in by_name and self.rng.random() < 0.15 + 0.2 * self.vigilance:
            svc = [h for h in hosts if h["layer"] == "service" and not h.get("isolated")]
            if svc:
                return self._act("rotate_credentials", by_name,
                                 params={"host": self.rng.choice(svc)["host"]},
                                 rationale="invalidate any harvested credentials",
                                 strategy="credential rotation")

        # 6. Share detections with peer defenders when a channel exists.
        peers = [p for p in obs.data.get("peers", []) if p.get("role") == "defender"]
        if peers and "communicate" in by_name and detections and self.rng.random() < 0.12:
            dest = self.rng.choice(peers)["id"]
            return self._act(
                "communicate", by_name,
                params={"to": dest, "content": f"detections={detections}"},
                rationale="share detection", strategy="coordinated defense",
            )

        return self._act("wait", by_name, rationale="observe", strategy="observe")

    def _act(self, name, by_name, params=None, rationale="", strategy="") -> Action:
        if name not in by_name:
            name = "wait" if "wait" in by_name else next(iter(by_name))
        cat = by_name[name].category
        self._recent_categories = (self._recent_categories + [cat])[-16:]
        return Action(agent_id=self.agent_id, tool=name, params=params or {},
                      target=(params or {}).get("host"), rationale=rationale, strategy=strategy)

    def checkpoint(self) -> dict[str, float]:
        return {"base_vigilance": self.base_vigilance, "vigilance": self.vigilance}

    def load_checkpoint(self, state: dict[str, float]) -> None:
        self.base_vigilance = float(state.get("base_vigilance", self.base_vigilance))
        self.vigilance = float(state.get("vigilance", self.vigilance))

    def strategy_signature(self) -> dict[str, float]:
        sig = {"vigilance": self.vigilance}
        if self._recent_categories:
            n = len(self._recent_categories)
            for k, v in Counter(self._recent_categories).items():
                sig[f"cat_{k}"] = v / n
        return sig
