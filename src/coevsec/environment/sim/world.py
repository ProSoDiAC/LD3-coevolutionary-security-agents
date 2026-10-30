"""Ground-truth world state for the simulator.

The world is a layered network. The attacker starts outside and must chain
reconnaissance, exploitation, credential harvesting and lateral movement to
reach and exfiltrate the data store. The defender observes noisy telemetry and
can monitor, isolate, rotate credentials, deploy decoys and block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Network layers, ordered from outside to the crown jewel.
LAYERS = ["gateway", "service", "datastore"]


@dataclass
class Vulnerability:
    vid: str
    difficulty: float  # 0..1, probability penalty when exploiting
    discovered_by: set[str] = field(default_factory=set)  # attacker ids that found it


@dataclass
class Credential:
    cid: str
    unlocks: str  # host name this credential grants lateral access to
    harvested_by: set[str] = field(default_factory=set)
    valid: bool = True


@dataclass
class Host:
    name: str
    layer: str
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    credentials: list[Credential] = field(default_factory=list)
    is_decoy: bool = False

    # Dynamic state.
    isolated: bool = False  # defender isolated it; blocks traffic through it
    suspicion: float = 0.0  # accumulated by noisy attacker activity
    monitor_level: float = 0.0  # defender monitoring gain, decays each turn
    # Per-attacker footholds (agent_id -> has access).
    footholds: set[str] = field(default_factory=set)

    def reachable_from(self, layer: str) -> bool:
        """A host in layer L is reachable once the attacker holds the layer above."""
        idx = LAYERS.index(self.layer)
        if idx == 0:
            return True  # gateway reachable from the internet
        return layer == LAYERS[idx - 1]


@dataclass
class WorldState:
    hosts: dict[str, Host] = field(default_factory=dict)
    # Attacker knowledge is tracked in the environment, not here.
    turn: int = 0
    exfiltrated: bool = False
    detections: int = 0  # number of confirmed detections this episode

    def hosts_in_layer(self, layer: str) -> list[Host]:
        return [h for h in self.hosts.values() if h.layer == layer]

    def datastore(self) -> Host:
        stores = self.hosts_in_layer("datastore")
        return stores[0]

    def attacker_reached_layer(self, agent_id: str, layer: str) -> bool:
        """True if the attacker has a foothold on any host in ``layer``."""
        return any(
            agent_id in h.footholds and not h.isolated
            for h in self.hosts_in_layer(layer)
        )
