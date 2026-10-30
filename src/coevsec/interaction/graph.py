"""Time-varying interaction graph G_t = (V_t, E_t).

Nodes are agents; edges are communication/trust relationships. Supports static,
graph and dynamic topologies and exposes structural metrics (density,
centrality) used by the analysis engine.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx


class InteractionGraph:
    def __init__(self, topology: str = "none") -> None:
        self.topology = topology
        self.g = nx.DiGraph()
        self.snapshots: list[dict] = []

    def add_agent(self, agent_id: str, role: str) -> None:
        self.g.add_node(agent_id, role=role)

    def initialise(self) -> None:
        """Lay down the starting edge set for static / graph topologies."""
        nodes = list(self.g.nodes())
        self.g.remove_edges_from(list(self.g.edges()))
        if self.topology in ("none",) or len(nodes) < 2:
            return
        if self.topology == "static":
            for u in nodes:
                for v in nodes:
                    if u != v:
                        self.g.add_edge(u, v, weight=1)
            return
        # "graph" and "dynamic": bidirectional ring within each role so
        # communication can bootstrap; dynamic topologies then rewrite edges
        # from observed traffic.
        by_role: dict[str, list[str]] = defaultdict(list)
        for n, data in self.g.nodes(data=True):
            by_role[str(data.get("role", ""))].append(n)
        for members in by_role.values():
            members = sorted(members)
            if len(members) == 1:
                continue
            for i, u in enumerate(members):
                v = members[(i + 1) % len(members)]
                self.g.add_edge(u, v, weight=1)
                self.g.add_edge(v, u, weight=1)

    def neighbors(self, agent_id: str) -> list[str]:
        if agent_id not in self.g:
            return []
        return sorted(self.g.successors(agent_id))

    def can_send(self, sender: str, recipient: str, communication: str) -> bool:
        if communication == "none" or not recipient or sender == recipient:
            return False
        if communication == "direct":
            return recipient in self.g
        if communication == "graph":
            return self.g.has_edge(sender, recipient)
        return False

    def set_edges_from_counts(self, counts: dict[tuple[str, str], int], min_weight: int = 1) -> None:
        """Rebuild edges from communication counts (for dynamic topology)."""
        self.g.remove_edges_from(list(self.g.edges()))
        for (u, v), w in counts.items():
            if w >= min_weight and u in self.g and v in self.g:
                self.g.add_edge(u, v, weight=w)

    def snapshot(self, turn: int) -> dict:
        n = self.g.number_of_nodes()
        e = self.g.number_of_edges()
        density = nx.density(self.g) if n > 1 else 0.0
        try:
            centrality = nx.degree_centrality(self.g)
        except Exception:  # pragma: no cover - degenerate graphs
            centrality = {}
        snap = {
            "turn": turn,
            "nodes": n,
            "edges": e,
            "density": round(density, 4),
            "centrality": {k: round(v, 4) for k, v in centrality.items()},
        }
        self.snapshots.append(snap)
        return snap
