"""Coalition detection.

A coalition is a set of two or more same-role agents that communicate
reciprocally above a threshold within an episode. Coalitions are detected
post-hoc from communication counts - never declared - so the analysis can ask
*under which conditions* cooperation emerges.
"""

from __future__ import annotations

import networkx as nx


def detect_coalitions(counts: dict[tuple[str, str], int], roles: dict[str, str],
                      min_reciprocal: int = 2) -> list[set[str]]:
    """Return groups of same-role agents linked by reciprocal communication."""
    g = nx.Graph()
    for (u, v), w in counts.items():
        rev = counts.get((v, u), 0)
        if roles.get(u) == roles.get(v) and w >= min_reciprocal and rev >= min_reciprocal:
            g.add_edge(u, v)
    coalitions = [set(c) for c in nx.connected_components(g) if len(c) >= 2]
    return coalitions
