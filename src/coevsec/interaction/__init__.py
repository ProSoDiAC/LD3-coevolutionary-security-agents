"""Interaction layer: dynamic graph, message bus and coalition detection.

Represents the agent population as a time-varying graph G_t = (V_t, E_t) and provides the communication substrate over which
cooperation and coalitions may *emerge* - agents are never told they can
cooperate.
"""

from coevsec.interaction.graph import InteractionGraph
from coevsec.interaction.bus import MessageBus, Message
from coevsec.interaction.coalition import detect_coalitions

__all__ = ["InteractionGraph", "MessageBus", "Message", "detect_coalitions"]
