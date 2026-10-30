"""Message bus for agent-to-agent communication with a per-message cost.

Communication is optional and carries a configurable cost,
so cooperation only pays off under the right conditions. Messages sent on one
turn are delivered in the recipient's next observation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Message:
    sender: str
    recipient: str
    content: str
    turn: int


class MessageBus:
    def __init__(self, communication: str = "none", cost: float = 0.5) -> None:
        self.communication = communication
        self.cost = cost
        self._pending: dict[str, list[Message]] = defaultdict(list)
        self.history: list[Message] = []
        # Directed communication counts, used for coalition detection.
        self.counts: dict[tuple[str, str], int] = defaultdict(int)

    def enabled(self) -> bool:
        return self.communication != "none"

    def send(self, sender: str, recipient: str, content: str, turn: int) -> bool:
        if not self.enabled() or not recipient:
            return False
        msg = Message(sender=sender, recipient=recipient, content=content, turn=turn)
        self._pending[recipient].append(msg)
        self.history.append(msg)
        self.counts[(sender, recipient)] += 1
        return True

    def deliver(self, recipient: str) -> list[Message]:
        msgs = self._pending.pop(recipient, [])
        return msgs

    def inbox_text(self, recipient: str) -> str:
        msgs = self.deliver(recipient)
        if not msgs:
            return ""
        lines = ["Messages received:"]
        lines += [f"  from {m.sender}: {m.content}" for m in msgs]
        return "\n".join(lines)

    def reset_episode(self) -> None:
        self._pending.clear()
        self.history.clear()
        self.counts.clear()
