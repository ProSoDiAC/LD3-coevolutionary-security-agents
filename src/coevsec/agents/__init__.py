"""Agent layer: attacker/defender agents, policies and memory.

Adaptation levels are realised by composing a policy with a
memory implementation:

    STATIC     -> fixed policy params + NullMemory (Level 0)
    EPISODIC   -> params adapt within an episode + EpisodicMemory (Level 1)
    PERSISTENT -> params persist across episodes + PersistentMemory (Level 2)
"""

from coevsec.agents.shared.agent import Agent
from coevsec.agents.factory import make_agent

__all__ = ["Agent", "make_agent"]
