"""Pure-Python cyber environment simulator.

A seedable, deterministic in-memory state machine modelling the layered network
of (Internet -> Gateway -> Services -> Data Store). Fast
enough for the high-volume co-evolution runs (E2-E8).
"""

from coevsec.environment.sim.environment import SimEnvironment

__all__ = ["SimEnvironment"]
