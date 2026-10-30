"""Co-Evolutionary Security of Autonomous AI Agents.

A framework for studying cybersecurity as a co-evolutionary process between
autonomous attacker and defender agents that repeatedly interact, adapt, and
develop emergent offensive and defensive strategies.

Package layout:
    core         shared types, enums, tool specs, taxonomy, config
    llm          pluggable LLM backends (Ollama/L40 first)
    environment  CyberEnvironment interface + Sim and K8s backends
    agents       attacker/defender agents, policies, memory (adaptation levels)
    interaction  interaction graph, message bus, coalitions
    metrics      ASR, DR, costs, CEP, ESR, diversity, novelty, ...
    telemetry    trajectory JSONL writer + Postgres ingestion
    experiments  experiment controller + co-evolutionary episode loop
    analysis     metric tables, strategy evolution, emergence tagging
"""

__version__ = "0.1.0"
