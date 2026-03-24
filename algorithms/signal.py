"""
OptionSignal: the output of each algorithm.
Direction (CALL/PUT/NEUTRAL) + conviction + preferred contract parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptionSignal:
    """Signal produced by each algorithm."""
    direction: str = "NEUTRAL"     # "CALL", "PUT", or "NEUTRAL"
    conviction: float = 0.0        # 0-1 strength of signal
    scores: dict = field(default_factory=dict)  # Component breakdown
    preferred_delta: float = 0.40  # Algo's recommended delta target
    preferred_dte: int = 30        # Algo's recommended DTE
