"""
Dynamic Position Sizing via Kelly Criterion — size proportional to edge.

Every real prop desk sizes by conviction. A 0.85 conviction signal and a
0.51 signal should NOT get the same allocation. Kelly criterion gives the
mathematically optimal sizing, and half-Kelly is the industry standard
for safety (accounts for estimation error in your edge).

Kelly formula: f* = (p * b - q) / b
  p = win probability (mapped from ensemble conviction)
  b = win/loss ratio (take_profit / abs(stop_loss))
  q = 1 - p

We use half-Kelly (f*/2) to be conservative — standard at every major desk.
"""

from __future__ import annotations

import logging
import math

import config
from regime import RegimeParams

logger = logging.getLogger(__name__)


class PositionSizer:
    """Kelly-based position sizing scaled by conviction and vol regime."""

    def __init__(
        self,
        kelly_fraction: float = config.KELLY_FRACTION,
        max_position_pct: float = config.MAX_POSITION_PCT,
        min_position_pct: float = config.MIN_POSITION_PCT,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

    @staticmethod
    def _conviction_to_win_prob(conviction: float) -> float:
        """Map ensemble conviction (0-1) to estimated win probability.

        conviction 0.50 (buy threshold) → p ≈ 0.60 (barely profitable)
        conviction 0.75 (strong)        → p ≈ 0.68
        conviction 1.00 (maximum)       → p ≈ 0.75

        Linear: p = 0.45 + 0.30 * conviction, clamped to [0.50, 0.80]
        """
        p = 0.45 + 0.30 * conviction
        return max(0.50, min(p, 0.80))

    def compute_size(
        self,
        conviction: float,
        regime: RegimeParams,
        available_cash: float,
        contract_cost: float,
    ) -> int:
        """Compute number of contracts to buy using Kelly criterion.

        Args:
            conviction: Ensemble conviction score (0-1)
            regime: Current vol regime parameters
            available_cash: Cash available for new positions
            contract_cost: Cost of one contract (premium × 100)

        Returns:
            Number of contracts (0 if sizing says skip)
        """
        if contract_cost <= 0 or available_cash <= 0:
            return 0

        # Win probability from conviction
        p = self._conviction_to_win_prob(conviction)
        q = 1.0 - p

        # Win/loss ratio from current stop/take-profit, adjusted by regime
        take_profit = config.PREMIUM_TAKE_PROFIT_PCT
        stop_loss = abs(config.PREMIUM_STOP_LOSS_PCT) * regime.stop_loss_mult
        if stop_loss <= 0:
            stop_loss = 0.50  # fallback
        b = take_profit / stop_loss

        # Kelly fraction: f* = (p*b - q) / b
        kelly_f = (p * b - q) / b
        if kelly_f <= 0:
            logger.info(
                f"Kelly says no edge (p={p:.2f}, b={b:.2f}, f*={kelly_f:.3f})"
            )
            return 0

        # Apply fractional Kelly (half-Kelly standard)
        position_pct = kelly_f * self.kelly_fraction

        # Scale by regime
        position_pct *= regime.position_size_mult

        # Clamp to bounds
        position_pct = max(self.min_position_pct, min(position_pct, self.max_position_pct))

        # Convert to number of contracts
        budget = available_cash * position_pct
        num_contracts = math.floor(budget / contract_cost)

        # Hard cap at 5 contracts per position
        num_contracts = min(num_contracts, 5)

        logger.info(
            f"Kelly sizing: p={p:.2f} b={b:.2f} f*={kelly_f:.3f} "
            f"half-Kelly={kelly_f * self.kelly_fraction:.3f} "
            f"regime-adj={position_pct:.3f} → {num_contracts} contracts "
            f"(${budget:.0f} of ${available_cash:.0f})"
        )
        return num_contracts
