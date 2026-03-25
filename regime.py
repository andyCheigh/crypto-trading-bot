"""
Volatility Regime Detection — the macro overlay every real desk runs.

Uses the CBOE VIX index directly (fetched via ^VIX) to classify the market
into four regimes. VIX is the industry standard — computed model-free from
the entire SPY options chain, more accurate than any ATM IV proxy.

Hysteresis prevents flapping at regime boundaries: requires 2 consecutive
confirmations before switching regimes.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class VolRegime(Enum):
    LOW_VOL = "LOW_VOL"       # VIX < 15: quiet, grind-up market
    NORMAL = "NORMAL"         # VIX 15-25: standard conditions
    HIGH_VOL = "HIGH_VOL"     # VIX 25-35: elevated fear, wider ranges
    CRISIS = "CRISIS"         # VIX 35+: panic, correlation spikes


# Ordered for bump-up logic
_REGIME_ORDER = [VolRegime.LOW_VOL, VolRegime.NORMAL, VolRegime.HIGH_VOL, VolRegime.CRISIS]


@dataclass
class RegimeParams:
    """Regime-adapted trading parameters."""
    regime: VolRegime = VolRegime.NORMAL
    vix_level: float = 0.0               # current VIX for logging/notifications
    position_size_mult: float = 1.0      # multiplier on base position size
    buy_threshold_adj: float = 0.0       # added to BUY_THRESHOLD
    stop_loss_mult: float = 1.0          # widens premium stop loss in high vol
    trailing_stop_mult: float = 1.0      # widens trailing stop in high vol
    algo_weights: dict = field(default_factory=lambda: {
        "vol_arb": 0.35,
        "gamma_exposure": 0.35,
        "options_flow": 0.30,
    })


# Pre-computed regime parameter tables
_REGIME_PARAMS = {
    VolRegime.LOW_VOL: RegimeParams(
        regime=VolRegime.LOW_VOL,
        position_size_mult=1.2,     # lean in during calm
        buy_threshold_adj=-0.05,    # lower bar — signals are cleaner
        stop_loss_mult=1.0,
        trailing_stop_mult=1.0,
        algo_weights={              # flow signals dominate in quiet markets
            "vol_arb": 0.30,
            "gamma_exposure": 0.30,
            "options_flow": 0.40,
        },
    ),
    VolRegime.NORMAL: RegimeParams(
        regime=VolRegime.NORMAL,
        position_size_mult=1.0,
        buy_threshold_adj=0.0,
        stop_loss_mult=1.0,
        trailing_stop_mult=1.0,
        algo_weights={
            "vol_arb": 0.35,
            "gamma_exposure": 0.35,
            "options_flow": 0.30,
        },
    ),
    VolRegime.HIGH_VOL: RegimeParams(
        regime=VolRegime.HIGH_VOL,
        position_size_mult=0.7,     # reduce size — wider swings
        buy_threshold_adj=0.03,     # slightly tighter bar but not enough to kill signal generation
        stop_loss_mult=1.3,         # widen stops to avoid noise
        trailing_stop_mult=1.2,
        algo_weights={              # GEX dominates — dealer hedging drives price
            "vol_arb": 0.35,
            "gamma_exposure": 0.40,
            "options_flow": 0.25,
        },
    ),
    VolRegime.CRISIS: RegimeParams(
        regime=VolRegime.CRISIS,
        position_size_mult=0.4,     # survival mode — capital preservation
        buy_threshold_adj=0.15,     # high conviction required but not impossible
        stop_loss_mult=1.5,         # wide stops — everything is noisy
        trailing_stop_mult=1.4,
        algo_weights={              # GEX is king in crisis — dealer flows = the market
            "vol_arb": 0.30,
            "gamma_exposure": 0.50,
            "options_flow": 0.20,
        },
    ),
}


class VolRegimeDetector:
    """Detects market vol regime from CBOE VIX."""

    # VIX thresholds — calibrated to actual market regimes
    # VIX 22-25 is common and not truly elevated — real fear starts at 25+
    VIX_LOW = 15.0        # below: low vol, grind-up
    VIX_HIGH = 25.0       # above: genuinely elevated fear
    VIX_CRISIS = 35.0     # above: full panic

    # Hysteresis: require N consecutive confirmations before switching
    CONFIRMATIONS_REQUIRED = 2

    def __init__(self):
        self._current_regime = VolRegime.NORMAL
        self._pending_regime: VolRegime | None = None
        self._confirmation_count = 0

    @property
    def current_regime(self) -> VolRegime:
        return self._current_regime

    def detect(self, vix_level: float) -> RegimeParams:
        """Detect regime from current VIX level and return adapted parameters.

        Args:
            vix_level: Current CBOE VIX index value (e.g., 18.5)

        Returns:
            RegimeParams for the confirmed regime
        """
        if vix_level <= 0:
            logger.warning("VIX data unavailable, maintaining current regime")
            params = copy.copy(_REGIME_PARAMS[self._current_regime])
            params.vix_level = 0.0
            return params

        # Classify raw regime from VIX level
        if vix_level < self.VIX_LOW:
            raw_regime = VolRegime.LOW_VOL
        elif vix_level < self.VIX_HIGH:
            raw_regime = VolRegime.NORMAL
        elif vix_level < self.VIX_CRISIS:
            raw_regime = VolRegime.HIGH_VOL
        else:
            raw_regime = VolRegime.CRISIS

        # Hysteresis: only switch after consecutive confirmations
        if raw_regime != self._current_regime:
            if raw_regime == self._pending_regime:
                self._confirmation_count += 1
            else:
                self._pending_regime = raw_regime
                self._confirmation_count = 1

            if self._confirmation_count >= self.CONFIRMATIONS_REQUIRED:
                old = self._current_regime
                self._current_regime = raw_regime
                self._pending_regime = None
                self._confirmation_count = 0
                logger.info(
                    f"REGIME CHANGE: {old.value} → {self._current_regime.value} "
                    f"(VIX: {vix_level:.1f})"
                )
        else:
            # Regime unchanged, reset pending
            self._pending_regime = None
            self._confirmation_count = 0

        params = copy.copy(_REGIME_PARAMS[self._current_regime])
        params.vix_level = vix_level
        logger.info(
            f"Regime: {self._current_regime.value} | VIX: {vix_level:.1f} | "
            f"Size mult: {params.position_size_mult}x | "
            f"Buy adj: {params.buy_threshold_adj:+.2f}"
        )
        return params
