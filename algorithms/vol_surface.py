"""
Algorithm 3: Implied Volatility Surface & Greeks-Based Signal

The signature edge of options-heavy prop desks. Uses the information embedded
in the options market to predict underlying direction.

Key insight: options market makers (informed flow) move IV before the stock moves.

Signals:
- IV percentile rank (is vol cheap or expensive historically?)
- Put/Call skew (fear gauge, informed directional bets)
- Put/Call volume ratio (smart money positioning)
- Gamma exposure (dealer hedging creates price magnets/repellers)
- Theta/Vega ratio (carry vs vol exposure optimization)
- Max pain convergence (price tends toward max pain near expiry)

Buy: Low IV rank + positive skew shift + accumulation near max pain
Sell: IV spike + negative gamma exposure + max pain divergence
"""

import numpy as np
import pandas as pd

from data_fetcher import StockData


class VolSurfaceAlgorithm:
    name = "vol_surface"

    def _iv_percentile(self, data: StockData) -> float:
        """IV percentile: where current IV sits vs realized vol history.
        Low IV percentile = vol is cheap = good time to be long.
        """
        if data.options.iv_atm == 0 or data.volatility_20d == 0:
            return 0.5
        # Compare implied to realized: ratio < 1 means IV is cheap
        iv_rv_ratio = data.options.iv_atm / data.volatility_20d
        # Convert to a 0-1 percentile (assuming typical range 0.5 to 2.0)
        percentile = (iv_rv_ratio - 0.5) / 1.5
        return float(np.clip(percentile, 0.0, 1.0))

    def _skew_signal(self, data: StockData) -> float:
        """Positive skew (puts > calls IV) indicates fear.
        Contrarian signal: extreme fear = buying opportunity.
        """
        skew = data.options.iv_skew
        if skew > 0.05:   # Puts significantly more expensive
            return min(skew / 0.15, 1.0)
        elif skew < -0.05:  # Calls more expensive (euphoria)
            return -min(abs(skew) / 0.15, 1.0)
        return 0.0

    def _gamma_signal(self, data: StockData) -> float:
        """Dealer gamma exposure signal.
        Positive gamma: dealers hedge by selling rallies / buying dips (stabilizing).
        Negative gamma: dealers amplify moves (destabilizing).
        For buying: positive gamma near current price = support.
        """
        gex = data.options.net_gamma_exposure
        if gex == 0:
            return 0.0
        # Normalize by market cap for cross-stock comparison
        if data.fundamentals.market_cap > 0:
            norm_gex = gex / (data.fundamentals.market_cap / 1e6)
            return float(np.clip(norm_gex, -1.0, 1.0))
        return 0.0

    def _max_pain_signal(self, data: StockData) -> float:
        """Price tends to gravitate toward max pain.
        If current price is below max pain, expect upward pull.
        """
        if data.options.max_pain == 0 or data.price == 0:
            return 0.0
        deviation = (data.options.max_pain - data.price) / data.price
        # Positive = price below max pain (bullish pull)
        return float(np.clip(deviation / 0.05, -1.0, 1.0))

    def _theta_vega_efficiency(self, data: StockData) -> float:
        """Theta/Vega ratio: how much time decay you collect per unit of vol risk.
        Higher absolute ratio when theta is negative = more efficient carry.
        """
        if data.options.avg_vega == 0:
            return 0.0
        ratio = abs(data.options.avg_theta) / (data.options.avg_vega + 1e-10)
        return float(np.clip(ratio, 0.0, 1.0))

    def buy_score(self, data: StockData) -> float:
        """Options-informed buy signal."""
        score = 0.0

        # IV percentile rank (25% weight) - want LOW IV (cheap vol = good entry)
        iv_pct = self._iv_percentile(data)
        if iv_pct < 0.4:
            iv_score = (0.4 - iv_pct) / 0.4
            score += 0.25 * iv_score

        # Put/Call skew - contrarian fear signal (20% weight)
        skew_sig = self._skew_signal(data)
        if skew_sig > 0:
            score += 0.20 * skew_sig

        # Put/Call volume ratio - elevated = contrarian buy (15% weight)
        pcr = data.options.put_call_ratio
        if pcr > 1.0:
            pcr_score = min((pcr - 1.0) / 0.8, 1.0)
            score += 0.15 * pcr_score

        # Gamma exposure - positive gamma = supportive (15% weight)
        gamma_sig = self._gamma_signal(data)
        if gamma_sig > 0:
            score += 0.15 * gamma_sig

        # Max pain convergence - price below max pain (15% weight)
        mp_sig = self._max_pain_signal(data)
        if mp_sig > 0:
            score += 0.15 * mp_sig

        # Delta signal - near-ATM call delta skew (10% weight)
        if data.options.avg_delta_calls > 0.05:
            delta_score = min(data.options.avg_delta_calls / 0.3, 1.0)
            score += 0.10 * delta_score

        return round(min(score, 1.0), 4)

    def sell_score(self, data: StockData, entry_price: float) -> float:
        """Options-informed sell signal."""
        score = 0.0

        # IV spike (vol getting expensive) - 30%
        iv_pct = self._iv_percentile(data)
        if iv_pct > 0.7:
            score += 0.30 * min((iv_pct - 0.5) / 0.5, 1.0)

        # Skew reversal: calls getting expensive = euphoria top (25%)
        skew_sig = self._skew_signal(data)
        if skew_sig < 0:
            score += 0.25 * abs(skew_sig)

        # Negative gamma = destabilizing, amplifies sell-offs (20%)
        gamma_sig = self._gamma_signal(data)
        if gamma_sig < 0:
            score += 0.20 * abs(gamma_sig)

        # Max pain above price diverging further (15%)
        mp_sig = self._max_pain_signal(data)
        if mp_sig < 0:
            score += 0.15 * abs(mp_sig)

        # Put/Call ratio dropping below 0.7 = complacency (10%)
        pcr = data.options.put_call_ratio
        if pcr < 0.7:
            score += 0.10 * min((0.7 - pcr) / 0.4, 1.0)

        return round(min(score, 1.0), 4)
