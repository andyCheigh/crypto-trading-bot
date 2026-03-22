"""
Algorithm 1: Volatility Arbitrage (The Jane Street Edge)

Core principle: Implied volatility systematically overprices realized volatility
(the Variance Risk Premium). When this premium is extreme, it creates
directional opportunities in the underlying.

How the top desks trade this:
- IV vs RV spread at multiple windows (5d, 10d, 20d realized vs current IV)
- Volatility cone: where current IV sits vs historical IV distribution
- GARCH-style vol forecasting to detect mean reversion in vol
- IV term structure: backwardation (front > back) signals fear/event premium
- RV acceleration/deceleration for regime detection

Buy underlying when:
  - IV is cheap (low percentile) relative to realized → vol expansion expected
  - RV is decelerating while IV is flat → stock about to move, cheap entry
  - Term structure in contango (front < back) → calm market, supportive of longs

Sell underlying when:
  - IV spikes to extreme percentile → mean reversion in vol = mean reversion in price
  - IV >> RV spread widens → expensive fear premium, take profit before vol crush
  - Term structure in steep backwardation → near-term event risk
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_fetcher import StockData


class VolArbAlgorithm:
    name = "vol_arb"

    def _rv_cone_percentile(self, data: StockData) -> float:
        """Where current RV sits vs its own 60-day rolling distribution.
        Low percentile = quiet period (potential energy building).
        High percentile = extended move (exhaustion risk).
        """
        if data.ohlcv is None or len(data.ohlcv) < 60:
            return 0.5
        daily_ret = data.ohlcv["Close"].pct_change()
        rv_series = daily_ret.rolling(20).std() * np.sqrt(252)
        rv_series = rv_series.dropna()
        if len(rv_series) < 10:
            return 0.5
        current_rv = float(rv_series.iloc[-1])
        percentile = float((rv_series < current_rv).sum() / len(rv_series))
        return percentile

    def _vol_regime(self, data: StockData) -> str:
        """Detect vol regime: compressing, expanding, or stable."""
        ivs = data.iv_surface
        if ivs.rv_5d == 0 or ivs.rv_20d == 0:
            return "stable"
        # RV acceleration: short-term vol vs longer-term
        rv_ratio = ivs.rv_5d / ivs.rv_20d
        if rv_ratio > 1.3:
            return "expanding"
        elif rv_ratio < 0.7:
            return "compressing"
        return "stable"

    def _garch_simple_forecast(self, data: StockData) -> float:
        """Simple EWMA vol forecast (GARCH(1,1) proxy).
        Returns forecasted vol. If forecast < current IV → IV is overpriced.
        """
        if data.ohlcv is None or len(data.ohlcv) < 30:
            return 0.0
        daily_ret = data.ohlcv["Close"].pct_change().dropna()
        # EWMA with lambda=0.94 (RiskMetrics standard)
        lam = 0.94
        var_t = daily_ret.var()
        for r in daily_ret.iloc[-20:]:
            var_t = lam * var_t + (1 - lam) * r ** 2
        return float(np.sqrt(var_t * 252))

    def buy_score(self, data: StockData) -> float:
        """Buy when vol is cheap relative to what we forecast → upside potential."""
        score = 0.0
        ivs = data.iv_surface

        # 1. IV-RV spread analysis (25%)
        # Small premium (IV slightly > RV) is normal and healthy for longs
        # Large premium (IV >> RV) means fear overpriced → mean revert down
        # Negative premium (IV < RV) means cheap vol → buy
        if ivs.iv_atm > 0 and ivs.rv_20d > 0:
            spread_ratio = ivs.iv_rv_spread / ivs.rv_20d
            if spread_ratio < 0.15:  # IV not too expensive
                # Scale: -0.3 = max score, 0.15 = zero
                iv_score = min((0.15 - spread_ratio) / 0.45, 1.0)
                score += 0.25 * max(iv_score, 0)

        # 2. GARCH forecast vs IV: forecast > IV means vol will expand (15%)
        garch_vol = self._garch_simple_forecast(data)
        if garch_vol > 0 and ivs.iv_atm > 0:
            vol_discount = (garch_vol - ivs.iv_atm) / ivs.iv_atm
            if vol_discount > 0:
                score += 0.15 * min(vol_discount / 0.15, 1.0)

        # 3. RV cone: low to mid percentile = room to run (15%)
        rv_pct = self._rv_cone_percentile(data)
        if rv_pct < 0.5:
            score += 0.15 * ((0.5 - rv_pct) / 0.5)

        # 4. Vol regime: compressing = coiled spring, energy building (15%)
        regime = self._vol_regime(data)
        if regime == "compressing":
            score += 0.15

        # 5. Term structure: contango or flat = calm market (15%)
        if ivs.iv_term_slope < 0.01:  # Not in backwardation
            score += 0.15 * min((0.01 - ivs.iv_term_slope) / 0.06, 1.0)

        # 6. 25d skew: elevated put skew = fear = contrarian buy (15%)
        if ivs.skew_25d > 0.01:
            score += 0.15 * min(ivs.skew_25d / 0.06, 1.0)

        return round(min(score, 1.0), 4)

    def sell_score(self, data: StockData, entry_price: float) -> float:
        """Sell when vol is expensive or exhaustion signals appear."""
        score = 0.0
        ivs = data.iv_surface

        # 1. IV spike: IV >> RV (expensive premium) (30%)
        if ivs.iv_atm > 0 and ivs.rv_20d > 0:
            spread = ivs.iv_rv_spread / ivs.rv_20d
            if spread > 0.2:
                score += 0.30 * min(spread / 0.5, 1.0)

        # 2. RV expanding fast (exhaustion coming) (25%)
        regime = self._vol_regime(data)
        if regime == "expanding":
            score += 0.25

        # 3. Term structure backwardation (front >> back): event/fear (25%)
        if ivs.iv_term_slope > 0.03:
            score += 0.25 * min(ivs.iv_term_slope / 0.08, 1.0)

        # 4. Skew collapse (puts getting cheap): complacency (20%)
        if ivs.skew_25d < -0.01:
            score += 0.20 * min(abs(ivs.skew_25d) / 0.05, 1.0)

        return round(min(score, 1.0), 4)

    def swing_score(self, data: StockData) -> float:
        """How strong is the case for holding overnight?
        Returns 0-1: higher = stronger swing signal.
        """
        ivs = data.iv_surface
        score = 0.0

        # Low IV + compressing vol = coiled spring, worth holding
        if ivs.iv_atm > 0 and ivs.rv_20d > 0:
            if ivs.iv_rv_spread < 0:
                score += 0.3

        if self._vol_regime(data) == "compressing":
            score += 0.3

        # Contango term structure = no near-term risk
        if ivs.iv_term_slope < -0.02:
            score += 0.2

        # Not near earnings
        if data.days_to_earnings > 7:
            score += 0.2

        return round(min(score, 1.0), 4)
